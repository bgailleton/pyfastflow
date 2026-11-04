"""Assembly of the sediment-free, CuPy-only GOLEM program."""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flow import (
    depression_binding_plan,
    make_accumulation,
    make_depression_solver,
    make_depressions,
    make_receivers,
)
from pyfastflow.flow._program_cupy import (
    BLOCK,
    _cupy_only,
    _grid_leaf_plan,
    _noop_factory,
    _reconstruct_epsilon_factory,
    _reconstruct_epsilon_plan,
)
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.grid._mask import GridMaskAccessor

from ._fluvial import fluvial_factory, fluvial_plan
from ._hillslope import hillslope_factory, hillslope_plan


def _pointer_jump_plan(_frozen, _be):
    return {
        "q_init.SOURCE": "source", "q_init.q": "drainage_area",
        "copy_rec_to_work.rec": "rec", "copy_rec_to_work.work": "pj_work",
        "step_a_copy.q_curr": "drainage_area", "step_a_copy.q_next": "pj_q_work",
        "step_a_core.rec_curr": "pj_work", "step_a_core.rec_next": "pj_work2",
        "step_a_core.q_curr": "drainage_area", "step_a_core.q_next": "pj_q_work",
        "step_b_copy.q_curr": "pj_q_work", "step_b_copy.q_next": "drainage_area",
        "step_b_core.rec_curr": "pj_work2", "step_b_core.rec_next": "pj_work",
        "step_b_core.q_curr": "pj_q_work", "step_b_core.q_next": "drainage_area",
    }


def build_golem_nosed_program() -> type:
    """Build the experimental sediment-free GOLEM Program type.

    ``z`` is the bare rock surface.  ``initialize`` captures its outlet
    elevations; subsequently every operator respects those fixed values.
    ``run_n_step`` performs uplift, SFD routing/accumulation, implicit
    stream-power incision, and implicit linear hillslope diffusion.
    """
    b = ProgramBuilder("GolemNoSedProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("dx", default=1.0)
    b.config("topology", choices=("D4", "D8"), default="D8")
    b.config(
        "boundary", choices=("normal", "periodic_EW", "periodic_NS"),
        default="normal",
    )
    b.config("outlet", choices=("edge", "mask"), default="edge")
    b.config("nodata", choices=(False, True), default=False)
    b.config(
        "local_minima",
        choices=("none", "reconstruct_epsilon", "cordonnier_carve", "cordonnier_jump"),
        default="cordonnier_carve",
    )
    b.config("diffusion_iterations", default=20)
    b.config("newton_iterations", default=8)

    shape = (Dim("ny"), Dim("nx"))
    flat = Dim("ny") * Dim("nx")
    # ``auto`` stores a uniform parameter as a constant and accepts a full-grid
    # field when spatially variable physics is needed.
    b.param("uplift", "auto", "f32", value=0.0, shape=shape)
    b.param("erodibility", "auto", "f32", value=0.0, shape=shape)
    b.param("hillslope_diffusivity", "auto", "f32", value=0.0, shape=shape)
    b.param("dt", "auto", "f32", value=1.0, shape=shape)
    b.param("m", "auto", "f32", value=0.4, shape=shape)
    b.param("n", "auto", "f32", value=1.0, shape=shape)
    b.param("slope_floor", "auto", "f32", value=1.0e-6, shape=shape)
    b.param("source", "auto", "f32", value=1.0, shape=shape)
    b.param("ndep", "scalar", "i32", value=0)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)

    b.data("z", "f32", shape, role="state", shape_source=True)
    b.data("outlet_z", "f32", shape, role="state")
    b.data("rec", "i32", shape, role="output")
    b.data("drainage_area", "f32", shape, role="output")
    b.data("erosion_rate", "f32", shape, role="output")
    b.data("filled", "f32", shape, role="output")
    b.data("epsilon_distance", "f32", shape, role="output")

    def grid_structure(be, *, topology, boundary, outlet, nodata, **_):
        _cupy_only(be)
        return make_grid_group(
            be, topology=topology, boundary=boundary, outlet=outlet,
            nodata=nodata,
        )

    def grid_params(be, pool, *, nx, ny, dx, topology, boundary, outlet, nodata):
        del boundary
        return make_grid_parameters(
            be, pool, nx, ny, dx, topology=topology, outlet=outlet,
            nodata=nodata,
        )

    b.bundle("grid", grid_structure, grid_params, dims=("nx", "ny"),
             config=("dx", "topology", "boundary", "outlet", "nodata"))

    # Persistent depression-reconstruction storage mirrors SFDFlowProgram.
    for name in (
        "bid", "rec_jump", "basin_saddlenode", "basin_route", "b_rcv",
        "parent", "epsilon_ancestor", "epsilon_ancestor_work",
    ):
        b.data(name, "i32", (flat,), role="internal")
    for name in ("z_prime", "epsilon_distance_work"):
        b.data(name, "f32", (flat,), role="internal")
    for name in ("is_border", "rerouted"):
        b.data(name, "u8", (flat,), role="internal")
    for name in ("basin_saddle", "basin_outlet"):
        b.data(name, "i64", (flat,), role="internal")
    b.data("frontier", "i32", (2 * flat,), role="internal")
    b.data("counters", "i32", (flat,), role="internal")
    b.data("queued_gen", "i32", (flat,), role="internal")

    # Solver workspaces are temporary and return to the program pool after
    # their operation finishes.
    for name in ("pj_work", "pj_work2", "fluvial_rec_a", "fluvial_rec_b"):
        b.data(name, "i32", (flat,), lifetime="temp")
    for name in (
        "pj_q_work", "fluvial_z0", "fluvial_guess", "fluvial_z_a",
        "fluvial_z_b", "fluvial_alpha_a", "fluvial_alpha_b",
        "diffusion_z0", "diffusion_a", "diffusion_b",
    ):
        b.data(name, "f32", (flat,), lifetime="temp")

    def route_factory(mode):
        def factory(be, bundles, config):
            _cupy_only(be)
            return make_receivers(
                be, bundles["grid"], topology=config["topology"], mode=mode,
            )["receivers"]
        return factory

    b.add("route", route_factory("steepest"), bind={
        "grid": "grid", "z": "z", "rec": "rec",
    })

    def cordonnier_factory(reroute):
        def factory(be, bundles, config):
            _cupy_only(be)
            deps = make_depressions(
                be, bundles["grid"], bundles.param("ndep"),
                method="optimized", reroute=reroute,
                n_flat=config["nx"] * config["ny"],
            )
            return make_depression_solver(
                be, deps, bundles.bundle_params("grid"), method="optimized",
                reroute=reroute, n_flat=config["nx"] * config["ny"],
                block_size=BLOCK,
            )[0]
        return factory

    def depression_plan(frozen, reroute):
        plan = depression_binding_plan(frozen, method="optimized", reroute=reroute)
        return {key: "basin_outlet" if value == "outlet" else value
                for key, value in plan.items()}

    b.add("resolve_carve", cordonnier_factory("carve"),
          bind=lambda f, be: depression_plan(f, "carve"))
    b.add("resolve_jump", cordonnier_factory("jump"),
          bind=lambda f, be: depression_plan(f, "jump"))
    b.add("resolve_reconstruct_epsilon", _reconstruct_epsilon_factory,
          bind=_reconstruct_epsilon_plan)
    b.add("resolve_none", _noop_factory, bind={})
    b.dispatch("resolve_minima", on="local_minima", cases={
        "none": "resolve_none",
        "reconstruct_epsilon": "resolve_reconstruct_epsilon",
        "cordonnier_carve": "resolve_carve",
        "cordonnier_jump": "resolve_jump",
    })

    def accumulation_factory(be, bundles, config):
        _cupy_only(be)
        return make_accumulation(
            be, bundles["grid"], method="pointer_jump_push",
            n_flat=config["nx"] * config["ny"],
        )["sequence"].freeze()

    b.add("accumulate", accumulation_factory, bind=_pointer_jump_plan)

    def scale_area_factory(be, bundles, config):
        _cupy_only(be)
        n_cells = config["nx"] * config["ny"]
        return KernelBuilder(
            f'''extern "C" __global__ void golem_scale_drainage_area(float* area) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= {n_cells}) return;
                if (!$ctx.grid.is_active(i)$) {{ area[i] = 0.0f; return; }}
                float dx = $ctx.grid.dx(i)$;
                area[i] *= dx * dx;
            }}''', domain=n_cells,
        ).compose("grid", bundles["grid"]).freeze()

    b.add("scale_drainage_area", scale_area_factory,
          bind={"grid": "grid", "area": "drainage_area"})

    def uplift_factory(be, bundles, config):
        _cupy_only(be)
        n_cells = config["nx"] * config["ny"]
        return KernelBuilder(
            f'''extern "C" __global__ void golem_apply_uplift(float* z) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= {n_cells} || !$ctx.grid.is_active(i)$ || $ctx.grid.can_out(i)$)
                    return;
                z[i] += $ctx.UPLIFT.get(i)$ * fmaxf($ctx.DT.get(i)$, 0.0f);
            }}''', domain=n_cells,
        ).compose("grid", bundles["grid"]).freeze()

    b.add("apply_uplift", uplift_factory,
          bind={"grid": "grid", "z": "z", "UPLIFT": "uplift", "DT": "dt"})

    def capture_outlets_factory(be, bundles, config):
        _cupy_only(be)
        n_cells = config["nx"] * config["ny"]
        return KernelBuilder(
            f'''extern "C" __global__ void golem_capture_outlets(
                    const float* z, float* outlet_z) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < {n_cells} && $ctx.grid.can_out(i)$) outlet_z[i] = z[i];
            }}''', domain=n_cells,
        ).compose("grid", bundles["grid"]).freeze()

    def restore_outlets_factory(be, bundles, config):
        _cupy_only(be)
        n_cells = config["nx"] * config["ny"]
        return KernelBuilder(
            f'''extern "C" __global__ void golem_restore_outlets(
                    float* z, const float* outlet_z) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < {n_cells} && $ctx.grid.can_out(i)$) z[i] = outlet_z[i];
            }}''', domain=n_cells,
        ).compose("grid", bundles["grid"]).freeze()

    b.add("capture_outlet_z", capture_outlets_factory,
          bind={"grid": "grid", "z": "z", "outlet_z": "outlet_z"})
    b.add("restore_outlet_z", restore_outlets_factory,
          bind={"grid": "grid", "z": "z", "outlet_z": "outlet_z"})
    b.add("fluvial", fluvial_factory, bind=fluvial_plan)
    b.add("hillslope", hillslope_factory, bind=hillslope_plan)

    b.pipeline("initialize", ("capture_outlet_z", "restore_outlet_z"))
    b.pipeline("route_and_accumulate", (
        "route", "resolve_minima", "accumulate", "scale_drainage_area",
    ))
    b.pipeline("run_n_step", (
        "apply_uplift", "restore_outlet_z",
        "route", "resolve_minima", "accumulate", "scale_drainage_area",
        "fluvial", "restore_outlet_z", "hillslope", "restore_outlet_z",
    ))

    program = b.freeze()
    program.outlet_mask = property(
        lambda self: GridMaskAccessor(self, "OUTLET_MASK", "outlet='mask'"),
    )
    program.nodata_mask = property(
        lambda self: GridMaskAccessor(self, "NODATA_MASK", "nodata=True"),
    )
    return program


GolemNoSedProgram = build_golem_nosed_program()

__all__ = ["GolemNoSedProgram", "build_golem_nosed_program"]
