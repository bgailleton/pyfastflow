"""CuPy SFD flow Program: routing, local-minima resolution, and accumulation.

Typical use::

    flow = SFDFlowProgram(
        Backend.from_name("cupy"),
        nx=2048,
        ny=2048,
        accumulation="pointer_jump_push",
        local_minima="cordonnier_carve",
        dx=30.0,
    )
    flow.z.from_numpy(dem.astype("float32"))
    flow.route()
    flow.resolve_minima()
    flow.accumulate()
    area = flow.drainage.array
    flow.close()

``local_minima="reconstruct_epsilon"`` derives the receiver graph directly
from reconstruction's acyclic parent links. It also exposes ``filled`` and
``epsilon_distance``; SFD itself does not need to re-derive receivers from
those two fields.

To route on a quadtree, pass it with its dimensions; every array then has
``tree.capacity`` entries on one row, and the tree stays open after ``close``::

    flow = SFDFlowProgram(be, tree=tree, topology=tree.topology, **tree.dims)
    flow.source.from_numpy(cell_area)  # shape (1, capacity); area per leaf, zero on spare entries

``dx``, ``boundary``, ``outlet``, ``nodata`` and ``dynamic_grid`` then belong to
the tree and must keep their defaults. ``reconstruct_epsilon`` and
``compute_slope_correction`` need the raster and are unavailable on a tree.
"""

import math

from pyfastflow.core import Backend, HostBlockBuilder, KernelBuilder, SequenceBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.core.context.slot import ProgramError
from pyfastflow.flow import (
    BOUNDARIES,
    LOCAL_MINIMA,
    RECEIVER_MODES,
    TOPOLOGIES,
    depression_binding_plan,
    make_accumulation,
    make_depression_solver,
    make_depressions,
    make_receivers,
)
from pyfastflow.flow._program_cupy import (
    BLOCK, _cupy_only, _noop_factory, _grid_leaf_plan,
    _reconstruct_epsilon_factory, _reconstruct_epsilon_plan,
)
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.grid._mask import GridMaskAccessor


def _pointer_jump_plan(_frozen, _be):
    return {
        "q_init.SOURCE": "source", "q_init.q": "drainage",
        "copy_rec_to_work.rec": "rec", "copy_rec_to_work.work": "pj_work",
        "step_a_copy.q_curr": "drainage", "step_a_copy.q_next": "pj_q_work",
        "step_a_core.rec_curr": "pj_work", "step_a_core.rec_next": "pj_work2",
        "step_a_core.q_curr": "drainage", "step_a_core.q_next": "pj_q_work",
        "step_b_copy.q_curr": "pj_q_work", "step_b_copy.q_next": "drainage",
        "step_b_core.rec_curr": "pj_work2", "step_b_core.rec_next": "pj_work",
        "step_b_core.q_curr": "pj_q_work", "step_b_core.q_next": "drainage",
    }


def _rake_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "rec": "rec", "q": "drainage", "SOURCE": "source", "ITER": "rake_iteration",
        "donors": "donors", "ndonors": "ndonors", "donors_alt": "donors_alt",
        "ndonors_alt": "ndonors_alt", "q_alt": "rake_q_alt", "src": "rake_src",
    })


def build_sfd_flow_program() -> type:
    """Build the configurable CuPy SFD flow Program class."""
    b = ProgramBuilder("SFDFlowProgram")
    b.dim("ny").dim("nx")
    b.config("ny")
    b.config("nx")
    b.config("local_minima", choices=LOCAL_MINIMA, default="cordonnier_carve")
    b.config(
        "accumulation",
        choices=("rake_compress", "pointer_jump_push", "pj"),
        default="pointer_jump_push",
    )
    b.config("dx", default=1.0)
    b.config("topology", choices=TOPOLOGIES, default="D8")
    b.config("boundary", choices=BOUNDARIES, default="normal")
    b.config("outlet", choices=("edge", "mask"), default="edge")
    b.config("nodata", choices=(False, True), default=False)
    b.config("dynamic_grid", choices=(False, True), default=False)
    b.config("tree", default=None)
    b.config("receiver_mode", choices=RECEIVER_MODES, default="steepest")
    b.param("receiver_seed", "auto", "i32", value=0)
    b.param("source", "auto", "f32", value=1.0,
            shape=(Dim("ny"), Dim("nx")))
    b.param("ndep", "scalar", "i32", value=0)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)
    b.param("rake_iteration", "scalar", "i32", value=0)

    flat = Dim("ny") * Dim("nx")
    b.data("z", "f32", (Dim("ny"), Dim("nx")), role="input", shape_source=True)
    b.data("boundary_z", "f32", (Dim("ny"), Dim("nx")), role="input")
    b.data("rec", "i32", (Dim("ny"), Dim("nx")), role="output")
    b.data("drainage", "f32", (Dim("ny"), Dim("nx")), role="output")
    b.data("slope_correction", "f32", (Dim("ny"), Dim("nx")), role="output")
    b.data("filled", "f32", (Dim("ny"), Dim("nx")), role="output")
    b.data("epsilon_distance", "f32", (Dim("ny"), Dim("nx")), role="output")

    def grid_structure(be, *, nx, ny, dx, topology, boundary, outlet, nodata,
                       dynamic_grid, tree):
        _cupy_only(be)
        if tree is not None:
            if (nx, ny) != (tree.capacity, 1):
                raise ProgramError(f"tree: dims must be tree.dims {tree.dims}, got nx={nx}, ny={ny}")
            if topology != tree.topology:
                raise ProgramError(f"tree: topology {topology!r} differs from the tree's {tree.topology!r}")
            if (dx, boundary, outlet, nodata, dynamic_grid) != (1.0, "normal", "edge", False, False):
                raise ProgramError("tree: dx, boundary, outlet, nodata and dynamic_grid belong to the tree")
            return tree.group
        return make_grid_group(be, topology=topology, boundary=boundary,
                               outlet=outlet, nodata=nodata)
    def grid_params(be, pool, *, nx, ny, dx, topology, boundary, outlet, nodata,
                    dynamic_grid, tree):
        del boundary
        if tree is not None:
            return tree.parameters
        return make_grid_parameters(be, pool, nx, ny, dx, topology=topology,
                                    outlet=outlet, nodata=nodata,
                                    nx_mode="scalar" if dynamic_grid else "const",
                                    ny_mode="scalar" if dynamic_grid else "const",
                                    dx_mode="scalar" if dynamic_grid else "const")
    b.bundle("grid", grid_structure, grid_params, dims=("nx", "ny"),
             config=("dx", "topology", "boundary", "outlet", "nodata",
                     "dynamic_grid", "tree"))

    # Cordonnier and reconstruction storage. These stay owned until close().
    for name in ("bid", "rec_jump", "basin_saddlenode", "basin_route", "b_rcv", "parent", "epsilon_ancestor", "epsilon_ancestor_work"):
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

    # Choice-specific accumulation scratch is allocated lazily as Program temps.
    for name in ("pj_work", "pj_work2", "rake_src", "ndonors", "ndonors_alt"):
        b.data(name, "i32", (flat,), lifetime="temp")
    for name in ("pj_q_work", "rake_q_alt"):
        b.data(name, "f32", (flat,), lifetime="temp")
    def donor_shape(values):
        tree = values["tree"]
        slots = 8 if tree is None else tree.n_neighbours
        return (slots * values["ny"] * values["nx"],)
    for name in ("donors", "donors_alt"):
        b.data(name, "i32", donor_shape, lifetime="temp")

    def route_factory(mode):
        def factory(be, bundles, config):
            _cupy_only(be)
            return make_receivers(be, bundles["grid"],
                                  topology=config["topology"], mode=mode)["receivers"]
        return factory
    for mode in RECEIVER_MODES:
        binding = {"grid": "grid", "z": "z", "rec": "rec"}
        if mode != "steepest":
            binding["rand_unit.SEED"] = "receiver_seed"
        b.add(f"route_{mode}", route_factory(mode), bind=binding)
    b.dispatch("route", on="receiver_mode", cases={
        mode: f"route_{mode}" for mode in RECEIVER_MODES
    })

    def slope_correction_factory(_be, bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void sfd_slope_correction(
    const float* z, const int* rec, float* correction,
    const float* min_slope, const float* max_correction) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.grid.n()$) return;
    correction[i] = 1.0f;
    int r = rec[i];
    if (r == i || !$ctx.grid.is_active(i)$) return;
    float length = $ctx.grid.dist_between_nodes(i, r)$;
    float drop = z[i] - z[r];
    if (!(length > 0.0f) || !(drop > 0.0f) || !isfinite(drop)) return;
    // Fit unsigned x/y gradient components to all downhill neighbour slopes.
    // D4 recovers the cardinal estimate; D8 also incorporates diagonals.
    float axx = 0.0f, axy = 0.0f, ayy = 0.0f;
    float bx = 0.0f, by = 0.0f, strongest = 0.0f;
    for (int k = 0; k < $ctx.grid.N_NEIGHBOURS.get(0)$; ++k) {{
        int j = $ctx.grid.neighbour(i, k)$;
        if (j < 0 || !$ctx.grid.is_active(j)$) continue;
        float dz = z[i] - z[j];
        if (!(dz > 0.0f) || !isfinite(dz)) continue;
        int dr, dc;
        $ctx.grid.delta(k, &dr, &dc)$;
        float norm = sqrtf((float)(dr * dr + dc * dc));
        float ux = abs(dc) / norm, uy = abs(dr) / norm;
        float s = dz / $ctx.grid.dist(i, k)$;
        axx += ux * ux; axy += ux * uy; ayy += uy * uy;
        bx += ux * s; by += uy * s;
        strongest = fmaxf(strongest, s);
    }}
    float determinant = axx * ayy - axy * axy;
    float gradient = strongest;
    if (determinant > 1.0e-6f) {{
        float gx = fmaxf((bx * ayy - by * axy) / determinant, 0.0f);
        float gy = fmaxf((by * axx - bx * axy) / determinant, 0.0f);
        gradient = hypotf(gx, gy);
    }}
    if (!(gradient > 0.0f) || !isfinite(gradient)) return;
    float link_slope = fmaxf(drop / length, min_slope[0]);
    correction[i] = fmaxf(fminf(gradient / link_slope, max_correction[0]),
                          1.0e-6f);
}}''', domain=n).compose("grid", bundles["grid"]).freeze()
    b.param("min_link_slope", "scalar", "f32", value=1.0e-6)
    b.param("max_slope_correction", "scalar", "f32", value=100.0)
    b.add("compute_slope_correction", slope_correction_factory, bind={
        "grid": "grid", "z": "z", "rec": "rec",
        "correction": "slope_correction",
        "min_slope": "min_link_slope.handle",
        "max_correction": "max_slope_correction.handle",
    })

    def clear_inactive_factory(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void sfd_clear_inactive(
    int* rec, const int* active_n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {n} && i >= active_n[0]) rec[i] = i;
}}''', domain=n).freeze()
    b.param("active_n", "scalar", "i32", value=0)
    b.add("clear_inactive_receivers", clear_inactive_factory,
          bind={"rec": "rec", "active_n": "active_n.handle"})

    def restore_outlets_factory(_be, bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void sfd_restore_outlets(
    float* z, const float* boundary_z) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.grid.n()$) return;
    if ($ctx.grid.can_out(i)$) z[i] = boundary_z[i];
}}''', domain=n).compose("grid", bundles["grid"]).freeze()
    b.add("restore_outlets", restore_outlets_factory,
          bind={"grid": "grid", "z": "z", "boundary_z": "boundary_z"})

    def cordonnier_factory(reroute):
        def factory(be, bundles, config):
            _cupy_only(be); n = config["nx"] * config["ny"]
            deps = make_depressions(be, bundles["grid"], bundles.param("ndep"), method="optimized", reroute=reroute, n_flat=n)
            return make_depression_solver(be, deps, bundles.bundle_params("grid"), method="optimized", reroute=reroute, n_flat=n, block_size=BLOCK)[0]
        return factory
    def depression_plan(frozen, reroute):
        plan = depression_binding_plan(frozen, method="optimized", reroute=reroute)
        return {key: "basin_outlet" if value == "outlet" else value
                for key, value in plan.items()}
    b.add("resolve_carve", cordonnier_factory("carve"),
          bind=lambda f, be: depression_plan(f, "carve"))
    b.add("resolve_jump", cordonnier_factory("jump"),
          bind=lambda f, be: depression_plan(f, "jump"))
    b.add("resolve_reconstruct_epsilon", _reconstruct_epsilon_factory, bind=_reconstruct_epsilon_plan)
    b.add("resolve_none", _noop_factory, bind={})
    b.dispatch("resolve_minima", on="local_minima", cases={
        "none": "resolve_none",
        "reconstruct_epsilon": "resolve_reconstruct_epsilon",
        "cordonnier_carve": "resolve_carve",
        "cordonnier_jump": "resolve_jump",
    })

    def pj_factory(be, bundles, config):
        return make_accumulation(be, bundles["grid"], method="pointer_jump_push", n_flat=config["nx"] * config["ny"])["sequence"].freeze()
    def rake_factory(be, bundles, config):
        tree = config["tree"]
        nk = tree.n_neighbours if tree is not None else (8 if config["topology"] == "D8" else 4)
        return make_accumulation(be, bundles["grid"], method="rake_compress",
                                 n_flat=config["nx"] * config["ny"], n_neighbours=nk)["sequence"].freeze()
    b.add("accumulate_pointer_jump", pj_factory, bind=_pointer_jump_plan)
    b.add("accumulate_rake_compress", rake_factory, bind=_rake_plan)
    b.dispatch("accumulate", on="accumulation", cases={
        "pointer_jump_push": "accumulate_pointer_jump", "pj": "accumulate_pointer_jump",
        "rake_compress": "accumulate_rake_compress",
    })
    program = b.freeze()
    program.outlet_mask = property(
        lambda self: GridMaskAccessor(self, "OUTLET_MASK", "outlet='mask'"),
    )
    program.nodata_mask = property(
        lambda self: GridMaskAccessor(self, "NODATA_MASK", "nodata=True"),
    )
    return program


SFDFlowProgram = build_sfd_flow_program()

__all__ = ["SFDFlowProgram", "build_sfd_flow_program"]
