"""Configurable CuPy SFD routing, local-minima resolution, and accumulation.

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
"""

import math

from pyfastflow.core import Backend, HostBlockBuilder, KernelBuilder, SequenceBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flow import (
    depression_binding_plan,
    make_accumulation,
    make_depression_solver,
    make_depressions,
    make_fill_reconstruct,
    make_receivers,
)
from pyfastflow.graphflood._cupy_reconstruct_epsilon import build_hops_init, build_hops_jump
from pyfastflow.grid import make_grid_group, make_grid_parameters

BLOCK = 256


def _cupy_only(be: Backend) -> None:
    if be.name != "cupy":
        raise ValueError(f"experimental SFDFlowProgram is cupy-only, got {be.name!r}")


def _grid_leaf_plan(frozen, values):
    """Make an exact Program bind map from unique leaf names plus grid leaves."""
    bound = frozen.build()
    try:
        plan = {}
        for address in bound.addresses():
            if len(address) >= 2 and address[-2] == "grid":
                plan[".".join(address)] = "grid"
            elif address[-1] in values:
                plan[".".join(address)] = values[address[-1]]
        return plan
    finally:
        bound.close()


def _reconstruct_epsilon_factory(be, bundles, config):
    """Build reconstruction, epsilon distance, and parent-to-rec as one sequence."""
    _cupy_only(be)
    nx, ny = config["nx"], config["ny"]
    n = nx * ny
    max_passes = 4 * max(nx, ny)
    if n < max_passes + 2:
        raise ValueError("reconstruct_epsilon requires nx*ny >= 4*max(nx, ny)+2")

    deps = make_fill_reconstruct(be, bundles["grid"], nx=nx, ny=ny)
    pass_p, active_p = bundles.param("pass_index"), bundles.param("active")

    def zero_pass(ctx): ctx.P.set(0)
    def bump_pass(ctx): ctx.P.set(int(ctx.P.read()) + 1)
    def zero_active(ctx): ctx.ACTIVE.set(0)
    def converged(ctx): return int(ctx.ACTIVE.read()) == 0

    reset = KernelBuilder(f'''extern "C" __global__ void reset_reconstruct(
            int* counters, int* queued_gen) {{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= {n}) return;
        counters[i] = 0;
        queued_gen[i] = -1;
    }}''', domain=n).freeze()
    copy_parent = KernelBuilder(f'''extern "C" __global__ void copy_parent(
            const int* parent, int* rec) {{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < {n}) rec[i] = parent[i];
    }}''', domain=n).freeze()
    hops_init = build_hops_init(n_flat=n)
    hops_jump = build_hops_jump(n_flat=n)
    hops_rounds = math.ceil(math.log2(max(2, n))) + 1
    if hops_rounds % 2: hops_rounds += 1

    sb = SequenceBuilder()
    sb.add("reset", reset)
    for name in ("init_filled", "sweep_row_lr", "sweep_row_rl", "sweep_col_tb", "sweep_col_bt", "frontier_init"):
        sb.add(name, deps[name])
    sb.add("zero_pass", HostBlockBuilder(zero_pass).freeze())
    sb.add("zero_active", HostBlockBuilder(zero_active).freeze())
    sb.add("relax", deps["relax"])
    sb.add("bump_pass", HostBlockBuilder(bump_pass).freeze())
    sb.add("converged", HostBlockBuilder(converged).freeze())
    sb.add("hops_init", hops_init)
    sb.add("hops_forward", hops_jump)
    sb.add("hops_backward", hops_jump)
    sb.add("copy_parent", copy_parent)
    for name in ("reset", "init_filled", "sweep_row_lr", "sweep_row_rl", "sweep_col_tb", "sweep_col_bt", "frontier_init", "zero_pass"):
        sb.step(name)
    sb.loop(body=("zero_active", "relax", "bump_pass"), max_times=max_passes, until="converged")
    sb.step("hops_init")
    sb.loop(body=("hops_forward", "hops_backward"), max_times=hops_rounds // 2)
    sb.step("copy_parent")
    return sb.freeze()


def _reconstruct_epsilon_plan(frozen, _be):
    values = {
        "z": "z", "filled": "filled", "parent": "parent",
        "frontier": "frontier", "counters": "counters",
        "queued_gen": "queued_gen", "active": "active.handle", "P": "pass_index",
        "ACTIVE": "active", "dist": "epsilon_distance", "anc": "epsilon_ancestor",
        "dist_in": "epsilon_distance", "dist_out": "epsilon_distance_work",
        "anc_in": "epsilon_ancestor", "anc_out": "epsilon_ancestor_work",
        "rec": "rec",
    }
    plan = _grid_leaf_plan(frozen, values)
    # Correct the mirrored ping-pong addresses that leaf-only mapping cannot distinguish.
    plan.update({
        "hops_forward.dist_in": "epsilon_distance", "hops_forward.dist_out": "epsilon_distance_work",
        "hops_forward.anc_in": "epsilon_ancestor", "hops_forward.anc_out": "epsilon_ancestor_work",
        "hops_backward.dist_in": "epsilon_distance_work", "hops_backward.dist_out": "epsilon_distance",
        "hops_backward.anc_in": "epsilon_ancestor_work", "hops_backward.anc_out": "epsilon_ancestor",
    })
    return plan


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
    b.config(
        "local_minima",
        choices=("reconstruct_epsilon", "cordonnier_carve", "cordonnier_jump"),
        default="cordonnier_carve",
    )
    b.config(
        "accumulation",
        choices=("rake_compress", "pointer_jump_push", "pj"),
        default="pointer_jump_push",
    )
    b.config("dx", default=1.0)
    b.param("source", "scalar", "f32", value=1.0)
    b.param("ndep", "scalar", "i32", value=0)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)
    b.param("rake_iteration", "scalar", "i32", value=0)

    flat = Dim("ny") * Dim("nx")
    b.data("z", "f32", (Dim("ny"), Dim("nx")), role="input", shape_source=True)
    b.data("rec", "i32", (Dim("ny"), Dim("nx")), role="output")
    b.data("drainage", "f32", (Dim("ny"), Dim("nx")), role="output")
    b.data("filled", "f32", (Dim("ny"), Dim("nx")), role="output")
    b.data("epsilon_distance", "f32", (Dim("ny"), Dim("nx")), role="output")

    def grid_structure(be, **_):
        _cupy_only(be); return make_grid_group(be, topology="D8", boundary="normal", outlet="edge")
    def grid_params(be, pool, *, nx, ny, dx):
        return make_grid_parameters(be, pool, nx, ny, dx, topology="D8", outlet="edge")
    b.bundle("grid", grid_structure, grid_params, dims=("nx", "ny"), config=("dx",))

    # Cordonnier and reconstruction storage. These stay owned until close().
    for name in ("bid", "rec_jump", "basin_saddlenode", "basin_route", "b_rcv", "parent", "epsilon_ancestor", "epsilon_ancestor_work"):
        b.data(name, "i32", (flat,), role="internal")
    for name in ("z_prime", "epsilon_distance_work"):
        b.data(name, "f32", (flat,), role="internal")
    for name in ("is_border", "rerouted"):
        b.data(name, "u8", (flat,), role="internal")
    for name in ("basin_saddle", "outlet"):
        b.data(name, "i64", (flat,), role="internal")
    b.data("frontier", "i32", (2 * flat,), role="internal")
    b.data("counters", "i32", (flat,), role="internal")
    b.data("queued_gen", "i32", (flat,), role="internal")

    # Choice-specific accumulation scratch is allocated lazily as Program temps.
    for name in ("pj_work", "pj_work2", "rake_src", "ndonors", "ndonors_alt"):
        b.data(name, "i32", (flat,), lifetime="temp")
    for name in ("pj_q_work", "rake_q_alt"):
        b.data(name, "f32", (flat,), lifetime="temp")
    for name in ("donors", "donors_alt"):
        b.data(name, "i32", (8 * flat,), lifetime="temp")

    def route_factory(be, bundles, _config):
        _cupy_only(be); return make_receivers(be, bundles["grid"], topology="D8", mode="steepest")["receivers"]
    b.add("route", route_factory, bind={"grid": "grid", "z": "z", "rec": "rec"})

    def cordonnier_factory(reroute):
        def factory(be, bundles, config):
            _cupy_only(be); n = config["nx"] * config["ny"]
            deps = make_depressions(be, bundles["grid"], bundles.param("ndep"), method="optimized", reroute=reroute, n_flat=n)
            return make_depression_solver(be, deps, bundles.bundle_params("grid"), method="optimized", reroute=reroute, n_flat=n, block_size=BLOCK)[0]
        return factory
    b.add("resolve_carve", cordonnier_factory("carve"), bind=lambda f, be: depression_binding_plan(f, method="optimized", reroute="carve"))
    b.add("resolve_jump", cordonnier_factory("jump"), bind=lambda f, be: depression_binding_plan(f, method="optimized", reroute="jump"))
    b.add("resolve_reconstruct_epsilon", _reconstruct_epsilon_factory, bind=_reconstruct_epsilon_plan)
    b.dispatch("resolve_minima", on="local_minima", cases={
        "reconstruct_epsilon": "resolve_reconstruct_epsilon",
        "cordonnier_carve": "resolve_carve",
        "cordonnier_jump": "resolve_jump",
    })

    def pj_factory(be, bundles, config):
        return make_accumulation(be, bundles["grid"], method="pointer_jump_push", n_flat=config["nx"] * config["ny"])["sequence"].freeze()
    def rake_factory(be, bundles, config):
        return make_accumulation(be, bundles["grid"], method="rake_compress", n_flat=config["nx"] * config["ny"], n_neighbours=8)["sequence"].freeze()
    b.add("accumulate_pointer_jump", pj_factory, bind=_pointer_jump_plan)
    b.add("accumulate_rake_compress", rake_factory, bind=_rake_plan)
    b.dispatch("accumulate", on="accumulation", cases={
        "pointer_jump_push": "accumulate_pointer_jump", "pj": "accumulate_pointer_jump",
        "rake_compress": "accumulate_rake_compress",
    })
    return b.freeze()


SFDFlowProgram = build_sfd_flow_program()

__all__ = ["SFDFlowProgram", "build_sfd_flow_program"]
