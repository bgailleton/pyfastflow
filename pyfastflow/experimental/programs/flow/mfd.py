"""CuPy MFD routing, optional local-minima resolution, and accumulation.

The public pipeline is deliberately small::

    flow = MFDFlowProgram(backend, nx=nx, ny=ny,
                          local_minima="cordonnier_carve")
    flow.z.from_numpy(dem)
    flow.route()
    flow.snapshot_receivers()
    flow.resolve_minima()
    flow.prepare_mfd_surface()
    flow.build_topology()
    flow.accumulate()

``local_minima="none"`` makes the first call a true no-op. Its topology is
the raw downslope MFD graph, so flow may terminate in interior sinks.
Cordonnier carve uses receiver-rank gating. ``fill_cordonnier`` instead derives
a filled elevation from the carved paths, then runs ordinary MFD on that field
with receiver distance only as an epsilon flat ordering. Reconstruction uses
its own filled surface and path-distance tie breaker. ``quantized_weight=True``
(the default) stores eight max-normalized unsigned-byte scores per cell;
accumulation renormalizes their sum so the complete discharge is still
partitioned.
"""

from pyfastflow.core import Backend, HostBlockBuilder, KernelBuilder, SequenceBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flow import (
    depression_binding_plan,
    make_accumulation,
    make_depression_solver,
    make_depressions,
    make_mfd_topology,
    make_receivers,
)
from pyfastflow.grid import make_grid_group, make_grid_parameters

from .sfd import (
    BLOCK,
    _cupy_only,
    _grid_leaf_plan,
    _noop_factory,
    _reconstruct_epsilon_factory,
    _reconstruct_epsilon_plan,
)


def _cordonnier_factory(reroute):
    def factory(be, bundles, config):
        _cupy_only(be)
        n = config["nx"] * config["ny"]
        deps = make_depressions(
            be, bundles["grid"], bundles.param("ndep"), method="optimized",
            reroute=reroute, n_flat=n,
        )
        return make_depression_solver(
            be, deps, bundles.bundle_params("grid"), method="optimized",
            reroute=reroute, n_flat=n, block_size=BLOCK,
        )[0]

    return factory


def _route_factory(be, bundles, _config):
    _cupy_only(be)
    return make_receivers(
        be, bundles["grid"], topology="D8", mode="steepest"
    )["receivers"]


def _snapshot_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank",
        n_flat=config["nx"] * config["ny"], topology="D8",
        diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )["snapshot_receivers"]


def _rank_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank",
        n_flat=config["nx"] * config["ny"], topology="D8",
        diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )["receiver_rank"]


def _cordonnier_fill_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_fill",
        n_flat=config["nx"] * config["ny"], topology="D8",
        diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )["receiver_fill"]


def _rank_plan(_frozen, _be):
    return {
        "init.rec": "rec", "init.ancestor": "rank_ancestor", "init.rank": "rank",
        "forward.ancestor_in": "rank_ancestor", "forward.rank_in": "rank",
        "forward.ancestor_out": "rank_ancestor_alt", "forward.rank_out": "rank_alt",
        "backward.ancestor_in": "rank_ancestor_alt", "backward.rank_in": "rank_alt",
        "backward.ancestor_out": "rank_ancestor", "backward.rank_out": "rank",
    }


def _cordonnier_fill_plan(_frozen, _be):
    return {
        "init.rec": "rec", "init.z": "z",
        "init.ancestor": "rank_ancestor", "init.rank": "rank",
        "init.filled": "filled",
        "forward.ancestor_in": "rank_ancestor",
        "forward.rank_in": "rank", "forward.filled_in": "filled",
        "forward.ancestor_out": "rank_ancestor_alt",
        "forward.rank_out": "rank_alt", "forward.filled_out": "z_prime",
        "backward.ancestor_in": "rank_ancestor_alt",
        "backward.rank_in": "rank_alt", "backward.filled_in": "z_prime",
        "backward.ancestor_out": "rank_ancestor",
        "backward.rank_out": "rank", "backward.filled_out": "filled",
    }


def _raw_topology_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    clear = KernelBuilder(
        f'''extern "C" __global__ void clear_flat_distance(float* dist) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) dist[i] = 0.0f;
        }}''',
        domain=n,
    ).freeze()
    topology = make_mfd_topology(
        be, bundles["grid"], method="surface", n_flat=n, topology="D8",
        diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )
    sb = SequenceBuilder()
    sb.add("clear_distance", clear)
    sb.add("dirs_weights", topology["dirs_weights"])
    sb.add("indegree_reset", topology["indegree_reset"])
    sb.add("indegree_count", topology["indegree_count"])
    sb.step("clear_distance").step("dirs_weights")
    sb.step("indegree_reset").step("indegree_count")
    return sb.freeze()


def _surface_topology_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    topology = make_mfd_topology(
        be, bundles["grid"], method="surface", n_flat=n, topology="D8",
        diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )
    sb = SequenceBuilder()
    for name in ("dirs_weights", "indegree_reset", "indegree_count"):
        sb.add(name, topology[name]).step(name)
    return sb.freeze()


def _rank_topology_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    topology = make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank", n_flat=n,
        topology="D8", diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )
    sb = SequenceBuilder()
    for name in ("dirs_weights", "indegree_reset", "indegree_count"):
        sb.add(name, topology[name]).step(name)
    return sb.freeze()


def _fill_topology_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    topology = make_mfd_topology(
        be, bundles["grid"], method="cordonnier_fill", n_flat=n,
        topology="D8", diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )
    sb = SequenceBuilder()
    for name in ("dirs_weights", "indegree_reset", "indegree_count"):
        sb.add(name, topology[name]).step(name)
    return sb.freeze()


def _topology_plan(kind):
    def plan(frozen, _be):
        values = {
            "dist": "epsilon_distance", "dirs": "directions",
            "mfd_w": "weights", "indegree": "indegree",
        }
        if kind == "raw":
            values.update({"filled": "z"})
        elif kind == "surface":
            values.update({"filled": "filled"})
        elif kind == "fill":
            values.update({"filled": "filled", "rank": "rank"})
        else:
            values.update({
                "z": "z", "rec_initial": "rec_initial", "rec": "rec",
                "rank": "rank",
            })
        return _grid_leaf_plan(frozen, values)

    return plan


def _accumulation_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    parts = make_accumulation(
        be, bundles["grid"], method="persistent_mfd", n_flat=n,
        n_neighbours=8,
        quantized_weight=config["quantized_weight"],
    )

    def prepare_frontier(ctx, indegree, frontier0, count, barrier):
        from pyfastflow.flow._cupy_mfd_accum import init_frontier_mfd

        ready = init_frontier_mfd(indegree.array, frontier0.array)
        count.array[0] = ready
        count.array[1] = 0
        barrier.array[0] = 0

    prepare = HostBlockBuilder(prepare_frontier).freeze()
    sb = SequenceBuilder()
    sb.add("q_init", parts["q_init"])
    sb.add("prepare_frontier", prepare)
    sb.add("accum", parts["accum"])
    sb.step("q_init").step("prepare_frontier").step("accum")
    return sb.freeze()


def _accumulation_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "SOURCE": "source", "accum": "drainage", "dirs": "directions",
        "mfd_w": "weights", "indegree": "indegree",
        "frontier0": "mfd_frontier0", "frontier1": "mfd_frontier1",
        "count": "mfd_count", "barrier": "mfd_barrier",
    })


def build_mfd_flow_program() -> type:
    """Build the reusable CuPy MFD flow Program class."""
    b = ProgramBuilder("MFDFlowProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx")
    b.config(
        "local_minima",
        choices=(
            "none", "reconstruct_epsilon", "cordonnier_carve",
            "fill_cordonnier",
        ),
        default="cordonnier_carve",
    )
    b.config("dx", default=1.0)
    b.config("quantized_weight", choices=(False, True), default=True)
    b.param("source", "scalar", "f32", value=1.0)
    b.param("ndep", "scalar", "i32", value=0)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)

    flat = Dim("ny") * Dim("nx")
    b.data("z", "f32", (Dim("ny"), Dim("nx")), role="input", shape_source=True)
    b.data("drainage", "f32", (Dim("ny"), Dim("nx")), role="output")
    b.data("filled", "f32", (Dim("ny"), Dim("nx")), role="output")
    b.data("rec", "i32", (Dim("ny"), Dim("nx")), role="output")

    def grid_structure(be, **_):
        _cupy_only(be)
        return make_grid_group(be, topology="D8", boundary="normal", outlet="edge")

    def grid_params(be, pool, *, nx, ny, dx):
        return make_grid_parameters(
            be, pool, nx, ny, dx, topology="D8", outlet="edge"
        )

    b.bundle("grid", grid_structure, grid_params, dims=("nx", "ny"), config=("dx",))

    for name in (
        "rec_initial", "rank_ancestor", "rank_ancestor_alt", "rank", "rank_alt",
        "bid", "rec_jump", "basin_saddlenode", "basin_route", "b_rcv",
        "parent", "epsilon_ancestor", "epsilon_ancestor_work", "mfd_frontier0",
        "mfd_frontier1", "indegree",
    ):
        b.data(name, "i32", (flat,), role="internal")
    for name in ("z_prime", "epsilon_distance", "epsilon_distance_work"):
        b.data(name, "f32", (flat,), role="internal")
    for name in ("is_border", "rerouted", "directions"):
        b.data(name, "u8", (flat,), role="internal")
    for name in ("basin_saddle", "outlet"):
        b.data(name, "i64", (flat,), role="internal")
    b.data(
        "weights",
        lambda config: "u8" if config["quantized_weight"] else "f32",
        (8 * flat,), role="internal",
    )
    b.data("frontier", "i32", (2 * flat,), role="internal")
    b.data("counters", "i32", (flat,), role="internal")
    b.data("queued_gen", "i32", (flat,), role="internal")
    b.data("mfd_count", "i32", (2,), role="internal")
    b.data("mfd_barrier", "u32", (1,), role="internal")

    b.add("route", _route_factory, bind={"grid": "grid", "z": "z", "rec": "rec"})
    b.add("snapshot_receivers", _snapshot_factory, bind={
        "rec": "rec", "rec_initial": "rec_initial",
    })
    b.add("compute_rank", _rank_factory, bind=_rank_plan)
    b.add("compute_cordonnier_fill", _cordonnier_fill_factory,
          bind=_cordonnier_fill_plan)
    b.add("resolve_none", _noop_factory, bind={})
    b.add("resolve_reconstruct_epsilon", _reconstruct_epsilon_factory, bind=_reconstruct_epsilon_plan)
    b.add("resolve_carve", _cordonnier_factory("carve"), bind=lambda f, be: depression_binding_plan(f, method="optimized", reroute="carve"))
    b.dispatch("resolve_minima", on="local_minima", cases={
        "none": "resolve_none",
        "reconstruct_epsilon": "resolve_reconstruct_epsilon",
        "cordonnier_carve": "resolve_carve",
        "fill_cordonnier": "resolve_carve",
    })
    b.dispatch("prepare_mfd_surface", on="local_minima", cases={
        "none": "resolve_none",
        "reconstruct_epsilon": "resolve_none",
        "cordonnier_carve": "compute_rank",
        "fill_cordonnier": "compute_cordonnier_fill",
    })

    b.add("topology_none", _raw_topology_factory, bind=_topology_plan("raw"))
    b.add("topology_reconstruct", _surface_topology_factory, bind=_topology_plan("surface"))
    b.add("topology_rank", _rank_topology_factory, bind=_topology_plan("rank"))
    b.add("topology_fill", _fill_topology_factory, bind=_topology_plan("fill"))
    b.dispatch("build_topology", on="local_minima", cases={
        "none": "topology_none",
        "reconstruct_epsilon": "topology_reconstruct",
        "cordonnier_carve": "topology_rank",
        "fill_cordonnier": "topology_fill",
    })
    b.add("accumulate", _accumulation_factory, bind=_accumulation_plan)
    return b.freeze()


MFDFlowProgram = build_mfd_flow_program()

__all__ = ["MFDFlowProgram", "build_mfd_flow_program"]
