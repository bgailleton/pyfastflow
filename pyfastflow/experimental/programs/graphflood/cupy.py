"""CuPy GraphFlood using Cordonnier carving and persistent MFD.

``h`` is persistent model state. Initialise it with ``h.from_numpy(...)``,
call ``reset_h()`` for a dry surface, or call ``initialize_h_from_fill()`` to
start with the reconstructed depression-storage depth before the first
``run_n_step(n)``. Between iterations, ``fill_hydraulic_surface()`` fills the
current ``z + h`` surface while preserving existing water depth.
Precipitation is a depth rate; the accumulation initialiser converts it to
per-cell discharge using the grid cell area. Max-normalized unsigned-byte MFD
scores are enabled by default and may be disabled with
``quantized_weight=False``. ``mfd_local_minima="rank_cordonnier"`` keeps the
carved receiver-rank gate; ``"fill_cordonnier"`` instead converts the carved
paths into a filled hydraulic surface and runs ordinary MFD on that surface;
``"reconstruct_epsilon"`` uses morphological reconstruction plus its epsilon
flat ordering and does not run Cordonnier.
"""

import math

from pyfastflow.core import KernelBuilder, RoutineBuilder, SequenceBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flow import (
    depression_binding_plan,
    make_accumulation,
    make_depression_solver,
    make_depressions,
    make_mfd_topology,
    make_receivers,
)
from pyfastflow.graphflood._cupy_friction import build_friction_velocity
from pyfastflow.grid import make_grid_group, make_grid_parameters

from ..flow.sfd import (
    BLOCK,
    _cupy_only,
    _grid_leaf_plan,
    _noop_factory,
    _reconstruct_epsilon_factory,
)


def _make_surface_factory(be, _bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void make_hydraulic_surface(
                const float* z, const float* h, float* surface) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) surface[i] = z[i] + h[i];
        }}''',
        domain=n,
    ).freeze()


def _route_factory(be, bundles, _config):
    _cupy_only(be)
    return make_receivers(
        be, bundles["grid"], topology="D8", mode="steepest",
        diagonal_partition_correction=False,
    )["receivers"]


def _snapshot_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank",
        n_flat=config["nx"] * config["ny"], topology="D8",
        quantized_weight=config["quantized_weight"],
    )["snapshot_receivers"]


def _carve_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    deps = make_depressions(
        be, bundles["grid"], bundles.param("ndep"), method="optimized",
        reroute="carve", n_flat=n,
    )
    return make_depression_solver(
        be, deps, bundles.bundle_params("grid"), method="optimized",
        reroute="carve", n_flat=n, block_size=BLOCK,
    )[0]


def _carve_plan(frozen, _be):
    plan = depression_binding_plan(frozen, method="optimized", reroute="carve")
    return {target: ("surface" if value == "z" else value)
            for target, value in plan.items()}


def _rank_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank",
        n_flat=config["nx"] * config["ny"], topology="D8",
        quantized_weight=config["quantized_weight"],
    )["receiver_rank"]


def _rank_plan(_frozen, _be):
    return {
        "init.rec": "rec", "init.ancestor": "rank_ancestor", "init.rank": "rank",
        "forward.ancestor_in": "rank_ancestor", "forward.rank_in": "rank",
        "forward.ancestor_out": "rank_ancestor_alt", "forward.rank_out": "rank_alt",
        "backward.ancestor_in": "rank_ancestor_alt", "backward.rank_in": "rank_alt",
        "backward.ancestor_out": "rank_ancestor", "backward.rank_out": "rank",
    }


def _cordonnier_fill_factory(be, _bundles, config):
    """Fill z+h by taking the path maximum along carved SFD receivers."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    init = KernelBuilder(
        f'''extern "C" __global__ void graphflood_fill_init(
                const float* surface, const int* rec,
                int* ancestor, int* rank, float* spill) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            ancestor[i] = rec[i];
            rank[i] = rec[i] == i ? 0 : 1;
            spill[i] = surface[i];
        }}''', domain=n,
    ).freeze()
    jump = KernelBuilder(
        f'''extern "C" __global__ void graphflood_fill_jump(
                const int* ancestor_in, const int* rank_in,
                const float* spill_in, int* ancestor_out, int* rank_out,
                float* spill_out) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            int a = ancestor_in[i];
            if (a == i) {{
                ancestor_out[i] = i;
                rank_out[i] = rank_in[i];
                spill_out[i] = spill_in[i];
            }} else {{
                ancestor_out[i] = ancestor_in[a];
                rank_out[i] = rank_in[i] + rank_in[a];
                spill_out[i] = fmaxf(spill_in[i], spill_in[a]);
            }}
        }}''', domain=n,
    ).freeze()
    apply = KernelBuilder(
        f'''extern "C" __global__ void graphflood_apply_fill(
                float* h, float* surface, const float* spill) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float filled = spill[i];
            float added_depth = filled - surface[i];
            if (added_depth > 0.0f) h[i] += added_depth;
            surface[i] = filled;
        }}''', domain=n,
    ).freeze()
    rounds = math.ceil(math.log2(max(2, n))) + 1
    if rounds % 2:
        rounds += 1
    return (SequenceBuilder()
            .add("init", init).add("forward", jump).add("backward", jump)
            .add("apply", apply).step("init")
            .loop(("forward", "backward"), max_times=rounds // 2)
            .step("apply").freeze())


def _cordonnier_fill_plan(_frozen, _be):
    return {
        "init.surface": "surface", "init.rec": "rec",
        "init.ancestor": "rank_ancestor", "init.rank": "rank",
        "init.spill": "z_prime",
        "forward.ancestor_in": "rank_ancestor",
        "forward.rank_in": "rank", "forward.spill_in": "z_prime",
        "forward.ancestor_out": "rank_ancestor_alt",
        "forward.rank_out": "rank_alt",
        "forward.spill_out": "steepest_slope",
        "backward.ancestor_in": "rank_ancestor_alt",
        "backward.rank_in": "rank_alt",
        "backward.spill_in": "steepest_slope",
        "backward.ancestor_out": "rank_ancestor",
        "backward.rank_out": "rank", "backward.spill_out": "z_prime",
        "apply.h": "h", "apply.surface": "surface",
        "apply.spill": "z_prime",
    }


def _build_hydraulic_topology(be, grid, n, quantized_weight):
    """Rank-gated MFD weights plus unmodified hydraulic slope diagnostics."""
    weight_type = "unsigned char" if quantized_weight else "float"
    max_decl = "float max_score = 0.0f;" if quantized_weight else ""
    max_forced = "max_score = 1.0f;" if quantized_weight else ""
    max_update = "if (score > max_score) max_score = score;" if quantized_weight else ""
    weight_write = (
        "weights[i * nk + k] = scores[k] > 0.0f "
        "? (unsigned char)max(1, __float2int_rn(255.0f * scores[k] / max_score)) : 0;"
        if quantized_weight else
        "weights[i * nk + k] = sum_score > 0.0f ? scores[k] / sum_score : 0.0f;"
    )
    dirs_weights = KernelBuilder(
        f'''extern "C" __global__ void graphflood_ranked_mfd(
                const float* surface, const int* rec_initial, const int* rec,
                const int* rank, unsigned char* dirs, {weight_type}* weights,
                float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            unsigned char mask = 0;
            float scores[8];
            float sum_score = 0.0f;
            {max_decl}
            float best_slope = 0.0f;
            float best_width = $ctx.grid.DX.get(0)$;
            for (int k = 0; k < 8; ++k) scores[k] = 0.0f;

            if (!$ctx.grid.can_out(i)$ && !$ctx.grid.nodata(i)$) {{
                // Manning uses the true hydraulic slope. MFD partitioning
                // separately uses slope times the link's effective width.
                for (int k = 0; k < nk; ++k) {{
                    int j = $ctx.grid.neighbour(i, k)$;
                    if (j == -1) continue;
                    float width = $ctx.grid.dist_from_k(k)$;
                    float slope = (surface[i] - surface[j]) / width;
                    if (slope > best_slope) {{
                        best_slope = slope;
                        best_width = width;
                    }}
                }}

                if (rec_initial[i] != rec[i]) {{
                    for (int k = 0; k < nk; ++k) {{
                        if ($ctx.grid.neighbour(i, k)$ == rec[i]) {{
                            mask = (unsigned char)(1u << k);
                            scores[k] = 1.0f;
                            sum_score = 1.0f;
                            {max_forced}
                            break;
                        }}
                    }}
                }} else {{
                    for (int k = 0; k < nk; ++k) {{
                        int j = $ctx.grid.neighbour(i, k)$;
                        if (j == -1 || rank[j] >= rank[i]) continue;
                        float width = $ctx.grid.dist_from_k(k)$;
                        float slope = (surface[i] - surface[j]) / width;
                        float score = slope > 0.0f ? slope * width : 0.0f;
                        scores[k] = score;
                        if (score > 0.0f) {{
                            mask |= (unsigned char)(1u << k);
                            sum_score += score;
                            {max_update}
                        }}
                    }}
                }}
            }}

            for (int k = 0; k < nk; ++k) {{ {weight_write} }}
            dirs[i] = mask;
            steepest_slope[i] = best_slope;
            flow_width[i] = best_width;
        }}''',
        domain=n,
    ).compose("grid", grid).freeze()

    generic = make_mfd_topology(
        be, grid,
        method="cordonnier_rank", n_flat=n, topology="D8",
        quantized_weight=quantized_weight,
    )
    return dirs_weights, generic["indegree_reset"], generic["indegree_count"]


def _build_filled_hydraulic_topology(be, grid, n, quantized_weight):
    """Ordinary MFD on a Cordonnier-filled surface with epsilon flat order."""
    weight_type = "unsigned char" if quantized_weight else "float"
    max_decl = "float max_score = 0.0f;" if quantized_weight else ""
    max_update = "if (score > max_score) max_score = score;" if quantized_weight else ""
    weight_write = (
        "weights[i * nk + k] = scores[k] > 0.0f "
        "? (unsigned char)max(1, __float2int_rn(255.0f * scores[k] / max_score)) : 0;"
        if quantized_weight else
        "weights[i * nk + k] = sum_score > 0.0f ? scores[k] / sum_score : 0.0f;"
    )
    dirs_weights = KernelBuilder(
        f'''extern "C" __global__ void graphflood_filled_mfd(
                const float* surface, const int* rank,
                unsigned char* dirs, {weight_type}* weights,
                float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            unsigned char mask = 0;
            float scores[8];
            float sum_score = 0.0f;
            {max_decl}
            float best_slope = 0.0f;
            float best_width = $ctx.grid.DX.get(0)$;
            for (int k = 0; k < 8; ++k) scores[k] = 0.0f;

            if (!$ctx.grid.can_out(i)$ && !$ctx.grid.nodata(i)$) {{
                float zi = surface[i];
                float ulp = nextafterf(zi, 1.0e30f) - zi;
                for (int k = 0; k < nk; ++k) {{
                    int j = $ctx.grid.neighbour(i, k)$;
                    if (j == -1) continue;
                    float drop = 0.0f;
                    if (zi > surface[j]) {{
                        drop = zi - surface[j];
                    }} else if (zi == surface[j] && rank[i] > rank[j]) {{
                        drop = ulp * (float)(rank[i] - rank[j]);
                    }}
                    if (drop <= 0.0f) continue;
                    float width = $ctx.grid.dist_from_k(k)$;
                    float slope = drop / width;
                    float score = slope * width;
                    scores[k] = score;
                    mask |= (unsigned char)(1u << k);
                    sum_score += score;
                    {max_update}
                    if (slope > best_slope) {{
                        best_slope = slope;
                        best_width = width;
                    }}
                }}
            }}

            for (int k = 0; k < nk; ++k) {{ {weight_write} }}
            dirs[i] = mask;
            steepest_slope[i] = best_slope;
            flow_width[i] = best_width;
        }}''', domain=n,
    ).compose("grid", grid).freeze()
    generic = make_mfd_topology(
        be, grid, method="cordonnier_rank", n_flat=n, topology="D8",
        quantized_weight=quantized_weight,
    )
    return dirs_weights, generic["indegree_reset"], generic["indegree_count"]


def _topology_factory(be, bundles, config):
    _cupy_only(be)
    dirs, reset, count = _build_hydraulic_topology(
        be, bundles["grid"], config["nx"] * config["ny"],
        config["quantized_weight"],
    )
    return (RoutineBuilder().step("dirs_weights", dirs)
            .step("indegree_reset", reset).step("indegree_count", count).freeze())


def _filled_topology_factory(be, bundles, config):
    _cupy_only(be)
    dirs, reset, count = _build_filled_hydraulic_topology(
        be, bundles["grid"], config["nx"] * config["ny"],
        config["quantized_weight"],
    )
    return (RoutineBuilder().step("dirs_weights", dirs)
            .step("indegree_reset", reset).step("indegree_count", count).freeze())


def _reconstructed_topology_factory(be, bundles, config):
    """MFD and Manning diagnostics on reconstruct+epsilon potential."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    topology = make_mfd_topology(
        be, bundles["grid"], method="surface", n_flat=n, topology="D8",
        diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )
    diagnostics = KernelBuilder(
        f'''extern "C" __global__ void graphflood_reconstruct_diagnostics(
                const float* filled, const float* dist,
                const unsigned char* dirs, float* steepest_slope,
                float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float best_slope = 0.0f;
            float best_width = $ctx.grid.DX.get(0)$;
            unsigned char mask = dirs[i];
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            for (int k = 0; k < nk; ++k) {{
                if (!(mask & (1u << k))) continue;
                int j = $ctx.grid.neighbour(i, k)$;
                float width = $ctx.grid.dist_from_k(k)$;
                float slope = ((filled[i] - filled[j])
                               + (dist[i] - dist[j])) / width;
                if (slope > best_slope) {{
                    best_slope = slope;
                    best_width = width;
                }}
            }}
            steepest_slope[i] = best_slope;
            flow_width[i] = best_width;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()
    return (RoutineBuilder()
            .step("dirs_weights", topology["dirs_weights"])
            .step("diagnostics", diagnostics)
            .step("indegree_reset", topology["indegree_reset"])
            .step("indegree_count", topology["indegree_count"]).freeze())


def _topology_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "surface": "surface", "rec_initial": "rec_initial", "rec": "rec",
        "rank": "rank", "dirs": "directions", "weights": "weights",
        "steepest_slope": "steepest_slope", "flow_width": "flow_width",
        "indegree": "indegree",
    })


def _filled_topology_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "surface": "surface", "rank": "rank", "dirs": "directions",
        "weights": "weights", "steepest_slope": "steepest_slope",
        "flow_width": "flow_width", "indegree": "indegree",
    })


def _reconstructed_topology_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "filled": "z_prime", "dist": "surface", "dirs": "directions",
        "mfd_w": "weights", "steepest_slope": "steepest_slope",
        "flow_width": "flow_width", "indegree": "indegree",
    })


def _frontier_factory(be, _bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    clear = KernelBuilder(
        '''extern "C" __global__ void graphflood_clear_frontier(
                int* count, unsigned int* barrier) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < 2) count[i] = 0;
            if (i == 0) barrier[0] = 0u;
        }''', domain=2,
    ).freeze()
    compact = KernelBuilder(
        f'''extern "C" __global__ void graphflood_compact_frontier(
                const int* indegree, int* frontier, int* count) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n} && indegree[i] == 0) {{
                int p = atomicAdd(&count[0], 1);
                frontier[p] = i;
            }}
        }}''', domain=n,
    ).freeze()
    return RoutineBuilder().step("clear", clear).step("compact", compact).freeze()


def _accumulation_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    q_init = KernelBuilder(
        f'''extern "C" __global__ void graphflood_q_init(float* Qi) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float dx = $ctx.grid.DX.get(0)$;
            Qi[i] = $ctx.grid.nodata(i)$ ? 0.0f
                : $ctx.PRECIPITATION.get(0)$ * dx * dx;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()
    accum = make_accumulation(
        be, bundles["grid"], method="persistent_mfd", n_flat=n,
        n_neighbours=8,
        quantized_weight=config["quantized_weight"],
    )["accum"]
    return RoutineBuilder().step("q_init", q_init).step("accum", accum).freeze()


def _update_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    friction = build_friction_velocity(config["friction_law"])
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_update_depth(
                float* h, const float* Qi, float* Qo,
                const float* steepest_slope, const float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.nodata(i)$) {{
                h[i] = 0.0f;
                Qo[i] = 0.0f;
                return;
            }}
            if ($ctx.grid.can_out(i)$) {{
                Qo[i] = Qi[i];
                h[i] = 0.0f;
                return;
            }}
            float qout = $ctx.friction(h[i], steepest_slope[i])$
                * h[i] * flow_width[i];
            Qo[i] = qout;
            float dx = $ctx.grid.DX.get(0)$;
            float next = h[i] + (Qi[i] - qout) / (dx * dx) * $ctx.DT.get(0)$;
            h[i] = next > 0.0f ? next : 0.0f;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).compose("friction", friction).freeze()


def _reset_h_factory(be, _bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_reset_h(float* h) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) h[i] = 0.0f;
        }}''', domain=n,
    ).freeze()


def _copy_fill_depth_factory(be, _bundles, config):
    """Turn the reconstructed surface into initial physical water depth."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_copy_fill_depth(
                const float* z, const float* filled, float* h) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float depth = filled[i] - z[i];
            h[i] = depth > 0.0f ? depth : 0.0f;
        }}''', domain=n,
    ).freeze()


def _merge_fill_depth_factory(be, _bundles, config):
    """Add reconstructed storage without losing sub-ULP existing depth."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_merge_fill_depth(
                const float* z, const float* filled, float* h) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float depth = filled[i] - z[i];
            if (depth > h[i]) h[i] = depth;
        }}''', domain=n,
    ).freeze()


def _fill_surface_plan(source, destination, distance="fill_epsilon_distance"):
    def plan_for(frozen, _be):
        values = {
            "z": source, "filled": destination, "parent": "fill_parent",
            "frontier": "fill_frontier", "counters": "fill_counters",
            "queued_gen": "fill_queued_gen", "active": "active.handle",
            "P": "pass_index", "ACTIVE": "active",
            "dist": distance, "anc": "fill_epsilon_ancestor",
            "dist_in": distance,
            "dist_out": "fill_epsilon_distance_work",
            "anc_in": "fill_epsilon_ancestor",
            "anc_out": "fill_epsilon_ancestor_work",
            "rec": "rec",
        }
        plan = _grid_leaf_plan(frozen, values)
        plan.update({
            "hops_forward.dist_in": distance,
            "hops_forward.dist_out": "fill_epsilon_distance_work",
            "hops_forward.anc_in": "fill_epsilon_ancestor",
            "hops_forward.anc_out": "fill_epsilon_ancestor_work",
            "hops_backward.dist_in": "fill_epsilon_distance_work",
            "hops_backward.dist_out": distance,
            "hops_backward.anc_in": "fill_epsilon_ancestor_work",
            "hops_backward.anc_out": "fill_epsilon_ancestor",
        })
        return plan

    return plan_for


def build_graphflood_program() -> type:
    """Build the regular-grid D8 CuPy GraphFlood Program class."""
    b = ProgramBuilder("GraphFloodProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("dx", default=1.0)
    b.config("friction_law", choices=("manning",), default="manning")
    b.config("quantized_weight", choices=(False, True), default=True)
    b.config(
        "mfd_local_minima",
        choices=(
            "rank_cordonnier", "fill_cordonnier", "reconstruct_epsilon",
        ),
        default="rank_cordonnier",
    )

    b.param("precipitation", "scalar", "f32", value=0.0)
    b.param("friction_coefficient", "scalar", "f32", value=0.033)
    b.param("friction_exponent", "scalar", "f32", value=2.0 / 3.0)
    b.param("dt", "scalar", "f32", value=1.0e-3)
    b.param("ndep", "scalar", "i32", value=0)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)

    flat = Dim("ny") * Dim("nx")
    shape = (Dim("ny"), Dim("nx"))
    b.data("z", "f32", shape, role="input", shape_source=True)
    b.data("h", "f32", shape, role="state")
    b.data("Qi", "f32", shape, role="output")
    b.data("Qo", "f32", shape, role="output")
    b.data("surface", "f32", shape, role="internal")
    b.data("rec", "i32", shape, role="output")

    def grid_structure(be, **_):
        _cupy_only(be)
        return make_grid_group(be, topology="D8", boundary="normal", outlet="edge")

    def grid_params(be, pool, *, nx, ny, dx):
        return make_grid_parameters(
            be, pool, nx, ny, dx, topology="D8", outlet="edge"
        )

    b.bundle("grid", grid_structure, grid_params,
             dims=("nx", "ny"), config=("dx",))

    for name in (
        "rec_initial", "rank_ancestor", "rank_ancestor_alt", "rank", "rank_alt",
        "bid", "basin_saddlenode", "basin_route", "b_rcv",
        "mfd_frontier0", "mfd_frontier1", "indegree",
    ):
        b.data(name, "i32", (flat,), role="internal")
    for name in ("z_prime", "steepest_slope", "flow_width"):
        b.data(name, "f32", (flat,), role="internal")
    b.data(
        "weights",
        lambda config: "u8" if config["quantized_weight"] else "f32",
        (8 * flat,), role="internal",
    )
    for name in ("is_border", "directions"):
        b.data(name, "u8", (flat,), role="internal")
    for name in ("basin_saddle", "outlet"):
        b.data(name, "i64", (flat,), role="internal")
    b.data("mfd_count", "i32", (2,), role="internal")
    b.data("mfd_barrier", "u32", (1,), role="internal")

    # One-shot fill initialization scratch. Keeping every field temporary
    # releases it back to the program pool as soon as the operation returns.
    for name in ("fill_parent", "fill_epsilon_ancestor", "fill_epsilon_ancestor_work",
                 "fill_counters", "fill_queued_gen"):
        b.data(name, "i32", (flat,), lifetime="temp")
    for name in ("fill_epsilon_distance", "fill_epsilon_distance_work"):
        b.data(name, "f32", (flat,), lifetime="temp")
    b.data("fill_frontier", "i32", (2 * flat,), lifetime="temp")

    b.add("reset_h", _reset_h_factory, bind={"h": "h"})
    b.add("reconstruct_fill_surface", _reconstruct_epsilon_factory,
          bind=_fill_surface_plan("z", "surface"))
    b.add("copy_fill_depth", _copy_fill_depth_factory,
          bind={"z": "z", "filled": "surface", "h": "h"})
    b.pipeline("initialize_h_from_fill",
               ("reconstruct_fill_surface", "copy_fill_depth"))
    b.add("make_surface", _make_surface_factory,
          bind={"z": "z", "h": "h", "surface": "surface"})
    b.add("reconstruct_hydraulic_surface", _reconstruct_epsilon_factory,
          bind=_fill_surface_plan("surface", "z_prime", distance="surface"))
    b.add("copy_hydraulic_fill_depth", _merge_fill_depth_factory,
          bind={"z": "z", "filled": "z_prime", "h": "h"})
    b.pipeline("fill_hydraulic_surface", (
        "make_surface", "reconstruct_hydraulic_surface",
        "copy_hydraulic_fill_depth",
    ))
    b.add("route", _route_factory,
          bind={"grid": "grid", "z": "surface", "rec": "rec"})
    b.add("snapshot_receivers", _snapshot_factory,
          bind={"rec": "rec", "rec_initial": "rec_initial"})
    b.add("skip_local_minima", _noop_factory, bind={})
    b.add("resolve_cordonnier", _carve_factory, bind=_carve_plan)
    b.dispatch("route_local_minima", on="mfd_local_minima", cases={
        "rank_cordonnier": "route",
        "fill_cordonnier": "route",
        "reconstruct_epsilon": "skip_local_minima",
    })
    b.dispatch("snapshot_local_minima", on="mfd_local_minima", cases={
        "rank_cordonnier": "snapshot_receivers",
        "fill_cordonnier": "skip_local_minima",
        "reconstruct_epsilon": "skip_local_minima",
    })
    b.dispatch("resolve_minima", on="mfd_local_minima", cases={
        "rank_cordonnier": "resolve_cordonnier",
        "fill_cordonnier": "resolve_cordonnier",
        "reconstruct_epsilon": "reconstruct_hydraulic_surface",
    })
    b.add("compute_rank", _rank_factory, bind=_rank_plan)
    b.add("compute_cordonnier_fill", _cordonnier_fill_factory,
          bind=_cordonnier_fill_plan)
    b.dispatch("prepare_mfd_surface", on="mfd_local_minima", cases={
        "rank_cordonnier": "compute_rank",
        "fill_cordonnier": "compute_cordonnier_fill",
        "reconstruct_epsilon": "copy_hydraulic_fill_depth",
    })
    b.add("build_rank_topology", _topology_factory, bind=_topology_plan)
    b.add("build_fill_topology", _filled_topology_factory,
          bind=_filled_topology_plan)
    b.add("build_reconstructed_topology", _reconstructed_topology_factory,
          bind=_reconstructed_topology_plan)
    b.dispatch("build_topology", on="mfd_local_minima", cases={
        "rank_cordonnier": "build_rank_topology",
        "fill_cordonnier": "build_fill_topology",
        "reconstruct_epsilon": "build_reconstructed_topology",
    })
    b.add("prepare_frontier", _frontier_factory, bind={
        "clear.count": "mfd_count", "clear.barrier": "mfd_barrier",
        "compact.indegree": "indegree", "compact.frontier": "mfd_frontier0",
        "compact.count": "mfd_count",
    })
    b.add("accumulate", _accumulation_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "PRECIPITATION": "precipitation", "Qi": "Qi",
              "frontier0": "mfd_frontier0", "frontier1": "mfd_frontier1",
              "count": "mfd_count", "barrier": "mfd_barrier",
              "dirs": "directions", "mfd_w": "weights", "accum": "Qi",
              "indegree": "indegree",
          }))
    b.add("update_depth", _update_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "h": "h", "Qi": "Qi", "Qo": "Qo",
              "steepest_slope": "steepest_slope", "flow_width": "flow_width",
              "MANNING": "friction_coefficient", "EXPO": "friction_exponent",
              "DT": "dt",
          }))
    b.pipeline("run_n_step", (
        "make_surface", "route_local_minima", "snapshot_local_minima",
        "resolve_minima", "prepare_mfd_surface", "build_topology",
        "prepare_frontier", "accumulate", "update_depth",
    ))
    return b.freeze()


GraphFloodProgram = build_graphflood_program()

__all__ = ["GraphFloodProgram", "build_graphflood_program"]
