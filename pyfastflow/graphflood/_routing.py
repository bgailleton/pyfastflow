"""Routing operations for GraphFlood.

The u8 quantized-weight branches are kept but no program exposes them:
``config.get("quantized_weight", False)`` always selects f32 weights.
"""

from pyfastflow.core import KernelBuilder, RoutineBuilder
from pyfastflow.flow import make_accumulation, make_mfd_topology
from pyfastflow.flow._program_cupy import _cupy_only, _grid_leaf_plan

def n_slots(config):
    """Neighbour slots per cell of a config's grid or tree."""
    tree = config.get("tree")
    if tree is not None:
        return tree.n_neighbours
    return 8 if config["topology"] == "D8" else 4


def _build_hydraulic_topology(be, grid, n, topology, quantized_weight):
    """Rank-gated MFD weights plus unmodified hydraulic slope diagnostics."""
    weight_type = "unsigned char" if quantized_weight else "float"
    max_decl = "double max_score = 0.0;" if quantized_weight else ""
    max_forced = "max_score = 1.0;" if quantized_weight else ""
    max_update = "if (score > max_score) max_score = score;" if quantized_weight else ""
    weight_write = (
        "weights[i * nk + k] = scores[k] > 0.0 "
        "? (unsigned char)max(1, __double2int_rn(255.0 * scores[k] / max_score)) : 0;"
        if quantized_weight else
        "weights[i * nk + k] = sum_score > 0.0 "
        "? (float)(scores[k] / sum_score) : 0.0f;"
    )
    dirs_weights = KernelBuilder(
        f'''extern "C" __global__ void graphflood_ranked_mfd(
                const double* surface, const int* rec_initial, const int* rec,
                const int* rank, unsigned char* dirs, {weight_type}* weights,
                float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            unsigned char mask = 0;
            double scores[8];
            double sum_score = 0.0;
            {max_decl}
            double best_slope = 0.0;
            float best_width = $ctx.grid.dx(i)$;
            for (int k = 0; k < 8; ++k) scores[k] = 0.0;

            if (!$ctx.grid.can_out(i)$ && !$ctx.grid.nodata(i)$) {{
                // Manning uses the true hydraulic slope. MFD partitioning
                // separately uses slope times the link's effective width.
                for (int k = 0; k < nk; ++k) {{
                    int j = $ctx.grid.neighbour(i, k)$;
                    if (j == -1) continue;
                    double slope = (surface[i] - surface[j])
                                 / (double)$ctx.grid.dist(i, k)$;
                    if (slope > best_slope) {{
                        best_slope = slope;
                        best_width = $ctx.grid.flow_width(i, k)$;
                    }}
                }}

                if (rec_initial[i] != rec[i]) {{
                    for (int k = 0; k < nk; ++k) {{
                        if ($ctx.grid.neighbour(i, k)$ == rec[i]) {{
                            mask = (unsigned char)(1u << k);
                            scores[k] = 1.0;
                            sum_score = 1.0;
                            {max_forced}
                            break;
                        }}
                    }}
                }} else {{
                    for (int k = 0; k < nk; ++k) {{
                        int j = $ctx.grid.neighbour(i, k)$;
                        if (j == -1 || rank[j] >= rank[i]) continue;
                        double width = (double)$ctx.grid.width(i, k)$;
                        double slope = (surface[i] - surface[j])
                                     / (double)$ctx.grid.dist(i, k)$;
                        double score = slope > 0.0 ? slope * width : 0.0;
                        scores[k] = score;
                        if (score > 0.0) {{
                            mask |= (unsigned char)(1u << k);
                            sum_score += score;
                            {max_update}
                        }}
                    }}
                }}
            }}

            for (int k = 0; k < nk; ++k) {{ {weight_write} }}
            dirs[i] = mask;
            steepest_slope[i] = (float)best_slope;
            flow_width[i] = best_width;
        }}''',
        domain=n,
    ).compose("grid", grid).freeze()

    generic = make_mfd_topology(
        be, grid,
        method="cordonnier_rank", n_flat=n, topology=topology,
        quantized_weight=quantized_weight,
    )
    return dirs_weights, generic["indegree_reset"], generic["indegree_count"]

def _build_filled_hydraulic_topology(
        be, grid, n, topology, quantized_weight):
    """MFD on the exact conditioned surface, with precise physical slopes."""
    weight_type = "unsigned char" if quantized_weight else "float"
    max_decl = "float max_score = 0.0f;" if quantized_weight else ""
    max_update = "if (score > max_score) max_score = score;" if quantized_weight else ""
    weight_write = (
        "weights[i * nk + k] = scores[k] > 0.0f "
        "? (unsigned char)max(1, __float2int_rn(255.0f * scores[k] / max_score)) : 0;"
        if quantized_weight else
        "weights[i * nk + k] = sum_score > 0.0f "
        "? scores[k] / sum_score : 0.0f;"
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
            float best_width = $ctx.grid.dx(i)$;
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
                    float slope = drop / $ctx.grid.dist(i, k)$;
                    float score = slope * $ctx.grid.width(i, k)$;
                    scores[k] = score;
                    mask |= (unsigned char)(1u << k);
                    sum_score += score;
                    {max_update}
                    if (slope > best_slope) {{
                        best_slope = slope;
                        best_width = $ctx.grid.flow_width(i, k)$;
                    }}
                }}
            }}

            // Rank drops on a flat at 0 are denormal and flush to zero:
            // such a cell splits evenly between its receivers.
            if (mask != 0 && !(sum_score > 0.0f)) {{
                for (int k = 0; k < nk; ++k)
                    if (mask & (1u << k)) scores[k] = 1.0f;
                sum_score = (float)__popc((unsigned int)mask);
                {"max_score = 1.0f;" if quantized_weight else ""}
            }}
            for (int k = 0; k < nk; ++k) {{ {weight_write} }}
            dirs[i] = mask;
            steepest_slope[i] = best_slope;
            flow_width[i] = best_width;
        }}''', domain=n,
    ).compose("grid", grid).freeze()

    # Do not perturb the conditioned DAG. Only replace its hydraulic
    # diagnostic where neither endpoint was raised/rerouted and the true
    # double-precision surface has a valid downslope link.
    diagnostics = KernelBuilder(
        f'''extern "C" __global__ void graphflood_filled_diagnostics_f64(
                const double* physical_surface,
                const unsigned char* conditioned,
                const unsigned char* dirs,
                float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n} || conditioned[i]) return;
            double best_slope = 0.0;
            float best_width = flow_width[i];
            unsigned char mask = dirs[i];
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            for (int k = 0; k < nk; ++k) {{
                if (!(mask & (1u << k))) continue;
                int j = $ctx.grid.neighbour(i, k)$;
                if (j == -1 || conditioned[j]) continue;
                double slope = (physical_surface[i] - physical_surface[j])
                             / (double)$ctx.grid.dist(i, k)$;
                if (slope > best_slope) {{
                    best_slope = slope;
                    best_width = $ctx.grid.flow_width(i, k)$;
                }}
            }}
            if (best_slope > 0.0) {{
                steepest_slope[i] = (float)best_slope;
                flow_width[i] = best_width;
            }}
        }}''', domain=n,
    ).compose("grid", grid).freeze()
    generic = make_mfd_topology(
        be, grid, method="cordonnier_rank", n_flat=n, topology=topology,
        quantized_weight=quantized_weight,
    )
    return (
        dirs_weights, diagnostics,
        generic["indegree_reset"], generic["indegree_count"],
    )

def _topology_factory(be, bundles, config):
    _cupy_only(be)
    dirs, reset, count = _build_hydraulic_topology(
        be, bundles["grid"], config["nx"] * config["ny"],
        config["topology"],
        config.get("quantized_weight", False),
    )
    return (RoutineBuilder().step("dirs_weights", dirs)
            .step("indegree_reset", reset).step("indegree_count", count).freeze())

def _filled_topology_factory(be, bundles, config):
    _cupy_only(be)
    dirs, diagnostics, reset, count = _build_filled_hydraulic_topology(
        be, bundles["grid"], config["nx"] * config["ny"],
        config["topology"],
        config.get("quantized_weight", False),
    )
    return (RoutineBuilder().step("dirs_weights", dirs)
            .step("diagnostics", diagnostics)
            .step("indegree_reset", reset).step("indegree_count", count).freeze())

def _carve_topology_factory(be, bundles, config):
    """MFD on the virtual fill, with physical-or-subgrid hydraulic slope."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    dirs, diagnostics, reset, count = _build_filled_hydraulic_topology(
        be, bundles["grid"], n, config["topology"],
        config.get("quantized_weight", False),
    )
    effective_slope = KernelBuilder(
        f'''extern "C" __global__ void graphflood_carve_effective_slope(
                const double* physical_surface,
                const unsigned char* dirs,
                float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n} || $ctx.grid.nodata(i)$
                    || $ctx.grid.can_out(i)$) return;
            unsigned char mask = dirs[i];
            double best = 0.0;
            float width_best = flow_width[i];
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            for (int k = 0; k < nk; ++k) {{
                if (!(mask & (1u << k))) continue;
                int r = $ctx.grid.neighbour_raw(i, k)$;
                double slope = (physical_surface[i] - physical_surface[r])
                             / (double)$ctx.grid.dist(i, k)$;
                if (slope > best) {{
                    best = slope;
                    width_best = $ctx.grid.flow_width(i, k)$;
                }}
            }}
            steepest_slope[i] = best > 0.0
                ? (float)best
                : fmaxf($ctx.CARVE_SLOPE.get(i)$, 1.0e-12f);
            flow_width[i] = width_best;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()
    return (RoutineBuilder().step("dirs_weights", dirs)
            .step("diagnostics", diagnostics)
            .step("effective_slope", effective_slope)
            .step("indegree_reset", reset).step("indegree_count", count)
            .freeze())

def _reconstructed_topology_factory(be, bundles, config):
    """MFD and Manning diagnostics on reconstruct+epsilon potential."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    topology = make_mfd_topology(
        be, bundles["grid"], method="surface", n_flat=n,
        topology=config["topology"], diagonal_partition_correction=True,
        quantized_weight=config.get("quantized_weight", False),
    )
    diagnostics = KernelBuilder(
        f'''extern "C" __global__ void graphflood_reconstruct_diagnostics(
                const float* filled, const float* dist,
                const double* physical_surface,
                const unsigned char* conditioned,
                const unsigned char* dirs,
                float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            double best_slope = 0.0;
            float best_width = $ctx.grid.dx(i)$;
            unsigned char mask = dirs[i];
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            for (int k = 0; k < nk; ++k) {{
                if (!(mask & (1u << k))) continue;
                int j = $ctx.grid.neighbour(i, k)$;
                double width = (double)$ctx.grid.width(i, k)$;
                double conditioned_drop = ((double)filled[i]
                                           - (double)filled[j])
                                          + ((double)dist[i]
                                           - (double)dist[j]);
                double drop = conditioned_drop;
                if (!conditioned[i] && !conditioned[j]) {{
                    double physical_drop = physical_surface[i]
                                         - physical_surface[j];
                    if (physical_drop > 0.0) drop = physical_drop;
                }}
                double slope = drop / width;
                if (slope > best_slope) {{
                    best_slope = slope;
                    best_width = (float)width;
                }}
            }}
            steepest_slope[i] = (float)best_slope;
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
        "surface": "hydraulic_surface",
        "rec_initial": "rec_initial", "rec": "rec",
        "rank": "rank", "dirs": "directions", "weights": "weights",
        "steepest_slope": "steepest_slope", "flow_width": "flow_width",
        "indegree": "indegree",
    })

def _filled_topology_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "surface": "surface", "physical_surface": "hydraulic_surface",
        "conditioned": "is_border", "rank": "rank",
        "dirs": "directions",
        "weights": "weights", "steepest_slope": "steepest_slope",
        "flow_width": "flow_width", "indegree": "indegree",
    })

def _carve_topology_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "surface": "surface", "physical_surface": "hydraulic_surface",
        "conditioned": "hydraulic_conditioned", "rank": "rank",
        "dirs": "directions", "weights": "weights",
        "steepest_slope": "steepest_slope",
        "flow_width": "flow_width", "indegree": "indegree",
        "CARVE_SLOPE": "carve_slope_min",
    })

def _reconstructed_topology_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "filled": "z_prime", "dist": "surface",
        "physical_surface": "hydraulic_surface",
        "conditioned": "is_border", "dirs": "directions",
        "mfd_w": "weights",
        "steepest_slope": "steepest_slope",
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
            float dx = $ctx.grid.dx(i)$;
            Qi[i] = $ctx.grid.nodata(i)$ ? 0.0f
                : $ctx.PRECIPITATION.get(i)$ * dx * dx;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()
    accum = make_accumulation(
        be, bundles["grid"], method="persistent_mfd", n_flat=n,
        n_neighbours=n_slots(config),
        quantized_weight=config.get("quantized_weight", False),
    )["accum"]
    return RoutineBuilder().step("q_init", q_init).step("accum", accum).freeze()
