"""Surface operations for GraphFlood."""

import math
from pyfastflow.core import KernelBuilder, SequenceBuilder
from pyfastflow.flow import (
    depression_binding_plan, make_depression_solver, make_depressions,
    make_mfd_topology,
)
from pyfastflow.flow._program_cupy import BLOCK, _cupy_only, _grid_leaf_plan

def _make_surface_factory(be, _bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void make_hydraulic_surface(
                const float* z, const float* h, float* surface,
                double* hydraulic_surface) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) {{
                double eta = (double)z[i] + (double)h[i];
                surface[i] = (float)eta;
                hydraulic_surface[i] = eta;
            }}
        }}''',
        domain=n,
    ).freeze()

def _refresh_hydraulic_surface_factory(be, _bundles, config):
    """Refresh double precision eta after a minima method changes h."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void refresh_hydraulic_surface(
                const float* z, const float* h,
                double* hydraulic_surface) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n})
                hydraulic_surface[i] = (double)z[i] + (double)h[i];
        }}''',
        domain=n,
    ).freeze()

def _route_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_receivers_f64(
                const double* surface, int* rec) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.can_out(i)$) {{ rec[i] = i; return; }}

            int receiver = i;
            double best_slope = 0.0;
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            for (int k = 0; k < nk; ++k) {{
                int j = $ctx.grid.neighbour(i, k)$;
                if (j == -1) continue;
                double slope = (surface[i] - surface[j])
                             / (double)$ctx.grid.dist(i, k)$;
                if (slope > best_slope) {{
                    best_slope = slope;
                    receiver = j;
                }}
            }}
            rec[i] = receiver;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

def _snapshot_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank",
        n_flat=config["nx"] * config["ny"], topology=config["topology"],
        quantized_weight=config.get("quantized_weight", False),
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
    return {target: (
                "surface" if value == "z" else
                "basin_outlet" if value == "outlet" else value
            )
            for target, value in plan.items()}

def _rank_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank",
        n_flat=config["nx"] * config["ny"], topology=config["topology"],
        quantized_weight=config.get("quantized_weight", False),
    )["receiver_rank"]

def _rank_plan(_frozen, _be):
    return {
        "init.rec": "rec", "init.ancestor": "rank_ancestor", "init.rank": "rank",
        "forward.ancestor_in": "rank_ancestor", "forward.rank_in": "rank",
        "forward.ancestor_out": "rank_ancestor_alt", "forward.rank_out": "rank_alt",
        "backward.ancestor_in": "rank_ancestor_alt", "backward.rank_in": "rank_alt",
        "backward.ancestor_out": "rank_ancestor", "backward.rank_out": "rank",
    }

def _cordonnier_surface_factory(be, _bundles, config, *, fill_depth):
    """Build Cordonnier's path-maximum routing potential."""
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
    h_argument = "float* h, " if fill_depth else ""
    h_update = (
        "if (added_depth > 0.0f) h[i] += added_depth;"
        if fill_depth else ""
    )
    apply_name = "graphflood_apply_fill" if fill_depth else "graphflood_apply_carve"
    apply = KernelBuilder(
        f'''extern "C" __global__ void {apply_name}(
                {h_argument}float* surface, const float* spill,
                const int* rec_initial, const int* rec,
                unsigned char* conditioned) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float filled = spill[i];
            float added_depth = filled - surface[i];
            conditioned[i] = (added_depth > 0.0f
                              || rec_initial[i] != rec[i]) ? 1u : 0u;
            {h_update}
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

def _cordonnier_fill_factory(be, bundles, config):
    """Physically fill z+h to Cordonnier's path maximum."""
    return _cordonnier_surface_factory(
        be, bundles, config, fill_depth=True,
    )

def _cordonnier_carve_factory(be, bundles, config):
    """Build the same routing potential without changing water depth."""
    return _cordonnier_surface_factory(
        be, bundles, config, fill_depth=False,
    )

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
        "apply.spill": "z_prime", "apply.rec_initial": "rec_initial",
        "apply.rec": "rec", "apply.conditioned": "is_border",
    }

def _cordonnier_carve_plan(frozen, be):
    plan = _cordonnier_fill_plan(frozen, be)
    plan.pop("apply.h")
    plan["apply.conditioned"] = "hydraulic_conditioned"
    return plan

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
                const float* z, const float* filled, float* h,
                unsigned char* conditioned) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float depth = filled[i] - z[i];
            conditioned[i] = depth > h[i] ? 1u : 0u;
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
