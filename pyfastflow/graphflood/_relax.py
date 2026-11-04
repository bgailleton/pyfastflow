"""Analytical depth updates of GraphFloodRelax (and the shared warm-up).

local: relax h towards the Manning depth of Qi at the frozen steepest slope.
capped: the same with target and depth capped at DEPTH_CAP (warm-up).
bottom_up: receiver-first persistent solve of the controlling-receiver DAG,
then relaxation towards that candidate.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder, RoutineBuilder
from pyfastflow.flow._cupy_mfd_accum import persistent_grid_block
from pyfastflow.flow._program_cupy import _cupy_only, _grid_leaf_plan

from ._friction import build_friction_velocity
from ._routing import n_slots


def _local_analytical_update_factory(be, bundles, config):
    """Original pointwise inverse of the frozen-slope discharge closure."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    friction = build_friction_velocity(config["friction_law"])
    index = (f"int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
             f"            if (i >= {n}) return;")
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_local_analytical(
                float* h, const float* Qi, float* Qo,
                const float* steepest_slope, const float* flow_width) {{
            {index}
            if ($ctx.grid.nodata(i)$) {{
                h[i] = 0.0f;
                Qo[i] = 0.0f;
                return;
            }}
            if ($ctx.grid.can_out(i)$) {{
                h[i] = 0.0f;
                Qo[i] = Qi[i];
                return;
            }}

            float q = fmaxf(Qi[i], 0.0f);
            float slope = fmaxf(steepest_slope[i], 1.0e-5f);
            float width = fmaxf(flow_width[i], 1.0e-9f);
            float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
            float alpha = fmaxf(1.0f + $ctx.EXPO.get(i)$, 1.0e-3f);
            float target = q > 0.0f
                ? powf(q * manning / (width * sqrtf(slope)), 1.0f / alpha)
                : 0.0f;
            float relaxation = fminf(1.0f, fmaxf(0.0f,
                $ctx.ANALYTICAL_RELAXATION.get(i)$));
            float next_h = fmaxf(
                h[i] + relaxation * (target - fmaxf(h[i], 0.0f)), 0.0f);
            h[i] = next_h;
            Qo[i] = $ctx.friction(next_h, slope, i)$ * next_h * width;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).compose("friction", friction).freeze()


def _capped_local_analytical_update_factory(be, bundles, config):
    """Local analytical update with target and depth capped at DEPTH_CAP."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    friction = build_friction_velocity(config["friction_law"])
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_capped_analytical(
                float* h, const float* Qi, float* Qo,
                const float* steepest_slope, const float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.nodata(i)$) {{ h[i] = 0.0f; Qo[i] = 0.0f; return; }}
            if ($ctx.grid.can_out(i)$) {{ h[i] = 0.0f; Qo[i] = Qi[i]; return; }}

            float q = fmaxf(Qi[i], 0.0f);
            float slope = fmaxf(steepest_slope[i], 1.0e-5f);
            float width = fmaxf(flow_width[i], 1.0e-9f);
            float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
            float alpha = fmaxf(1.0f + $ctx.EXPO.get(i)$, 1.0e-3f);
            float target = q > 0.0f
                ? powf(q * manning / (width * sqrtf(slope)), 1.0f / alpha)
                : 0.0f;
            float depth_cap = fmaxf($ctx.DEPTH_CAP.get(0)$, 0.0f);
            target = fminf(target, depth_cap);
            float relaxation = fminf(1.0f, fmaxf(0.0f,
                $ctx.ANALYTICAL_RELAXATION.get(i)$));
            float next_h = fminf(fmaxf(
                h[i] + relaxation * (target - fmaxf(h[i], 0.0f)), 0.0f),
                depth_cap);
            h[i] = next_h;
            Qo[i] = $ctx.friction(next_h, slope, i)$ * next_h * width;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).compose("friction", friction).freeze()


def _local_analytical_update_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "h": "h", "Qi": "Qi", "Qo": "Qo",
        "steepest_slope": "steepest_slope", "flow_width": "flow_width",
        "MANNING": "friction_coefficient", "EXPO": "friction_exponent",
        "ANALYTICAL_RELAXATION": "relaxation",
    })


def _warmup_update_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "h": "h", "Qi": "Qi", "Qo": "Qo",
        "steepest_slope": "steepest_slope", "flow_width": "flow_width",
        "MANNING": "friction_coefficient", "EXPO": "friction_exponent",
        "ANALYTICAL_RELAXATION": "warmup_relaxation",
        "DEPTH_CAP": "depth_cap",
    })


def _bottom_up_analytical_update_factory(be, bundles, config):
    """Receiver-first persistent Kahn solve of the frozen hydraulic DAG."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    nk = n_slots(config)
    persistent_grid, persistent_block = persistent_grid_block(
        blocks_per_sm=1, threads=256,
    )
    resident_threads = persistent_grid[0] * persistent_block[0]
    carve = config["mfd_local_minima"] == "carve_cordonnier"
    cut_assignment = (
        f"hydraulic_cut[i] = (control[i] < {nk} && best <= 0.0f) ? 1u : 0u;"
        if carve else "hydraulic_cut[i] = 0u;"
    )

    clear = KernelBuilder(
        '''extern "C" __global__ void graphflood_reverse_clear(
                int* count, unsigned int* barrier) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < 2) count[i] = 0;
            if (i == 0) barrier[0] = 0u;
        }''', domain=2,
    ).freeze()

    # Freeze one controlling receiver for the nonlinear sweep. Its SFD subset
    # supplies the only dependencies needed by the scalar Manning closure;
    # the full MFD graph is rebuilt by the outer pipeline before the next
    # sweep.
    prepare = KernelBuilder(
        f'''extern "C" __global__ void graphflood_reverse_prepare(
                const float* z, const float* h,
                const unsigned char* directions, int* remaining,
                unsigned char* control, unsigned char* hydraulic_cut,
                int* frontier, int* count) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            unsigned int mask = (unsigned int)directions[i];
            control[i] = 255u;
            float best = -3.402823466e+38f;
            if (!$ctx.grid.nodata(i)$ && mask) {{
                #pragma unroll
                for (int k = 0; k < {nk}; ++k) {{
                    if (!(mask & (1u << k))) continue;
                    int r = $ctx.grid.neighbour_raw(i, k)$;
                    float slope = ((z[i] - z[r]) + (h[i] - h[r]))
                                / $ctx.grid.dist(i, k)$;
                    if (slope > best) {{
                        best = slope;
                        control[i] = (unsigned char)k;
                    }}
                }}
            }}
            {cut_assignment}
            // The nonlinear candidate only reads the controlling receiver.
            // Waiting for every MFD receiver adds dependencies without
            // changing the result and makes the reverse walk much deeper.
            remaining[i] = control[i] < {nk} ? 1 : 0;
            if (!$ctx.grid.nodata(i)$ && remaining[i] == 0) {{
                int p = atomicAdd(&count[0], 1);
                frontier[p] = i;
            }}
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    solve = KernelBuilder(
        f'''extern "C" __global__ void graphflood_reverse_hydraulics(
                int* __restrict__ frontier0, int* __restrict__ frontier1,
                int* __restrict__ count, unsigned int* __restrict__ barrier,
                const unsigned char* __restrict__ control,
                const unsigned char* __restrict__ hydraulic_cut,
                int* __restrict__ remaining, const float* __restrict__ z,
                const float* __restrict__ h, const float* __restrict__ Qi,
                float* __restrict__ candidate,
                float* __restrict__ solved_slope) {{
            __shared__ int staged[2048];
            __shared__ int staged_n;
            __shared__ unsigned int staged_base;
            __shared__ int s_size;

            int* frontiers[2] = {{frontier0, frontier1}};
            int parity = 0;
            unsigned int level = 0;
            // Three rotating counters: level L reads count[L % 3], pushes
            // into count[(L + 1) % 3] and clears count[(L + 2) % 3], which
            // no block touches during level L.
            while (true) {{
                int* c_out = &count[(level + 1) % 3];
                if (threadIdx.x == 0)
                    s_size = *((volatile int*)&count[level % 3]);
                __syncthreads();
                int size_in = s_size;
                if (size_in == 0) break;
                int* fin = frontiers[parity];
                int* fout = frontiers[1 - parity];
                if (threadIdx.x == 0) staged_n = 0;
                __syncthreads();

                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                int stride = gridDim.x * blockDim.x;
                for (int p = tid; p < size_in; p += stride) {{
                    int u = fin[p];
                    if ($ctx.grid.can_out(u)$) {{
                        candidate[u] = 0.0f;
                        solved_slope[u] = fmaxf(
                            $ctx.CARVE_SLOPE.get(u)$, 1.0e-12f);
                    }} else {{
                        float target_q = fmaxf(Qi[u], 0.0f);
                        float old_h = fmaxf(h[u], 0.0f);
                        float manning = fmaxf(
                            $ctx.MANNING.get(u)$, 1.0e-9f);
                        float alpha = fmaxf(
                            1.0f + $ctx.EXPO.get(u)$, 1.0e-3f);
                        int k = (int)control[u];
                        if (k >= {nk}) {{
                            // A non-outlet graph sink has no physical closure.
                            // It should not occur after local-minimum handling.
                            candidate[u] = old_h;
                            solved_slope[u] = fmaxf(
                                $ctx.CARVE_SLOPE.get(u)$, 1.0e-12f);
                        }} else {{
                            int r = $ctx.grid.neighbour_raw(u, k)$;
                            float width = $ctx.grid.flow_width(u, k)$;
                            float length = $ctx.grid.dist(u, k)$;
                            bool cut = hydraulic_cut[u] != 0u;
                            float inherited_slope = fmaxf(
                                solved_slope[r],
                                fmaxf($ctx.CARVE_SLOPE.get(u)$, 1.0e-12f));
                            float receiver_h = cut ? 0.0f : candidate[r];
                            float bed_drop = z[u] - z[r];

                            // The lower bracket enforces eta[u] >= eta[r].
                            // Unlike the explicit update, this solve must not
                            // hide an uphill link behind the slope floor.
                            float lo = cut ? 0.0f
                                : fmaxf(receiver_h - bed_drop, 0.0f);
                            float old_slope = cut ? inherited_slope : fmaxf(
                                (bed_drop + old_h - receiver_h) / length,
                                1.0e-12f);
                            float guess = target_q > 0.0f
                                ? powf(target_q * manning
                                       / (width * sqrtf(old_slope)),
                                       1.0f / alpha)
                                : lo;
                            float hi = fmaxf(fmaxf(old_h, guess),
                                             lo + 1.0e-7f);
                            #pragma unroll
                            for (int it = 0; it < 32; ++it) {{
                                float slope = cut ? inherited_slope : fmaxf(
                                    (bed_drop + hi - receiver_h) / length,
                                    0.0f);
                                float qhi = width / manning * powf(hi, alpha)
                                          * sqrtf(slope);
                                if (qhi >= target_q) break;
                                hi = hi * 2.0f + 1.0e-6f;
                            }}

                            float x = fminf(fmaxf(guess, lo), hi);
                            #pragma unroll
                            for (int it = 0; it < 16; ++it) {{
                                float slope = cut ? inherited_slope : fmaxf(
                                    (bed_drop + x - receiver_h) / length,
                                    0.0f);
                                float q = width / manning * powf(x, alpha)
                                        * sqrtf(slope);
                                if (q < target_q) lo = x;
                                else hi = x;
                                float derivative = q * alpha
                                    / fmaxf(x, 1.0e-12f);
                                if (!cut && slope > 0.0f)
                                    derivative += q * 0.5f / (slope * length);
                                float trial = x - (q - target_q)
                                    / fmaxf(derivative, 1.0e-20f);
                                if (!(trial > lo && trial < hi)
                                        || !isfinite(trial))
                                    trial = 0.5f * (lo + hi);
                                x = trial;
                            }}
                            float solved_h = target_q > 0.0f
                                ? x : lo;
                            candidate[u] = solved_h;
                            solved_slope[u] = cut ? inherited_slope : fmaxf(
                                (bed_drop + solved_h - receiver_h) / length,
                                0.0f);
                        }}
                    }}

                    __threadfence();
                    #pragma unroll
                    for (int k = 0; k < {nk}; ++k) {{
                        int donor = $ctx.grid.neighbour(u, k)$;
                        if (donor == -1) continue;
                        if ((int)control[donor] != {nk - 1} - k) continue;
                        int old = atomicAdd(&remaining[donor], -1);
                        if (old == 1) {{
                            int sp = atomicAdd(&staged_n, 1);
                            if (sp < 2048) staged[sp] = donor;
                            else {{
                                int pos = atomicAdd(c_out, 1);
                                fout[pos] = donor;
                            }}
                        }}
                    }}
                }}

                __syncthreads();
                int flush_n = min(staged_n, 2048);
                if (threadIdx.x == 0)
                    staged_base = atomicAdd(
                        (unsigned int*)c_out, (unsigned int)flush_n);
                __syncthreads();
                for (int i = threadIdx.x; i < flush_n; i += blockDim.x)
                    fout[staged_base + i] = staged[i];
                __threadfence();

                __syncthreads();
                if (threadIdx.x == 0) {{
                    if (blockIdx.x == 0) {{
                        count[(level + 2) % 3] = 0;
                        __threadfence();
                    }}
                    unsigned int target = (level + 1)
                                        * (unsigned int)gridDim.x;
                    atomicAdd(barrier, 1u);
                    unsigned int wait_ns = 32;
                    while (*((volatile unsigned int*)barrier) < target) {{
#if __CUDA_ARCH__ >= 700
                        __nanosleep(wait_ns);
                        if (wait_ns < 1024) wait_ns <<= 1;
#endif
                    }}
                }}
                __syncthreads();
                level++;
                parity = 1 - parity;
            }}
        }}''', domain=resident_threads, block=256,
    ).compose("grid", bundles["grid"]).freeze()

    apply = KernelBuilder(
        f'''extern "C" __global__ void graphflood_apply_bottom_up_candidate(
                const float* candidate, float* h) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.nodata(i)$ || $ctx.grid.can_out(i)$) {{
                h[i] = 0.0f;
                return;
            }}
            float relaxation = fminf(1.0f, fmaxf(0.0f,
                $ctx.ANALYTICAL_RELAXATION.get(i)$));
            float old_h = fmaxf(h[i], 0.0f);
            h[i] = fmaxf(
                old_h + relaxation * (candidate[i] - old_h), 0.0f);
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    diagnostics = KernelBuilder(
        f'''extern "C" __global__ void graphflood_bottom_up_diagnostics(
                const float* z, const float* h, const float* Qi,
                const unsigned char* control,
                const unsigned char* hydraulic_cut,
                const float* solved_slope, float* Qo) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.nodata(i)$) {{ Qo[i] = 0.0f; return; }}
            if ($ctx.grid.can_out(i)$) {{ Qo[i] = Qi[i]; return; }}
            int k = (int)control[i];
            if (k >= {nk}) {{ Qo[i] = 0.0f; return; }}
            int r = $ctx.grid.neighbour_raw(i, k)$;
            float width = $ctx.grid.flow_width(i, k)$;
            float slope = hydraulic_cut[i] ? solved_slope[i] : fmaxf(
                ((z[i] - z[r]) + (h[i] - h[r])) / $ctx.grid.dist(i, k)$, 0.0f);
            float depth = fmaxf(h[i], 0.0f);
            float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
            float alpha = fmaxf(1.0f + $ctx.EXPO.get(i)$, 1.0e-3f);
            Qo[i] = width / manning * powf(depth, alpha) * sqrtf(slope);
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    return (RoutineBuilder().step("clear", clear).step("prepare", prepare)
            .step("solve", solve).step("apply", apply)
            .step("diagnostics", diagnostics).freeze())


def _bottom_up_analytical_update_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "z": "z", "h": "h", "Qi": "Qi", "Qo": "Qo",
        "candidate": "z_prime",
        "directions": "directions", "remaining": "indegree",
        "control": "is_border", "hydraulic_cut": "hydraulic_cut",
        "solved_slope": "steepest_slope", "frontier": "mfd_frontier0",
        "frontier0": "mfd_frontier0", "frontier1": "mfd_frontier1",
        "count": "mfd_count", "barrier": "mfd_barrier",
        "MANNING": "friction_coefficient", "EXPO": "friction_exponent",
        "CARVE_SLOPE": "carve_slope_min",
        "ANALYTICAL_RELAXATION": "relaxation",
    })
