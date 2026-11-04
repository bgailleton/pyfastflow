"""CuPy kernels of GraphFloodParticles.

Every factory takes ``(be, bundles, config)`` and returns a frozen kernel or
routine; the program binds them by leaf name. See ``particles.py`` for the
algorithm. Neighbour k and nk - 1 - k are opposite (grid convention), so
pushed[j * nk + nk - 1 - k] is what neighbour j = neighbour(i, k) sends to i.

Author: B.G (10/2026)
"""

from pyfastflow.core import HelperBuilder, KernelBuilder, RoutineBuilder, new_uid
from pyfastflow.flow._cupy_mfd_accum import persistent_grid_block
from pyfastflow.flow._program_cupy import _cupy_only

from ._friction import _MIN_SLOPE

# Smallest head drop that counts as a link, and the slope floor of the depth
# update and of the final outflow.
_CONSTANTS = f"""
#define DROP_MIN 1.0e-6
#define MIN_SLOPE {_MIN_SLOPE}f
"""


def _cells(config):
    return config["nx"] * config["ny"]


def _kernel(be, bundles, text, domain, block=None, helpers=None):
    _cupy_only(be)
    text = _CONSTANTS + text
    node = (KernelBuilder(text, domain=domain) if block is None
            else KernelBuilder(text, domain=domain, block=block))
    node.compose("grid", bundles["grid"])
    for address, helper in (helpers or {}).items():
        node.compose(address, helper)
    return node.freeze()


_QINIT = r'''
// Start of the discharge field from the accumulation: every cell holds its
// inflow and has already sent weight * inflow to each receiver, so
// Qacc = rain + what the neighbours send holds exactly from the start.
extern "C" __global__ void gfp_qinit(
        const float* Qi, const unsigned char* directions,
        const float* weights, double* Qacc, float* pushed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    float q = fmaxf(Qi[i], 0.0f);
    Qacc[i] = (double)q;
    unsigned int mask = (unsigned int)directions[i];
    for (int k = 0; k < nk; ++k)
        pushed[i * nk + k] = (mask & (1u << k)) ? weights[i * nk + k] * q
                                                : 0.0f;
}
'''


def qinit_factory(be, bundles, config):
    return _kernel(be, bundles, _QINIT, _cells(config))


_REACH_SEED = r'''
// Start of the downstream sweep: every valid cell with Qi >= SOURCE_Q (and
// Qi > 0) is a source, reached and queued.
extern "C" __global__ void gfp_reach_seed(
        const float* Qi, int* reach, int* frontier0, int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    int seed = (!$ctx.grid.nodata(i)$ && Qi[i] > 0.0f
                && Qi[i] >= $ctx.SOURCE_Q.get(0)$) ? 1 : 0;
    reach[i] = seed;
    if (seed) frontier0[atomicAdd(&count[0], 1)] = i;
}
'''

_REACH_SWEEP = r'''
// Persistent breadth-first sweep downstream along the steepest descent of the
// pre-step DAG: from each reached cell only its steepest receiver (largest
// weight / link length, weights being proportional to drop) is followed, so
// the marked cells are the D8 paths from the sources. Three rotating
// counters: level L reads count[L % 3], fills count[(L + 1) % 3] and clears
// count[(L + 2) % 3].
extern "C" __global__ void gfp_reach_sweep(
        const unsigned char* dirs, const float* weights, int* reach,
        int* frontier0, int* frontier1, int* count, unsigned int* barrier) {
    __shared__ int s_size;
    int* frontiers[2] = {frontier0, frontier1};
    int phase = 0;
    unsigned int level = 0;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    while (true) {
        int* c_out = &count[(level + 1) % 3];
        if (threadIdx.x == 0) s_size = *((volatile int*)&count[level % 3]);
        __syncthreads();
        int size = s_size;
        if (size == 0) break;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = gridDim.x * blockDim.x;
        for (int p = tid; p < size; p += stride) {
            int u = frontiers[phase][p];
            unsigned int mask = (unsigned int)dirs[u];
            int steepest = -1;
            float best = 0.0f;
            for (int k = 0; k < nk; ++k) {
                if (!(mask & (1u << k))) continue;
                float slope = weights[u * nk + k] / $ctx.grid.dist(u, k)$;
                if (steepest < 0 || slope > best) {
                    steepest = k;
                    best = slope;
                }
            }
            if (steepest >= 0) {
                int r = $ctx.grid.neighbour_raw(u, steepest)$;
                if (atomicExch(&reach[r], 1) == 0)
                    frontiers[1 - phase][atomicAdd(c_out, 1)] = r;
            }
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            if (blockIdx.x == 0) {
                count[(level + 2) % 3] = 0;
                __threadfence();
            }
            unsigned int target = (level + 1) * gridDim.x;
            atomicAdd(barrier, 1u);
            while (*((volatile unsigned int*)barrier) < target) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(64);
#endif
            }
        }
        __syncthreads();
        ++level;
        phase = 1 - phase;
    }
}
'''

def reach_factory(be, bundles, config):
    """Steepest-descent paths downstream of the sources: seed, then sweep."""
    grid, block = persistent_grid_block(blocks_per_sm=1, threads=256)
    seed = _kernel(be, bundles, _REACH_SEED, _cells(config))
    sweep = _kernel(be, bundles, _REACH_SWEEP, grid[0] * block[0],
                    block=block[0])
    return RoutineBuilder().step("seed", seed).step("sweep", sweep).freeze()


_HUPDATE_NEWTON = r'''
// Depth update of one visit: solve q = width / n * x^alpha * sqrt(S(x)) with
// S(x) = (drop_base + x) / length against the receiver's fixed head
// (drop_base = z_i - receiver head), then relax towards it. The lower bound
// x >= -drop_base keeps the cell at least level with its receiver, which
// fills pits.
__device__ float HUPDATE(float h, float q, float width, float drop_base,
                         float length, float manning, float alpha,
                         float relaxation) {
    float lo = fmaxf(-drop_base, 0.0f);
    if (!(q > 0.0f))
        return fmaxf(h + relaxation * (lo - h), 0.0f);
    float s0 = fmaxf((drop_base + fmaxf(h, lo)) / length, MIN_SLOPE);
    float guess = powf(q * manning / (width * sqrtf(s0)), 1.0f / alpha);
    float hi = fmaxf(fmaxf(h, guess), lo + 1.0e-7f);
    for (int it = 0; it < 32; ++it) {
        float s = fmaxf((drop_base + hi) / length, 0.0f);
        if (width / manning * powf(hi, alpha) * sqrtf(s) >= q) break;
        hi = hi * 2.0f + 1.0e-6f;
    }
    float x = fminf(fmaxf(guess, lo), hi);
    for (int it = 0; it < 16; ++it) {
        float s = fmaxf((drop_base + x) / length, 0.0f);
        float qx = width / manning * powf(x, alpha) * sqrtf(s);
        if (qx < q) lo = x;
        else hi = x;
        float derivative = qx * alpha / fmaxf(x, 1.0e-12f);
        if (s > 0.0f) derivative += qx * 0.5f / (s * length);
        float trial = x - (qx - q) / fmaxf(derivative, 1.0e-20f);
        if (!(trial > lo && trial < hi) || !isfinite(trial))
            trial = 0.5f * (lo + hi);
        x = trial;
    }
    return fmaxf(h + relaxation * (x - h), 0.0f);
}
'''


_WALK = r'''
__device__ unsigned int gfp_hash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Per-cell golden-ratio sequence: the visit count indexes a Weyl sequence
// offset by a hash of the cell, so successive visits split evenly.
__device__ float gfp_weyl(unsigned int visit, unsigned int key) {
    unsigned int w = visit * 2654435769u + gfp_hash(key);
    return (float)(w >> 8) * (1.0f / 16777216.0f);
}

// Discharge-field walk. Particles carry nothing; they are live processors.
// pushed[c * nk + k] is what cell c sends to its neighbour k.
// Start: a Weyl draw over the release index picks a cell of the spawning
// area (uniform cdf), then a random offset within SPAWN_PAD rows and columns
// is walked through the grid neighbours (stopping at walls and nodata,
// wrapping on periodic edges). A start on nodata or on a cell with no
// discharge is drawn again; after SPAWN_TRIES draws the unpadded cell of
// the spawning area is used.
// Visit: the particle tries to lock the cell (moving on if it is taken).
// Holding it, it fills the cell to just above its lowest neighbour if it is
// a pit, takes as discharge the rain plus what its currently higher
// neighbours send it (a send from a neighbour that is now lower waits until
// that neighbour rewrites its sends), rewrites its own sends from that
// discharge (to neighbours lower by more than DROP_MIN, in proportion to
// drop), stores the discharge in Qacc and relaxes h by RELAXATION towards
// the newton depth against its steepest receiver. A cell only ever
// writes its own sends, so every send set sums to its sender's discharge
// and no water is created.
// Move: to a neighbour below the lowest head of the path so far, by the
// per-cell golden-ratio routing on the drops, else along the warm-up DAG;
// at most WALK_STEPS cells, stopping at an outlet.
// With PROPAGATE > 0, every neighbour whose send changed by more than
// PROPAGATE (relative) goes onto the thread's stack; before
// claiming a new particle the thread processes the stacked cells the same
// way (without walking). A full stack drops further cells.
extern "C" __global__ void gfp_walk(
        const double* cdf, const float* z, float* h, double* Qacc,
        float* pushed, const unsigned char* dirs, const float* weights,
        unsigned int* visits, int* locks, int* claim, double* stats) {
    int n = $ctx.CELL_COUNT.get(0)$;
    double total = cdf[n - 1];
    if (!(total > 0.0)) return;

    int count = $ctx.N_PARTICLES.get(0)$;
    unsigned int base = (unsigned int)$ctx.CLOCK_BASE.get(0)$;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    float dx = $ctx.grid.DX.get(0)$;
    float relaxation = fminf(1.0f, fmaxf($ctx.RELAXATION.get(0)$, 0.0f));
    double drop_min = DROP_MIN;
    int max_hops = $ctx.WALK_STEPS.get(0)$;
    int pad = max($ctx.SPAWN_PAD.get(0)$, 0);
    unsigned int seed = gfp_hash((unsigned int)$ctx.SEED.get(0)$);
    double offset = (double)(seed >> 8) * (1.0 / 16777216.0);
    float tolerance = $ctx.PROPAGATE.get(0)$;
    bool propagate = tolerance > 0.0f;
    const int SPAWN_TRIES = 32;

    // Counts kept per thread and added to stats once, when it runs out of
    // particles: launched, exited, hop limit, stuck, skipped, processed,
    // correction processings, rejected start draws.
    double counts[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int stack[16];
    int stack_n = 0;
    while (true) {
        int i;
        bool walking = !(propagate && stack_n > 0);
        if (walking) {
            int p = atomicAdd(&claim[0], 1);
            if (p >= count) {
                for (int c = 0; c < 8; ++c)
                    if (counts[c] != 0.0) atomicAdd(&stats[c], counts[c]);
                return;
            }
            unsigned int index = base + (unsigned int)p;
            // Offsets and acceptance draws: one hash chain per particle.
            unsigned int r = gfp_hash(index ^ seed);
            for (int attempt = 0;; ++attempt) {
                // Weyl draws over the release index; retries shift by a
                // second irrational.
                double u = (double)index * 0.6180339887498949
                         + (double)attempt * 0.7548776662466927 + offset;
                u = (u - floor(u)) * total;
                int lo = 0, hi = n - 1;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (cdf[mid] > u) hi = mid;
                    else lo = mid + 1;
                }
                int base_cell = lo;
                i = base_cell;
                if (attempt > 0) r = gfp_hash(r);
                if (pad > 0) {
                    // Walk the offset one neighbour at a time, never away
                    // from it: diagonals while both components remain.
                    int span = 2 * pad + 1;
                    int rem_r = (int)(r % span) - pad;
                    int rem_c = (int)((r >> 16) % span) - pad;
                    for (int s = 0; s < 2 * pad && (rem_r || rem_c); ++s) {
                        int sr = (rem_r > 0) - (rem_r < 0);
                        int sc = (rem_c > 0) - (rem_c < 0);
                        int best = -1, best_score = 0, br = 0, bc = 0;
                        for (int k = 0; k < nk; ++k) {
                            int dr, dc;
                            $ctx.grid.delta(k, &dr, &dc)$;
                            if ((dr && dr != sr) || (dc && dc != sc)) continue;
                            int score = abs(dr) + abs(dc);
                            if (score > best_score) {
                                best = k; best_score = score; br = dr; bc = dc;
                            }
                        }
                        if (best < 0) break;
                        int j = $ctx.grid.neighbour(i, best)$;
                        if (j < 0) break;
                        i = j;
                        rem_r -= br;
                        rem_c -= bc;
                    }
                }
                if (!$ctx.grid.nodata(i)$ && __ldcg(&Qacc[i]) > 0.0) break;
                if (attempt + 1 >= SPAWN_TRIES) {
                    i = base_cell;
                    break;
                }
                counts[7] += 1.0;
            }
            counts[0] += 1.0;
            if ($ctx.grid.nodata(i)$ || !(__ldcg(&Qacc[i]) > 0.0)) {
                counts[3] += 1.0;
                continue;
            }
        } else {
            i = stack[--stack_n];
        }
        double floor_head = (double)z[i] + (double)__ldcg(&h[i]);
        int hop = 0;
        for (; hop < max_hops; ++hop) {
            if ($ctx.grid.can_out(i)$) {
                if (walking) counts[1] += 1.0;
                break;
            }
            unsigned int visit = atomicAdd(&visits[i], 1u);
            int nb[8];
            for (int k = 0; k < nk; ++k) {
                int j = $ctx.grid.neighbour(i, k)$;
                nb[k] = (j >= 0 && !$ctx.grid.nodata(j)$) ? j : -1;
            }

            if (atomicCAS(&locks[i], 0, 1) != 0) {
                counts[4] += 1.0;
            } else {
                counts[walking ? 5 : 6] += 1.0;
                float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
                float alpha = fmaxf(1.0f + $ctx.FRICTION_EXPONENT.get(i)$,
                                    1.0e-3f);
                float rain = $ctx.PRECIPITATION.get(i)$ * dx * dx;
                double zi = (double)z[i];
                double heads[8];
                double low_head = 1.0e300;
                float low_length = dx;
                for (int k = 0; k < nk; ++k) {
                    heads[k] = nb[k] >= 0
                        ? (double)z[nb[k]] + (double)__ldcg(&h[nb[k]]) : 1.0e300;
                    if (heads[k] < low_head) {
                        low_head = heads[k];
                        low_length = $ctx.grid.dist(i, k)$;
                    }
                }
                float h_old = fmaxf(__ldcg(&h[i]), 0.0f);
                double head = zi + (double)h_old;
                // A pit fills to just above its lowest neighbour.
                if (low_head < 1.0e299 && head - low_head <= drop_min) {
                    h_old = (float)(low_head + 2.0 * fmax(drop_min, 1.0e-6) - zi);
                    head = zi + (double)h_old;
                    h[i] = h_old;
                }

                // Discharge: rain plus the sends of the currently higher
                // neighbours.
                double q = (double)rain;
                for (int k = 0; k < nk; ++k)
                    if (nb[k] >= 0 && heads[k] - head > drop_min)
                        q += (double)__ldcg(&pushed[nb[k] * nk + $ctx.grid.reverse(k)$]);
                Qacc[i] = q;

                // Rewrite the sends from that discharge.
                float w[8];
                float w_sum = 0.0f;
                float best_slope = 0.0f, width = dx, length = dx;
                double receiver_head = 1.0e300;
                for (int k = 0; k < nk; ++k) {
                    w[k] = 0.0f;
                    if (nb[k] < 0) continue;
                    double drop = head - heads[k];
                    if (!(drop > drop_min)) continue;
                    w[k] = (float)drop;
                    w_sum += w[k];
                    float dist = $ctx.grid.dist(i, k)$;
                    if ((float)(drop / dist) > best_slope) {
                        best_slope = (float)(drop / dist);
                        width = dist;
                        length = dist;
                        receiver_head = heads[k];
                    }
                }
                for (int k = 0; k < nk; ++k) {
                    float out = w_sum > 0.0f
                              ? (float)((double)(w[k] / w_sum) * q) : 0.0f;
                    float old = __ldcg(&pushed[i * nk + k]);
                    pushed[i * nk + k] = out;
                    if (propagate && nb[k] >= 0 && stack_n < 16
                            && fabsf(out - old)
                               > tolerance * fmaxf(fmaxf(out, old), rain)
                            && !$ctx.grid.can_out(nb[k])$)
                        stack[stack_n++] = nb[k];
                }

                // Relax h against the steepest lower neighbour, else the
                // lowest one.
                if (!(best_slope > 0.0f)) {
                    receiver_head = low_head;
                    length = low_length;
                    width = low_length;
                }
                if (receiver_head < 1.0e299)
                    h[i] = $ctx.hupdate(h_old, (float)q, width,
                                        (float)(zi - receiver_head), length,
                                        manning, alpha, relaxation)$;
                __threadfence();
                atomicExch(&locks[i], 0);
            }
            if (!walking) break;

            // Move on the current surface: neighbours below the path floor,
            // else the DAG.
            double head = (double)z[i] + (double)__ldcg(&h[i]);
            floor_head = fmin(floor_head, head);
            float scores[8];
            float score_sum = 0.0f;
            for (int k = 0; k < nk; ++k) {
                scores[k] = 0.0f;
                if (nb[k] < 0) continue;
                double other = (double)z[nb[k]] + (double)__ldcg(&h[nb[k]]);
                if (floor_head - other > drop_min) {
                    scores[k] = (float)(head - other);
                    score_sum += scores[k];
                }
            }
            if (!(score_sum > 0.0f)) {
                unsigned int mask = (unsigned int)dirs[i];
                for (int k = 0; k < nk; ++k) {
                    scores[k] = (mask & (1u << k)) ? weights[i * nk + k] : 0.0f;
                    score_sum += scores[k];
                }
            }
            if (!(score_sum > 0.0f)) {
                counts[3] += 1.0;
                break;
            }
            float pick = gfp_weyl(visit, (unsigned int)i ^ seed) * score_sum;
            int chosen = -1;
            for (int k = 0; k < nk; ++k) {
                if (scores[k] <= 0.0f) continue;
                chosen = k;
                pick -= scores[k];
                if (pick <= 0.0f) break;
            }
            i = $ctx.grid.neighbour(i, chosen)$;
        }
        if (walking && hop == max_hops) counts[2] += 1.0;
    }
}
'''


def newton_depth_helper():
    """Frozen helper ``(h, q, width, drop_base, length, manning, alpha,
    relaxation) -> h`` of ``_HUPDATE_NEWTON`` under a unique name."""
    text = _HUPDATE_NEWTON.replace("HUPDATE", f"gfp_hupdate{new_uid()}")
    return HelperBuilder(text.replace("MIN_SLOPE", f"{_MIN_SLOPE}f")).freeze()


def walk_factory(be, bundles, config):
    """Particle walk with the newton depth update."""
    return _kernel(be, bundles, _WALK, int(config["threads"]),
                   helpers={"hupdate": newton_depth_helper()})


_OUTFLOW = r'''
// Final fields: Qi from the discharge field and Qo the Manning outflow of
// the current depth at the steepest slope of the live surface (floored at
// MIN_SLOPE; an outlet passes its discharge). stats: sum|Qo - Qi|, sum Qi
// and the number of cells counted (valid cells that are not outlets).
extern "C" __global__ void gfp_outflow(
        const float* z, const float* h, const double* Qacc,
        float* Qi, float* Qo, float* stats) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    if ($ctx.grid.nodata(i)$) {
        Qi[i] = 0.0f;
        Qo[i] = 0.0f;
        return;
    }
    float q = (float)Qacc[i];
    Qi[i] = q;
    if ($ctx.grid.can_out(i)$) {
        Qo[i] = q;
        return;
    }
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    float dx = $ctx.grid.DX.get(0)$;
    float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
    float alpha = fmaxf(1.0f + $ctx.FRICTION_EXPONENT.get(i)$, 1.0e-3f);
    float depth = fmaxf(h[i], 0.0f);
    double head = (double)z[i] + (double)depth;
    float best_slope = 0.0f;
    float width = dx;
    for (int k = 0; k < nk; ++k) {
        int j = $ctx.grid.neighbour(i, k)$;
        if (j < 0 || $ctx.grid.nodata(j)$) continue;
        double drop = head - ((double)z[j] + (double)h[j]);
        if (drop <= 0.0) continue;
        float dist = $ctx.grid.dist(i, k)$;
        float slope = (float)(drop / (double)dist);
        if (slope > best_slope) {
            best_slope = slope;
            width = dist;
        }
    }
    float qout = width / manning * powf(depth, alpha)
               * sqrtf(fmaxf(best_slope, MIN_SLOPE));
    Qo[i] = qout;
    atomicAdd(&stats[0], fabsf(qout - q));
    atomicAdd(&stats[1], q);
    atomicAdd(&stats[2], 1.0f);
}
'''


def outflow_factory(be, bundles, config):
    return _kernel(be, bundles, _OUTFLOW, _cells(config))
