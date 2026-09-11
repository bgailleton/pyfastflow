"""Persistent CUDA kernel for multiple-flow accumulation."""

import cupy as cp

from ..core import KernelBuilder, new_uid


def persistent_grid_block(*, blocks_per_sm: int = 2, threads: int = 256) -> tuple:
    """
    (grid, block) launch dims for the persistent kernel: `blocks_per_sm *
    <this device's SM count>` blocks,
    of `threads` threads each - queried from the current cupy device, not
    sized off n_flat the way every other launch in this package is (the
    frontier itself, not the whole node range, bounds how much work a
    level does).

    Parameters
    ----------
    blocks_per_sm : int, optional
        Default 2.
    threads : int, optional
        Default 256.

    Returns
    -------
    tuple
        (grid, block) launch dims.

    """
    sm_count = cp.cuda.Device().attributes["MultiProcessorCount"]
    return (blocks_per_sm * sm_count,), (threads,)


def init_frontier_mfd(indegree_data, frontier_data) -> int:
    """
    Host-side frontier compaction: writes the flat indices of every cell
    with indegree 0 into the front of `frontier_data` (a raw cupy ndarray,
    e.g. a DataHandle's `.array`) and returns how many there were - the
    `count[p]` the caller must then store before the first launch.

    Plain cupy indexing, not a kernel: `cp.nonzero` has no equivalent
    device-side primitive this package's span/template mechanism reaches,
    and the reference implementation this ports does the same compaction
    as an ordinary host op rather than a custom kernel.

    Parameters
    ----------
    indegree_data, frontier_data : cupy.ndarray

    Returns
    -------
    int
        Count of cells with indegree 0.

    """
    ready = cp.nonzero(indegree_data == 0)[0].astype(cp.int32)
    n = int(ready.size)
    frontier_data[:n] = ready
    return n


def build_persistent_mfd(
    *,
    grid,
    n_flat: int,
    n_neighbours: int,
    fr_stage: int = 2048,
    blocks_per_sm: int = 2,
    threads: int = 256,
    quantized_weight: bool = False,
):
    """
    Two FrozenKernels (new builder/frozen/bound stack): "q_init" (composes
    `grid`, data arg (accum,), ordinary grid-stride over n_flat: accum[i] =
    nodata(i) ? 0 : SOURCE.get(i) - the nodata gate keeps a nodata cell's
    sentinel `filled` from injecting a spurious source unit into the live
    domain, so a caller must bind q_init's own `grid` PARAM leaves too, not
    just SOURCE/accum) and "accum" (data args (frontier0, frontier1, count,
    barrier, dirs,
    mfd_w, accum, indegree), the persistent kernel described in the module
    docstring). Both are bare FrozenKernels, not a Sequence - "q_init" is
    one ordinary n_flat-sized launch, "accum" is one persistent launch on
    `persistent_grid_block(...)`'s dims; there is no per-round host loop to
    sequence, unlike rake_compress/pointer_jump_push. A caller `.build()`s
    each, binds "q_init"'s `SOURCE` PARAM slot and both kernels' composed
    `grid`, then calls `.compile()` on each. The q-init kernel declares an
    n_flat-sized domain; the accumulation kernel stores
    `persistent_grid_block(blocks_per_sm=..., threads=...)` as its fixed
    resident domain, so callers never pass launch dimensions.

    `fr_stage` sizes the per-block shared staging buffer (`s_buf`) baked
    into "accum"'s generated source as a compile-time array length - a
    smaller value uses less shared memory per block at the cost of more
    direct-scatter spills past capacity.

    Parameters
    ----------
    grid : FrozenGroup
    n_flat, n_neighbours : int
    fr_stage, blocks_per_sm, threads : int, optional
        Default 2048.
    quantized_weight : bool, optional
        Consume max-normalized unsigned-byte scores rather than float32
        weights. Their integer sum is normalized during scattering.

    Returns
    -------
    dict
        {"q_init": FrozenKernel, "accum": FrozenKernel}.

    """
    NN = int(n_neighbours)
    t = f"pm{new_uid()}"
    persistent_grid, persistent_block = persistent_grid_block(
        blocks_per_sm=blocks_per_sm, threads=threads,
    )
    resident_threads = persistent_grid[0] * persistent_block[0]
    weight_type = "unsigned char" if quantized_weight else "float"
    weight_sum = (
        f"""int weight_sum = 0;
            #pragma unroll
            for (int k = 0; k < {NN}; k++)
                if (mask & (1u << k)) weight_sum += (int)mfd_w[base + k];"""
        if quantized_weight else ""
    )
    weight_value = (
        "(float)mfd_w[base + k] / (float)weight_sum"
        if quantized_weight else "mfd_w[base + k]"
    )

    q_init = (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_q_init(float* accum) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    accum[i] = $ctx.grid.nodata(i)$ ? 0.0f : $ctx.SOURCE.get(i)$;
}}
""", domain=n_flat)
        .compose("grid", grid)
        .freeze()
    )

    accum = (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_persistent_mfd(
    int* __restrict__ frontier0, int* __restrict__ frontier1,
    int* __restrict__ count, unsigned int* __restrict__ barrier,
    const unsigned char* __restrict__ dirs, const {weight_type}* __restrict__ mfd_w,
    float* __restrict__ accum, int* __restrict__ indegree)
{{
    __shared__ int s_buf[{fr_stage}];
    __shared__ int s_n;
    __shared__ unsigned int s_base;

    int* frontiers[2] = {{ frontier0, frontier1 }};
    int p = 0;
    unsigned int level = 0;

    while (true) {{
        int size_in = *((volatile int*)&count[p]);
        if (size_in == 0) break;
        int* fin  = frontiers[p];
        int* fout = frontiers[1 - p];

        if (threadIdx.x == 0) s_n = 0;
        __syncthreads();

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = gridDim.x * blockDim.x;
        for (int idx = tid; idx < size_in; idx += stride) {{
            int u = fin[idx];
            float au = accum[u];
            unsigned int mask = (unsigned int)dirs[u];
            int base = u * {NN};
            {weight_sum}
            #pragma unroll
            for (int k = 0; k < {NN}; k++) {{
                if (!(mask & (1u << k))) continue;
                int r = $ctx.grid.neighbour_raw(u, k)$;
                atomicAdd(&accum[r], au * ({weight_value}));
            }}
            __threadfence();
            #pragma unroll
            for (int k = 0; k < {NN}; k++) {{
                if (!(mask & (1u << k))) continue;
                int r = $ctx.grid.neighbour_raw(u, k)$;
                int old = atomicAdd(&indegree[r], -1);
                if (old == 1) {{
                    int sp = atomicAdd(&s_n, 1);
                    if (sp < {fr_stage}) s_buf[sp] = r;
                    else {{ int pos = atomicAdd(&count[1 - p], 1); fout[pos] = r; }}
                }}
            }}
        }}

        __syncthreads();
        int n_flush = min(s_n, {fr_stage});
        if (threadIdx.x == 0)
            s_base = atomicAdd((unsigned int*)&count[1 - p], (unsigned int)n_flush);
        __syncthreads();
        for (int i = threadIdx.x; i < n_flush; i += blockDim.x)
            fout[s_base + i] = s_buf[i];
        __threadfence();

        __syncthreads();
        if (threadIdx.x == 0) {{
            if (blockIdx.x == 0) count[p] = 0;
            unsigned int target = (level + 1) * (unsigned int)gridDim.x;
            atomicAdd(barrier, 1u);
            unsigned int ns = 32;
            while (*((volatile unsigned int*)barrier) < target) {{
#if __CUDA_ARCH__ >= 700
                __nanosleep(ns);
                if (ns < 1024) ns <<= 1;
#endif
            }}
        }}
        __syncthreads();

        level++;
        p = 1 - p;
    }}
}}
""", domain=resident_threads, block=threads)
        .compose("grid", grid)
        .freeze()
    )

    return {"q_init": q_init, "accum": accum}
