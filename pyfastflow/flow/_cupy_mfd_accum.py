"""
cupy-only persistent-kernel MFD accumulation ("persistent_mfd" method of
make_accumulation), on the new builder/frozen/bound stack (../core/context/
builder.py, frozen.py, bound.py) - see make_accumulation's own docstring for
the call-site contract.

No Taichi/Quadrants counterpart, and there never should be one: the
mechanism is a hand-rolled, monotonic grid-wide barrier (a global atomic
counter every block increments then spins on, level by level) around a
persistent kernel that is launched exactly once and loops internally over
levels - built on CUDA's raw `__global__`/`__shared__`/`atomicAdd`/
`__threadfence()` primitives, none of which the closure backends' kernel
model (one `ti.kernel`/Quadrants kernel per launch, no persistent-thread
loop, no portable grid-wide barrier inside one kernel) can express.

Algorithm (level-synchronous, double-buffered frontier, shared-memory
staging): each level, every resident thread of a fixed, small grid pulls
node indices from `frontier[p]` (size `count[p]`, grid-stride loop). For
node u, the receiver mask `dirs[u]` (bit k set == direction k, this grid's
own D8 numbering) picks which of `mfd_w[u*NN + k]` weights to atomic-add
`accum[u]` into `accum[neighbour]` (`ctx.grid.neighbour_raw(u, k)` - the
grid's own raw neighbour arithmetic, trusted the same way the mask itself is
trusted to have only ever set bits for directions the topology already
validated). A `__threadfence()` publishes every one of those writes before
any thread decrements `indegree[neighbour]`; a decrement that lands the
count exactly on zero stages that neighbour into a per-block shared buffer
(`s_buf`, capacity `fr_stage`), spilling to a direct `atomicAdd` into
`count[1-p]` past that capacity. After the grid-stride loop, each block
flushes its staged cells into `frontier[1-p]` through one reserved
contiguous range (one `atomicAdd` per block, not per cell), fences again,
then every block increments the global `barrier` counter and spins until it
reads `(level+1) * gridDim.x` - the point every block has published
everything for this level - before moving on. The loop exits once
`count[p]`, reloaded through a volatile pointer each iteration, is zero.

`accum` is seeded through a `SOURCE` PARAM slot (any mode - a caller binds a
Parameter there after `.build()`) by a separate `q_init` kernel, not
hardcoded to 1.0: the persistent kernel's very first level reads `accum[u]`
for every cell already in the initial frontier before any atomic_add has
landed on it, so that seed must be a prior, real, finished launch - the same
reasoning `_cupy_accum.py`'s `build_atomic` splits `q_init`/`accum` on.

`dirs`, `mfd_w`, `indegree`, `frontier0`, `frontier1`, `count`, `barrier`
are all caller-supplied data args, exactly like `rec` is for the SFD
methods in `_cupy_accum.py` - this module does not build MFD topology
(mask/weights/indegree computation), only accumulation over an
already-built one. `count` is 2 int32 (`count[0]`, `count[1]`, the two
ping-pong frontier sizes) and `barrier` is 1 uint32; initializing them
(`count[p] = n0` from the ready-cell count, `count[1-p] = 0`,
`barrier[0] = 0`, `frontier[p][:n0] = <ready cell indices>`) is the
caller's job before every call - see `init_frontier_mfd` below for the
host-side compaction step, kept as a plain function rather than a builder
member since it is pure host/cupy indexing, not a kernel.

`n_neighbours` (D4=4/D8=8) is a required build-time python int, not read off
`grid` the way the pre-port script read it from a bound Parameter: this
factory's `grid` is a bare `make_grid_group` FrozenGroup (structure only, no
bound Parameter values - see ../grid/__init__.py's own module docstring), so
there is no build-time value to read off it, the same reason
make_accumulation's own `method="atomic"` requires `n_flat` explicitly under
this stack. Baking it in as `{NN}` (rather than reading it at runtime via
`ctx.grid.N_NEIGHBOURS.get(0)`, the pattern every other ported flow block
uses) preserves the pre-port script's `#pragma unroll` on the two per-node
direction loops - deliberately kept, not dropped for uniformity with those
other blocks, since this is the one hot loop in the package still doing a
fully unrolled fixed-trip-count neighbour walk.

Author: B.G (08/2026)
"""

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

    Author: B.G (08/2026)
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

    Author: B.G (08/2026)
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

    Returns
    -------
    dict
        {"q_init": FrozenKernel, "accum": FrozenKernel}.

    Author: B.G (08/2026)
    """
    NN = int(n_neighbours)
    t = f"pm{new_uid()}"
    persistent_grid, persistent_block = persistent_grid_block(
        blocks_per_sm=blocks_per_sm, threads=threads,
    )
    resident_threads = persistent_grid[0] * persistent_block[0]

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
    const unsigned char* __restrict__ dirs, const float* __restrict__ mfd_w,
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
            #pragma unroll
            for (int k = 0; k < {NN}; k++) {{
                if (!(mask & (1u << k))) continue;
                int r = $ctx.grid.neighbour_raw(u, k)$;
                atomicAdd(&accum[r], au * mfd_w[base + k]);
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
