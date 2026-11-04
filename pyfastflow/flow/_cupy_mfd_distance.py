"""Grid-aware drainage coordinates on a fixed MFD DAG."""

from ..core import KernelBuilder, RoutineBuilder, new_uid
from ._cupy_mfd_accum import persistent_grid_block


def build_mfd_distance(*, grid, n_flat: int, n_neighbours: int,
                       blocks_per_sm: int = 1, threads: int = 256):
    """Return a persistent source/outlet distance routine for an MFD DAG."""
    n = int(n_flat)
    nn = int(n_neighbours)
    tag = f"md{new_uid()}"
    launch, block = persistent_grid_block(
        blocks_per_sm=blocks_per_sm, threads=threads,
    )
    resident = launch[0] * block[0]

    clear = KernelBuilder(
        f'''extern "C" __global__ void {tag}_clear(
                int* count, unsigned int* barrier) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < 2) count[i] = 0;
            if (i == 0) barrier[0] = 0u;
        }}''', domain=2,
    ).freeze()

    prepare_forward = KernelBuilder(
        f'''extern "C" __global__ void {tag}_prepare_forward(
                const unsigned char* dirs, float* upstream,
                int* remaining, int* frontier, int* count) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            upstream[i] = 0.0f;
            int degree = 0;
            if (!$ctx.grid.nodata(i)$) {{
                #pragma unroll
                for (int k = 0; k < {nn}; ++k) {{
                    int donor = $ctx.grid.neighbour(i, k)$;
                    if (donor == -1) continue;
                    int reverse = $ctx.grid.reverse(k)$;
                    if (dirs[donor] & (1u << reverse)) ++degree;
                }}
            }}
            remaining[i] = degree;
            if (!$ctx.grid.nodata(i)$ && degree == 0) {{
                int p = atomicAdd(&count[0], 1);
                frontier[p] = i;
            }}
        }}''', domain=n,
    ).compose("grid", grid).freeze()

    prepare_reverse = KernelBuilder(
        f'''extern "C" __global__ void {tag}_prepare_reverse(
                const unsigned char* dirs, float* downstream,
                int* remaining, int* frontier, int* count) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            downstream[i] = 0.0f;
            int degree = $ctx.grid.nodata(i)$ ? 0 : __popc((unsigned int)dirs[i]);
            remaining[i] = degree;
            if (!$ctx.grid.nodata(i)$ && degree == 0) {{
                int p = atomicAdd(&count[0], 1);
                frontier[p] = i;
            }}
        }}''', domain=n,
    ).compose("grid", grid).freeze()

    def walk(reverse):
        if reverse:
            propagation = f'''
                #pragma unroll
                for (int k = 0; k < {nn}; ++k) {{
                    int v = $ctx.grid.neighbour(u, k)$;
                    if (v == -1) continue;
                    int rk = {nn - 1} - k;
                    if (!(dirs[v] & (1u << rk))) continue;
                    float candidate = distance[u] + $ctx.grid.dist(u, k)$;
                    atomicMax((unsigned int*)&distance[v],
                              __float_as_uint(candidate));
                    __threadfence();
                    if (atomicAdd(&remaining[v], -1) == 1) {{
                        int q = atomicAdd(c_out, 1);
                        frontiers[1 - phase][q] = v;
                    }}
                }}'''
        else:
            propagation = f'''
                #pragma unroll
                for (int k = 0; k < {nn}; ++k) {{
                    if (!(dirs[u] & (1u << k))) continue;
                    int v = $ctx.grid.neighbour_raw(u, k)$;
                    float candidate = distance[u] + $ctx.grid.dist(u, k)$;
                    atomicMax((unsigned int*)&distance[v],
                              __float_as_uint(candidate));
                    __threadfence();
                    if (atomicAdd(&remaining[v], -1) == 1) {{
                        int q = atomicAdd(c_out, 1);
                        frontiers[1 - phase][q] = v;
                    }}
                }}'''
        return KernelBuilder(
            f'''extern "C" __global__ void {tag}_walk_{int(reverse)}(
                    const unsigned char* dirs, float* distance,
                    int* remaining, int* frontier0, int* frontier1,
                    int* count, unsigned int* barrier) {{
                __shared__ int s_size;
                int* frontiers[2] = {{frontier0, frontier1}};
                int phase = 0;
                unsigned int level = 0;
                // Rotating counters; see build_persistent_mfd.
                while (true) {{
                    int* c_out = &count[(level + 1) % 3];
                    if (threadIdx.x == 0)
                        s_size = *((volatile int*)&count[level % 3]);
                    __syncthreads();
                    int size = s_size;
                    if (size == 0) break;
                    int tid = blockIdx.x * blockDim.x + threadIdx.x;
                    int stride = gridDim.x * blockDim.x;
                    for (int p = tid; p < size; p += stride) {{
                        int u = frontiers[phase][p];
                        {propagation}
                    }}
                    __threadfence();
                    __syncthreads();
                    if (threadIdx.x == 0) {{
                        if (blockIdx.x == 0) {{
                            count[(level + 2) % 3] = 0;
                            __threadfence();
                        }}
                        unsigned int target = (level + 1) * gridDim.x;
                        atomicAdd(barrier, 1u);
                        while (*((volatile unsigned int*)barrier) < target) {{
#if __CUDA_ARCH__ >= 700
                            __nanosleep(64);
#endif
                        }}
                    }}
                    __syncthreads();
                    ++level;
                    phase = 1 - phase;
                }}
            }}''', domain=resident, block=threads,
        ).compose("grid", grid).freeze()

    clear_max = KernelBuilder(
        f'''extern "C" __global__ void {tag}_clear_max(float* maximum) {{
            if (blockIdx.x == 0 && threadIdx.x == 0) maximum[0] = 0.0f;
        }}''', domain=1,
    ).freeze()

    find_max = KernelBuilder(
        f'''extern "C" __global__ void {tag}_find_max(
                const float* downstream, float* maximum) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n})
                atomicMax((unsigned int*)maximum,
                          __float_as_uint(downstream[i]));
        }}''', domain=n,
    ).freeze()

    normalize = KernelBuilder(
        f'''extern "C" __global__ void {tag}_normalize(
                const float* downstream, const float* maximum,
                float* position) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.nodata(i)$) {{ position[i] = -1.0f; return; }}
            float scale = maximum[0];
            position[i] = scale > 0.0f ? downstream[i] / scale : 0.0f;
        }}''', domain=n,
    ).compose("grid", grid).freeze()

    return (RoutineBuilder()
            .step("clear_forward", clear)
            .step("prepare_forward", prepare_forward)
            .step("walk_forward", walk(False))
            .step("clear_reverse", clear)
            .step("prepare_reverse", prepare_reverse)
            .step("walk_reverse", walk(True))
            .step("clear_max", clear_max)
            .step("find_max", find_max)
            .step("normalize", normalize).freeze())
