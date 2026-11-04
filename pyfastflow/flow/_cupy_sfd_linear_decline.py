"""Direct implicit linear-decline solve on a local SFD receiver forest.

Each sweep uses a persistent work queue. Donors are found through grid
neighbours, so receiver links must remain adjacent, including periodic wraps.
No grid-wide device barrier or donor CSR is needed.
"""

from pyfastflow.core import KernelBuilder, RoutineBuilder, new_uid

def build_sfd_linear_decline(*, grid, n_flat: int, n_neighbours: int):
    """Build a CuPy implicit solve for ``E = phi*S - psi*Q``.

    The caller applies uplift before this solve. ``z`` is then the old
    elevation for this fluvial substep. ``q`` is the resulting outgoing
    sediment volume flux per unit time.
    """
    n = int(n_flat)
    nk = int(n_neighbours)
    tag = f"sfdld{new_uid()}"
    # The active frontier becomes narrow near the roots. A small resident
    # worker pool avoids thousands of idle threads contending on one queue.
    workers, threads = 4, 256
    resident = workers * threads

    reset = KernelBuilder(f'''
extern "C" __global__ void {tag}_reset(int* queue, int* count) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {n}) queue[i] = -1;
    if (i < 4) count[i] = 0;
}}''', domain=max(n, 4)).freeze()

    prepare = KernelBuilder(f'''
extern "C" __global__ void {tag}_prepare(
        const int* rec, const float* z, float* old_z,
        int* remaining, float* sum_q0, float* sum_qp,
        int* queue, int* count) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    old_z[i] = z[i];
    sum_q0[i] = sum_qp[i] = 0.0f;
    int degree = 0;
    if ($ctx.grid.is_active(i)$) {{
        atomicAdd(&count[3], 1);
        #pragma unroll
        for (int k = 0; k < {nk}; ++k) {{
            int j = $ctx.grid.neighbour(i, k)$;
            if (j < 0 || j == i || rec[j] != i) continue;
            bool seen = false;
            for (int earlier = 0; earlier < k; ++earlier)
                if ($ctx.grid.neighbour(i, earlier)$ == j) seen = true;
            if (!seen) ++degree;
        }}
    }}
    remaining[i] = degree;
    if ($ctx.grid.is_active(i)$ && degree == 0) {{
        int pos = atomicAdd(&count[1], 1);
        queue[pos] = i;
    }}
}}''', domain=n).compose("grid", grid).freeze()

    downstream = KernelBuilder(f'''
extern "C" __global__ void {tag}_downstream(
        const int* rec, const float* z, const float* phi, const float* psi,
        float* sum_q0, float* sum_qp, float* q0, float* qp,
        int* remaining, int* queue, int* count) {{
    int lane = threadIdx.x & 31;
    while (true) {{
        int base = -1;
        int batch = 0;
        if (lane == 0) while (base < 0) {{
            volatile int* counters = (volatile int*)count;
            if (counters[2] >= counters[3]) break;
            int head = counters[0];
            int available = counters[1] - head;
            if (available > 0) {{
                int take = min(available, 32);
                if (atomicCAS(&count[0], head, head + take) == head) {{
                    base = head;
                    batch = take;
                }}
            }} else {{
#if __CUDA_ARCH__ >= 700
                __nanosleep(256);
#endif
            }}
        }}
        base = __shfl_sync(0xffffffffu, base, 0);
        batch = __shfl_sync(0xffffffffu, batch, 0);
        if (base < 0) return;
        if (lane < batch) {{
        int slot = base + lane;
        volatile int* ready = (volatile int*)queue;
        int i;
        while ((i = ready[slot]) < 0) {{
#if __CUDA_ARCH__ >= 700
            __nanosleep(256);
#endif
        }}
        int r = rec[i];
        if (r >= 0 && r < {n} && r != i && $ctx.grid.is_active(r)$) {{
            float dt = fmaxf($ctx.DT.get(i)$, 0.0f);
            float s = $ctx.grid.area(i)$;
            float alpha = s - dt * sum_qp[i];
            float beta = sum_q0[i];
            float d = fmaxf($ctx.grid.dist_between_nodes(i, r)$, 1.0e-12f);
            float k = fmaxf(phi[i], 0.0f) / d;
            float g = fmaxf(psi[i], 0.0f);
            float den = 1.0f + dt * k + alpha * g;
            float a = (alpha * k * (z[i] - z[r])
                       + beta * (1.0f + dt * k)) / den;
            float b = -alpha * k / den;
            q0[i] = a;
            qp[i] = b;
            atomicAdd(&sum_q0[r], a);
            atomicAdd(&sum_qp[r], b);
            __threadfence();
            if (atomicSub(&remaining[r], 1) == 1) {{
                int pos = atomicAdd(&count[1], 1);
                queue[pos] = r;
                __threadfence();
            }}
        }} else {{
            q0[i] = qp[i] = 0.0f;
        }}
        __threadfence();
        }}
        __syncwarp();
        if (lane == 0) atomicAdd(&count[2], batch);
    }}
}}''', domain=resident, block=threads).compose("grid", grid).freeze()

    prepare_upstream = KernelBuilder(f'''
extern "C" __global__ void {tag}_prepare_upstream(
        const int* rec, int* queue, int* count) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n} || !$ctx.grid.is_active(i)$) return;
    atomicAdd(&count[3], 1);
    int r = rec[i];
    if (r < 0 || r >= {n} || r == i || !$ctx.grid.is_active(r)$) {{
        int pos = atomicAdd(&count[1], 1);
        queue[pos] = i;
    }}
}}''', domain=n).compose("grid", grid).freeze()

    upstream = KernelBuilder(f'''
extern "C" __global__ void {tag}_upstream(
        const int* rec, float* z, const float* old_z,
        const float* phi, const float* psi,
        const float* sum_q0, const float* sum_qp,
        const float* q0, const float* qp, float* q,
        int* queue, int* count) {{
    int lane = threadIdx.x & 31;
    while (true) {{
        int base = -1;
        int batch = 0;
        if (lane == 0) while (base < 0) {{
            volatile int* counters = (volatile int*)count;
            if (counters[2] >= counters[3]) break;
            int head = counters[0];
            int available = counters[1] - head;
            if (available > 0) {{
                int take = min(available, 32);
                if (atomicCAS(&count[0], head, head + take) == head) {{
                    base = head;
                    batch = take;
                }}
            }} else {{
#if __CUDA_ARCH__ >= 700
                __nanosleep(256);
#endif
            }}
        }}
        base = __shfl_sync(0xffffffffu, base, 0);
        batch = __shfl_sync(0xffffffffu, batch, 0);
        if (base < 0) return;
        if (lane < batch) {{
        int slot = base + lane;
        volatile int* ready = (volatile int*)queue;
        int i;
        while ((i = ready[slot]) < 0) {{
#if __CUDA_ARCH__ >= 700
            __nanosleep(256);
#endif
        }}
        int r = rec[i];
        float old = old_z[i];
        if (r >= 0 && r < {n} && r != i && $ctx.grid.is_active(r)$) {{
            float dt = fmaxf($ctx.DT.get(i)$, 0.0f);
            float qi = q0[i] + qp[i] * (z[r] - old_z[r]);
            float d = fmaxf($ctx.grid.dist_between_nodes(i, r)$, 1.0e-12f);
            float k = fmaxf(phi[i], 0.0f) / d;
            float g = fmaxf(psi[i], 0.0f);
            float next = (old + dt * (k * z[r] + g * qi))
                       / (1.0f + dt * k);
            z[i] = next;
            q[i] = qi;
        }} else if ($ctx.grid.can_out(i)$) {{
            q[i] = sum_q0[i];
        }} else {{
            float dt = fmaxf($ctx.DT.get(i)$, 0.0f);
            float s = $ctx.grid.area(i)$;
            float rise = dt > 0.0f
                ? dt * sum_q0[i] / (s - dt * sum_qp[i]) : 0.0f;
            z[i] = old + rise;
            q[i] = 0.0f;
        }}
        __threadfence();
        #pragma unroll
        for (int k = 0; k < {nk}; ++k) {{
            int j = $ctx.grid.neighbour(i, k)$;
            if (j < 0 || j == i || rec[j] != i) continue;
            bool seen = false;
            for (int earlier = 0; earlier < k; ++earlier)
                if ($ctx.grid.neighbour(i, earlier)$ == j) seen = true;
            if (seen) continue;
            int pos = atomicAdd(&count[1], 1);
            queue[pos] = j;
        }}
        __threadfence();
        }}
        __syncwarp();
        if (lane == 0) atomicAdd(&count[2], batch);
    }}
}}''', domain=resident, block=threads).compose("grid", grid).freeze()

    routine = RoutineBuilder()
    for name, kernel in (("reset_downstream", reset),
                         ("prepare", prepare), ("downstream", downstream),
                         ("reset_upstream", reset),
                         ("prepare_upstream", prepare_upstream),
                         ("upstream", upstream)):
        routine.step(name, kernel)
    return routine.freeze()


__all__ = ["build_sfd_linear_decline"]
