"""CUDA templates for flow accumulation."""

from ..core import HelperBuilder, KernelBuilder, SequenceBuilder, new_uid


def build_ping_pong_helpers():
    """
    get_src(src, tid)/update_src(src, tid, flip) HelperBuilders - see
    _closure_accum.py's build_ping_pong_helpers for the sign/magnitude
    encoding and the ITER-sharing note (both apply identically here).

    """
    get_src = HelperBuilder(
        """
__device__ int pf_get_src(const int* src, int tid) {
    int entry = src[tid];
    int it = $ctx.ITER.get(0)$;
    int flip = entry < 0;
    if (abs(entry) == (it + 1)) flip = !flip;
    return flip;
}
"""
    ).freeze()
    update_src = HelperBuilder(
        """
__device__ void pf_update_src(int* src, int tid, int flip) {
    int it = $ctx.ITER.get(0)$;
    src[tid] = (flip ? 1 : -1) * (it + 1);
}
"""
    ).freeze()
    return get_src, update_src


def build_atomic(*, n_flat: int):
    """
    Two FrozenKernels (new builder/frozen/bound stack - ../core/context/
    builder.py, frozen.py, bound.py), data args (q,) and (rec, q): "q_init"
    (q[i] = SOURCE.get(i)) must be run, and finish, before "accum" (the
    atomic descent). Two real launches rather than one, unlike the closure
    backends' single KernelBuilder (see _closure_accum.py's build_atomic) -
    a single CUDA __global__ has no portable grid-wide barrier the way two
    consecutive top-level Taichi/Quadrants for-loops do, so without a second
    launch a thread could atomicAdd into q[j] before node j's own thread has
    run its q[j] = SOURCE.get(j) initialization.

    `SOURCE` is each kernel's own wired PARAM slot (any mode) - a caller
    binds a Parameter there, on each, after `.build()`; there is no Need
    indirection in this stack. ``q`` remains DATA because CUDA updates it
    with ``atomicAdd``.

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    dict
        {"q_init": FrozenKernel, "accum": FrozenKernel}.

    """
    t = f"pa{new_uid()}"
    q_init = (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_q_init(float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q[i] = $ctx.SOURCE.get(i)$;
}}
"""
        , domain=n_flat).freeze()
    )
    # wi is re-read from `SOURCE`, not from q[i]: q[i] is a live accumulation
    # target every other thread may already be atomic-adding into by the
    # time this thread runs (thread order across blocks is not guaranteed),
    # so reading q[i] here would race against those writes and silently
    # inflate wi with contributions that arrived early - exactly the bug an
    # earlier version of this kernel had (q[i] as a shortcut for wi, to avoid
    # re-binding `source`), caught by a max_abs deviation of ~1e23 at
    # n_flat=1e6 in _verify_accum.py.
    # accum_downstream_atomic_kernel and the closure-backend port, which
    # both re-read the weight function/Parameter directly for this reason.
    accum = (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_accum_downstream_atomic(const int* rec, float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (rec[i] == i) return;
    float wi = $ctx.SOURCE.get(i)$;
    int j = rec[i];
    int guard = 0;
    while (j != rec[j] && guard < {n_flat}) {{
        atomicAdd(&q[j], wi);
        j = rec[j];
        guard++;
    }}
    atomicAdd(&q[j], wi);
}}
"""
        , domain=n_flat).freeze()
    )
    return {"q_init": q_init, "accum": accum}


def build_rake_compress(*, n_neighbours: int, logn: int, n_flat: int):
    """
    SequenceBuilder for the rake-and-compress accumulation, plus the
    KernelBuilders it is made of - see _closure_accum.py's build_rake_compress
    for the step sequence and the loop-vs-unroll design choice.

    Unlike the closure backends, the ITER bump is a separate composed step
    ("bump_iteration", not folded into rake_compress_accum's own body): a
    single CUDA `__global__` has no guarantee that a second top-level loop
    runs strictly after the first the way two consecutive Taichi/Quadrants
    `for` loops do, so the loop body here is `["rake_step",
    "bump_iteration"]`, two real launches per round, mirroring build_atomic's
    own closure/cupy split for the same reason. Composed names: "zero_init",
    "reset_iteration", "q_init", "receivers_to_donors", "rake_step" (the
    rake_compress_accum kernel), "bump_iteration", "decrement_iteration",
    "fuse_accum_buffers". PARAM addresses needing the same bound Parameter:
    "q_init.SOURCE" (SOURCE); "reset_iteration.ITER", "rake_step.ITER" (its
    own `share("ITER", "get_src.ITER", "update_src.ITER")` collapses its two
    composed helpers' own ITER occurrences into this one address),
    "bump_iteration.ITER", "decrement_iteration.ITER",
    "fuse_accum_buffers.get_src.ITER" (ITER - five addresses; one fewer than
    the closure backend's four plus the extra "bump_iteration.ITER" this
    backend needs in place of the closure's folded-in bump).

    Every kernel here launches over an n_flat-sized index space except the
    single-thread iteration bookkeeping steps ("reset_iteration",
    "bump_iteration", "decrement_iteration"). Each kernel owns its domain
    and optional block size, so sequence compilation needs no per-step launch
    overrides.

    Parameters
    ----------
    n_neighbours : int
    logn : int
    n_flat : int

    Returns
    -------
    tuple[SequenceBuilder, dict]

    """
    NN = n_neighbours
    t = f"pr{new_uid()}"

    get_src, update_src = build_ping_pong_helpers()

    zero_init = KernelBuilder(
        f"""
extern "C" __global__ void {t}_zero_init(int* ndonors, int* ndonors_alt, int* src) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    ndonors[i] = 0;
    ndonors_alt[i] = 0;
    src[i] = 0;
}}
""", domain=n_flat).freeze()
    reset_iteration = KernelBuilder(
        f"""
extern "C" __global__ void {t}_reset_iteration() {{
    $ctx.ITER.set_node(0, 0)$;
}}
""", domain=1, block=1).freeze()
    decrement_iteration = KernelBuilder(
        f"""
extern "C" __global__ void {t}_decrement_iteration() {{
    int cur = $ctx.ITER.get(0)$;
    $ctx.ITER.set_node(0, cur - 1)$;
}}
""", domain=1, block=1).freeze()
    # Unlike the closure backends (rake_compress_accum's own second
    # top-level `for` loop bumps ITER, ordered after the rake pass as a
    # separate offloaded task for free), cupy has no such guarantee inside
    # one `__global__` - a genuinely separate, single-thread launch is
    # required after every rake pass, mirroring the closure/cupy split
    # build_atomic already has for its own barrier reason.
    bump_iteration = KernelBuilder(
        f"""
extern "C" __global__ void {t}_bump_iteration() {{
    int cur = $ctx.ITER.get(0)$;
    $ctx.ITER.set_node(0, cur + 1)$;
}}
""", domain=1, block=1).freeze()
    q_init = KernelBuilder(
        f"""
extern "C" __global__ void {t}_q_init(float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q[i] = $ctx.SOURCE.get(i)$;
}}
""", domain=n_flat).freeze()
    receivers_to_donors = KernelBuilder(
        f"""
extern "C" __global__ void {t}_receivers_to_donors(const int* rec, int* donors, int* ndonors) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n_flat}) return;
    int rcv = rec[tid];
    if (rcv != tid) {{
        int old_val = atomicAdd(&ndonors[rcv], 1);
        donors[rcv * {NN} + old_val] = tid;
    }}
}}
""", domain=n_flat).freeze()
    rake_compress_accum = (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_rake_compress_accum(int* donors, int* ndonors, float* q, int* src,
                                         int* donors_alt, int* ndonors_alt, float* q_alt) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n_flat}) return;
    int flip = $ctx.get_src(src, tid)$;

    int worked = 0;
    int todo = flip ? ndonors_alt[tid] : ndonors[tid];
    int base = tid * {NN};
    int donors_local[{NN}];
    for (int t2 = 0; t2 < {NN}; t2++) donors_local[t2] = -1;
    float q_added = 0.0f;

    int i = 0;
    while (i < todo && i < {NN}) {{
        if (donors_local[i] == -1) {{
            donors_local[i] = flip ? donors_alt[base + i] : donors[base + i];
        }}
        int did = donors_local[i];

        int flip_donor = $ctx.get_src(src, did)$;
        int ndnr_val = flip_donor ? ndonors_alt[did] : ndonors[did];

        if (ndnr_val <= 1) {{
            if (!worked) {{
                q_added = flip ? q_alt[tid] : q[tid];
            }}
            worked = 1;
            float q_val = flip_donor ? q_alt[did] : q[did];
            q_added += q_val;
            if (ndnr_val == 0) {{
                todo -= 1;
                if (todo > i) {{
                    donors_local[i] = flip ? donors_alt[base + todo] : donors[base + todo];
                }}
                i -= 1;
            }} else {{
                int donor_base = did * {NN};
                donors_local[i] = flip_donor ? donors_alt[donor_base] : donors[donor_base];
            }}
        }}
        i += 1;
    }}
    if (worked) {{
        if (flip) {{
            ndonors[tid] = todo;
            q[tid] = q_added;
            for (int j = 0; j < {NN}; j++) {{
                if (j < todo) donors[base + j] = donors_local[j];
            }}
        }} else {{
            ndonors_alt[tid] = todo;
            q_alt[tid] = q_added;
            for (int j = 0; j < {NN}; j++) {{
                if (j < todo) donors_alt[base + j] = donors_local[j];
            }}
        }}
        $ctx.update_src(src, tid, flip)$;
    }}
}}
""", domain=n_flat)
        .param("ITER")
        .compose("get_src", get_src)
        .compose("update_src", update_src)
        .share("ITER", "get_src.ITER", "update_src.ITER")
        .freeze()
    )
    fuse_accum_buffers = (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_fuse_accum_buffers(float* q, int* src, float* q_alt) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n_flat}) return;
    if ($ctx.get_src(src, tid)$) {{
        q[tid] = q_alt[tid];
    }}
}}
""", domain=n_flat)
        .compose("get_src", get_src)
        .freeze()
    )

    kernels = {
        "zero_init": zero_init,
        "reset_iteration": reset_iteration,
        "bump_iteration": bump_iteration,
        "decrement_iteration": decrement_iteration,
        "q_init": q_init,
        "receivers_to_donors": receivers_to_donors,
        "rake_compress_accum": rake_compress_accum,
        "fuse_accum_buffers": fuse_accum_buffers,
    }

    sb = SequenceBuilder()
    sb.add("zero_init", zero_init)
    sb.add("reset_iteration", reset_iteration)
    sb.add("q_init", q_init)
    sb.add("receivers_to_donors", receivers_to_donors)
    sb.add("rake_step", rake_compress_accum)
    sb.add("bump_iteration", bump_iteration)
    sb.add("decrement_iteration", decrement_iteration)
    sb.add("fuse_accum_buffers", fuse_accum_buffers)

    sb.step("zero_init")
    sb.step("reset_iteration")
    sb.step("q_init")
    sb.step("receivers_to_donors")
    sb.loop(body=["rake_step", "bump_iteration"], max_times=logn + 1)
    sb.step("decrement_iteration")
    sb.step("fuse_accum_buffers")

    return sb, kernels


def build_pointer_jump_push(*, rounds: int, n_flat: int):
    """
    SequenceBuilder for the pointer-jump-push accumulation, plus the
    KernelBuilders it is made of - see _closure_accum.py's
    build_pointer_jump_push for the step sequence, retirement rule and the
    two-address ping-pong shape (identical composed names/addresses here).

    Unlike the closure backend's single accum_pointer_jump_push_step (two
    consecutive top-level `for` loops give the copy-then-push barrier for
    free), cupy needs two separate `__global__` launches per occurrence
    (step_copy, step_core) - a `__syncthreads()` only orders threads within
    one block, and a push may target an index owned by a thread in a
    different block, so only a real launch boundary guarantees the q_next
    copy has landed everywhere before any atomicAdd into it (mirrors
    _cupy_receivers.py's build_atomic q_init/accum split for the same
    reason). Each occurrence therefore composes two names, not one: "step_a"
    is ("step_a_copy", "step_a_core") and "step_b" is ("step_b_copy",
    "step_b_core"), both pairs referenced together in the loop body so a
    round is always copy-then-core, for both directions.

    Parameters
    ----------
    rounds : int
        Already rounded to even by the caller.
    n_flat : int

    Returns
    -------
    tuple[SequenceBuilder, dict]

    """
    t = f"pj{new_uid()}"
    q_init = KernelBuilder(
        f"""
extern "C" __global__ void {t}_q_init(float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q[i] = $ctx.SOURCE.get(i)$;
}}
""", domain=n_flat).freeze()
    copy_rec_to_work = KernelBuilder(
        f"""
extern "C" __global__ void {t}_copy_rec_to_work(const int* rec, int* work) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    work[i] = rec[i];
}}
""", domain=n_flat).freeze()
    step_copy = KernelBuilder(
        f"""
extern "C" __global__ void {t}_step_copy(const float* q_curr, float* q_next) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q_next[i] = q_curr[i];
}}
""", domain=n_flat).freeze()
    step_core = (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_step_core(const int* rec_curr, int* rec_next, const float* q_curr, float* q_next) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int parent = rec_curr[i];
    rec_next[i] = parent;
    if (parent != i) {{
        float wi = q_curr[i];
        if (wi != 0.0f) {{
            atomicAdd(&q_next[parent], wi);
        }}
        int grandparent = rec_curr[parent];
        rec_next[i] = (grandparent == parent) ? i : grandparent;
    }}
}}
""", domain=n_flat).freeze()
    )

    kernels = {
        "q_init": q_init,
        "copy_rec_to_work": copy_rec_to_work,
        "accum_pointer_jump_push_step_copy": step_copy,
        "accum_pointer_jump_push_step_core": step_core,
    }

    sb = SequenceBuilder()
    sb.add("q_init", q_init)
    sb.add("copy_rec_to_work", copy_rec_to_work)
    sb.add("step_a_copy", step_copy)
    sb.add("step_a_core", step_core)
    sb.add("step_b_copy", step_copy)
    sb.add("step_b_core", step_core)

    sb.step("q_init")
    sb.step("copy_rec_to_work")
    sb.loop(
        body=["step_a_copy", "step_a_core", "step_b_copy", "step_b_core"],
        max_times=rounds // 2,
    )

    return sb, kernels
