"""Taichi and Quadrants templates for flow accumulation."""

from ..core import HelperBuilder, KernelBuilder, SequenceBuilder
from ._closure_shared import _tensor_annotation


# ---------------------------------------------------------------------------
# accumulation: ping-pong src encoding (get_src/update_src, reading the
# iteration scalar Parameter instead of taking iteration as a call argument)
# ---------------------------------------------------------------------------


def _get_src_tmpl(ctx, src, tid):
    entry = src[tid]
    it = ctx.ITER.get(0)
    flip = entry < 0
    flip = (not flip) if (abs(entry) == (it + 1)) else flip
    return flip


def _update_src_tmpl(ctx, src, tid, flip):
    it = ctx.ITER.get(0)
    src[tid] = (1 if flip else -1) * (it + 1)


def build_ping_pong_helpers():
    """
    get_src(src, tid)/update_src(src, tid, flip) HelperBuilders - same
    sign/magnitude encoding as the shared ping-pong helpers, each wiring its
    own ``ITER`` parameter. A caller composing both into a kernel that also
    wires its own `ITER` (rake_compress_accum does not - it only reaches
    `ITER` through these two helpers) would `share("ITER", "get_src.ITER",
    "update_src.ITER")` to collapse the two occurrences; rake_compress_accum
    reaches `ITER` directly at its own top level in addition to composing
    both helpers, so it shares its own wired `ITER` with both instead (see
    build_rake_compress).

    """
    get_src = HelperBuilder(_get_src_tmpl).freeze()
    update_src = HelperBuilder(_update_src_tmpl).freeze()
    return get_src, update_src


def build_atomic(*, backend: str, backend_mod, n_flat: int):
    """
    accum_downstream_atomic KernelBuilder (new builder/frozen/bound stack -
    ../core/context/builder.py, frozen.py, bound.py), data args (rec, q):
    q[i] is initialized from `ctx.SOURCE.get(i)`, then every node walks its
    receiver chain to the root atomic-adding its own weight into each
    downstream node. Requires an acyclic receiver graph (run after
    depression handling); the `guard < n_flat` bound makes a cycle degrade
    the result instead of hanging, rather than guaranteeing correctness on
    one.

    `SOURCE` is this kernel's own wired PARAM slot (any mode - const,
    scalar or field) - a caller binds a Parameter there after `.build()`,
    exactly like any other PARAM slot. ``ctx.bk.atomic_add`` is what a genuinely concurrent
    accumulation into a DATA-typed `q` needs - PARAM access stays strict
    get()/set_node() (a plain, non-atomic write), so `q` is wired as DATA,
    not PARAM, the same "genuinely concurrent write" classification
    ../ops/__init__.py's Reduce already establishes for its own accumulator.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    n_flat : int

    Returns
    -------
    KernelBuilder

    """
    T = _tensor_annotation(backend_mod, backend)
    NFLAT = n_flat

    def accum_downstream_atomic_tmpl(ctx, rec: T, q: T):
        for i in q:
            q[i] = ctx.SOURCE.get(i)
        for i in rec:
            if rec[i] == i:
                continue
            wi = ctx.SOURCE.get(i)
            j = rec[i]
            guard = 0
            while j != rec[j] and guard < NFLAT:
                ctx.bk.atomic_add(q[j], wi)
                j = rec[j]
                guard += 1
            ctx.bk.atomic_add(q[j], wi)

    return KernelBuilder(accum_downstream_atomic_tmpl).freeze()


def build_rake_compress(*, backend: str, backend_mod, n_neighbours: int, logn: int):
    """
    SequenceBuilder for the rake-and-compress accumulation, plus the
    KernelBuilders it is made of.

    Steps: zero_init (ndonors, ndonors_alt, src) -> reset_iteration (ITER=0)
    -> q_init (q[i]=SOURCE.get(i)) -> receivers_to_donors (atomic donor-list
    build) -> loop(["rake_step"], max_times=logn+1) -> decrement_iteration
    (undoes the loop's last bump, so fuse_accum_buffers reads the same
    iteration value the last rake round used - see make_accumulation's
    docstring on the off-by-one; rake_compress_accum's own second top-level
    `for` loop bumps ITER by 1 after every rake pass, as a separate
    offloaded task ordered after it) -> fuse_accum_buffers.

    Composed names: "zero_init", "reset_iteration", "q_init",
    "receivers_to_donors", "rake_step" (the rake_compress_accum kernel,
    referenced by the loop), "decrement_iteration", "fuse_accum_buffers".
    PARAM addresses needing a bound Parameter: "q_init.SOURCE",
    "reset_iteration.ITER", "rake_step.ITER", "rake_step.get_src.ITER",
    "rake_step.update_src.ITER", "decrement_iteration.ITER",
    "fuse_accum_buffers.get_src.ITER" (the same Parameter at all six ITER
    addresses. DATA addresses: this sequence's own {step}.{arg}
    for every kernel's own DATA name (see each template below).

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    n_neighbours : int
    logn : int

    Returns
    -------
    tuple[SequenceBuilder, dict]
        (sequence_builder, kernel_builders_dict) - the dict exposes every
        constituent FrozenKernel individually (keyed by its own name,
        "rake_step" aliased as "rake_compress_accum" for parity with the
        pre-port naming), for direct standalone use if ever wanted.

    """
    T = _tensor_annotation(backend_mod, backend)
    NN = n_neighbours

    get_src, update_src = build_ping_pong_helpers()

    def zero_init_tmpl(ctx, ndonors: T, ndonors_alt: T, src: T):
        for i in ndonors:
            ndonors[i] = 0
            ndonors_alt[i] = 0
            src[i] = 0

    def reset_iteration_tmpl(ctx):
        ctx.ITER.set_node(0, 0)

    def decrement_iteration_tmpl(ctx):
        ctx.ITER.set_node(0, ctx.ITER.get(0) - 1)

    def q_init_tmpl(ctx, q: T):
        for i in q:
            q[i] = ctx.SOURCE.get(i)

    def receivers_to_donors_tmpl(ctx, rec: T, donors: T, ndonors: T):
        for tid in rec:
            rcv = rec[tid]
            if rcv != tid:
                old_val = ctx.bk.atomic_add(ndonors[rcv], 1)
                donors[rcv * NN + old_val] = tid

    def rake_compress_accum_tmpl(ctx, donors: T, ndonors: T, q: T, src: T, donors_alt: T, ndonors_alt: T, q_alt: T):
        for tid in q:
            flip = ctx.get_src(src, tid)

            worked = False
            todo = ndonors[tid] if not flip else ndonors_alt[tid]
            base = tid * NN
            donors_local = ctx.bk.Vector([-1] * NN)
            q_added = 0.0

            i = 0
            while i < todo and i < NN:
                if donors_local[i] == -1:
                    donors_local[i] = donors[base + i] if not flip else donors_alt[base + i]
                did = donors_local[i]

                flip_donor = ctx.get_src(src, did)
                ndnr_val = ndonors[did] if not flip_donor else ndonors_alt[did]

                if ndnr_val <= 1:
                    if not worked:
                        q_added = q[tid] if not flip else q_alt[tid]
                    worked = True

                    q_val = q[did] if not flip_donor else q_alt[did]
                    q_added += q_val

                    if ndnr_val == 0:
                        todo -= 1
                        if todo > i:
                            donors_local[i] = donors[base + todo] if not flip else donors_alt[base + todo]
                        i -= 1
                    else:
                        donor_base = did * NN
                        donors_local[i] = donors[donor_base] if not flip_donor else donors_alt[donor_base]
                i += 1

            if worked:
                if flip:
                    ndonors[tid] = todo
                    q[tid] = q_added
                    for j in range(NN):
                        if j < todo:
                            donors[base + j] = donors_local[j]
                else:
                    ndonors_alt[tid] = todo
                    q_alt[tid] = q_added
                    for j in range(NN):
                        if j < todo:
                            donors_alt[base + j] = donors_local[j]
                ctx.update_src(src, tid, flip)
        for _ in range(1):
            ctx.ITER.set_node(0, ctx.ITER.get(0) + 1)

    def fuse_accum_buffers_tmpl(ctx, q: T, src: T, q_alt: T):
        for tid in q:
            if ctx.get_src(src, tid):
                q[tid] = q_alt[tid]

    zero_init = KernelBuilder(zero_init_tmpl).freeze()
    reset_iteration = KernelBuilder(reset_iteration_tmpl).freeze()
    decrement_iteration = KernelBuilder(decrement_iteration_tmpl).freeze()
    q_init = KernelBuilder(q_init_tmpl).freeze()
    receivers_to_donors = KernelBuilder(receivers_to_donors_tmpl).freeze()
    rake_compress_accum = (
        KernelBuilder(rake_compress_accum_tmpl)
        .compose("get_src", get_src)
        .compose("update_src", update_src)
        .freeze()
    )
    fuse_accum_buffers = (
        KernelBuilder(fuse_accum_buffers_tmpl)
        .compose("get_src", get_src)
        .freeze()
    )

    kernels = {
        "zero_init": zero_init,
        "reset_iteration": reset_iteration,
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
    sb.add("decrement_iteration", decrement_iteration)
    sb.add("fuse_accum_buffers", fuse_accum_buffers)

    sb.step("zero_init")
    sb.step("reset_iteration")
    sb.step("q_init")
    sb.step("receivers_to_donors")
    sb.loop(body=["rake_step"], max_times=logn + 1)
    sb.step("decrement_iteration")
    sb.step("fuse_accum_buffers")

    return sb, kernels


def build_pointer_jump_push(*, backend: str, backend_mod, rounds: int):
    """
    SequenceBuilder for pointer-jump-push accumulation and its constituent
    kernels. Two independently bound steps provide the ping-pong state.

    Steps: q_init (q[i]=SOURCE.get(i)) -> copy_rec_to_work (rec -> work, so
    round 0 is not a special case and rec itself is never written) ->
    loop(["step_a", "step_b"], max_times=rounds // 2) (`rounds`, already
    rounded to even by the caller - see make_accumulation). An even round
    count makes the net effect of alternating step_a/step_b land back in the
    same buffer roles it started from, so the result always ends up in
    whichever buffers "step_a"'s own rec_curr/q_curr address was bound to,
    with no host-side conditional copy-back.

    Composed names: ``q_init``, ``copy_rec_to_work``, ``step_a``, and
    ``step_b``. The same ``accum_pointer_jump_push_step`` FrozenKernel is
    composed twice with different DATA bindings.
    PARAM addresses needing a bound Parameter: "q_init.SOURCE". DATA
    addresses: "q_init.q", "copy_rec_to_work.rec"/"copy_rec_to_work.work",
    "step_a.rec_curr"/"step_a.rec_next"/"step_a.q_curr"/"step_a.q_next" bound
    to (work, work2, q, q_work), "step_b"'s own four bound to the mirror
    (work2, work, q_work, q).

    Retirement rule: when a node's parent
    is a sink in the current jumped graph (grandparent == parent), the node
    pushes once more and then points at itself, so it never re-pushes a
    growing sum into the sink.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    rounds : int
        Already rounded to even by the caller.

    Returns
    -------
    tuple[SequenceBuilder, dict]

    """
    T = _tensor_annotation(backend_mod, backend)

    def q_init_tmpl(ctx, q: T):
        for i in q:
            q[i] = ctx.SOURCE.get(i)

    def copy_rec_to_work_tmpl(ctx, rec: T, work: T):
        for i in rec:
            work[i] = rec[i]

    def accum_pointer_jump_push_step_tmpl(ctx, rec_curr: T, rec_next: T, q_curr: T, q_next: T):
        for i in q_next:
            q_next[i] = q_curr[i]
        for i in rec_curr:
            parent = rec_curr[i]
            rec_next[i] = parent
            if parent != i:
                wi = q_curr[i]
                if wi != 0.0:
                    ctx.bk.atomic_add(q_next[parent], wi)
                grandparent = rec_curr[parent]
                rec_next[i] = i if grandparent == parent else grandparent

    q_init = KernelBuilder(q_init_tmpl).freeze()
    copy_rec_to_work = KernelBuilder(copy_rec_to_work_tmpl).freeze()
    step = KernelBuilder(accum_pointer_jump_push_step_tmpl).freeze()

    kernels = {"q_init": q_init, "copy_rec_to_work": copy_rec_to_work, "accum_pointer_jump_push_step": step}

    sb = SequenceBuilder()
    sb.add("q_init", q_init)
    sb.add("copy_rec_to_work", copy_rec_to_work)
    sb.add("step_a", step)
    sb.add("step_b", step)

    sb.step("q_init")
    sb.step("copy_rec_to_work")
    sb.loop(body=["step_a", "step_b"], max_times=rounds // 2)

    return sb, kernels
