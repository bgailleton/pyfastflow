"""
make_receivers: the SFD (single-flow-direction) receiver factory, built on
the new builder/frozen/bound stack (../core/context/builder.py, frozen.py,
bound.py) and on a grid FrozenGroup from ../grid's make_grid_group.

Like grid/noise/ops there is no stateful context class - make_receivers
returns a dict of unbuilt structures (FrozenKernel/FrozenHelper): a
`receivers` FrozenKernel plus the distance/slope helpers it is made of, so a
caller can recombine them into its own kernel or routine rather than being
stuck with only the compiled receivers kernel. A caller `.build()`s the
member it wants, binds its PARAM/DATA addresses, `.compile()`s:

    be = Backend.from_name("taichi")
    grid = make_grid_group(be, topology="D8")
    recv = make_receivers(be, grid, topology="D8", mode="steepest")
    bound = recv["receivers"].build()
    bound.bind_leaf(grid_params)  # NX/NY/DX/N_NEIGHBOURS - see below
    bound.bind("z", z_field)
    bound.bind("rec", rec_field)
    receivers_kernel = bound.compile(be)
    receivers_kernel()

`mode` ("steepest"|"stochastic") and `h_aware` (False: kernel takes (z, rec)
and slopes read h as 0; True: kernel takes (z, h, rec) and slopes use
(zi-zj)+(hi-hj)) each pick one of four kernel body variants at build time -
see _closure_receivers.py/_cupy_receivers.py's build_receivers. `topology`
("D4"|"D8") must match whatever `grid` was itself built with (see
../grid/__init__.py's own module docstring on the two-call structure/data
split) - it is not readable off `grid` itself, a bare FrozenGroup carrying no
Parameter values yet, and only matters here for
`diagonal_partition_correction` (below); the neighbour loop itself is
already parametrised by `grid.N_NEIGHBOURS`, read as ordinary device data.

mode="stochastic" additionally needs a Parameter bound to the built
receivers kernel's `rand_unit.SEED` address (`rand_unit`'s own wired PARAM
slot, any mode - see rand_unit in the block modules; not build-phase-shared
with anything, so it stays at that nested address rather than being
promoted to the kernel's own top level the way `grid`'s own PARAM names
are) after `.build()`, exactly like any other PARAM slot - there is no Need
indirection in this stack - the host bumps the underlying Parameter between calls for a fresh draw.

`diagonal_partition_correction` only changes anything when `topology ==
"D8"`: it adds the sqrt(2) correction inside dist_from_k_corrected/
dist_between_nodes_corrected (see _closure_receivers.py's
build_distance_slope_helpers for exactly which k values count as diagonal and
why). Off, or on a D4 grid, dist_from_k_corrected/dist_between_nodes_corrected
call straight through to `grid`'s own dist_from_k/dist_between_nodes, no
correction applied - either way these two helpers always independently
compose their own occurrence of `grid` (see _closure_receivers.py's module
docstring for why: a uniform, always-two-occurrences shape lets
`build_receivers` collapse them with `share()` unconditionally, never a
variable number of occurrences depending on the flag).

Returned dict: `receivers` (FrozenKernel, data args (z, rec) or (z, h, rec)),
`dist_from_k_corrected`, `dist_between_nodes_corrected`, `slope_from_values_k`,
`slope_between_nodes` (FrozenHelper), plus `rand_unit` (FrozenHelper) only
when mode="stochastic". `receivers`'s own top-level PARAM slots are every
name `grid` itself wires (NX/NY/DX/N_NEIGHBOURS, plus NODATA_MASK/
OUTLET_MASK if `grid` has them) - bind those bare names once on the built
receivers kernel, not once per occurrence (`grid`'s FrozenGroup is composed
twice inside `receivers`'s own tree - once directly, once nested under
`slope.dist_from_k_corrected` - and build-phase-shared, `_share_leaf`, into
one address each).

A node with no downslope neighbour keeps `rec[i] = i` - a self-receiver, the
same convention a can_out (base level) node uses. This is the pit convention
depression handling later depends on.

rand_unit's hash is keyed on (node, k, seed) rather than (node, seed): legacy
draws once per (node, k) candidate inside the neighbour loop
(`ti.random()`), so a node-keyed hash would scale every candidate at a node
by the same factor and weaken the randomisation rather than reproduce it.
Reproducibility itself diverges from legacy (hash-based vs ti.random()'s
counter-based PRNG) - only the selection distribution's shape is preserved.

make_accumulation: the SFD downstream-accumulation factory, on the new
builder/frozen/bound stack (see its own docstring below for the per-method
details). `source` is bound directly, post-`.build()`, at the returned
kernel's own `SOURCE` PARAM slot - any mode (const, scalar or field all work
with no variant code, since every template reads `source.get(i)`) - there is
no Need indirection anywhere in this stack:

    source_p = TaichiParameter("SOURCE", dtype="f32", mode="const", value=1.0, pool=pool)
    accum = make_accumulation(be, grid, method="atomic", n_flat=n_flat)
    bound = accum["accum"].build()
    bound.bind("SOURCE", source_p)
    bound.bind("rec", rec)
    bound.bind("q", q)
    accum_kernel = bound.compile(be)
    accum_kernel()

`method`:
  - "atomic": on taichi/quadrants, one KernelBuilder ("accum", data args
    (rec, q)) - two top-level for-loops (q[i] = source.get(i), then the
    descent), which the closure backends already launch as two barrier-
    separated GPU dispatches. On cupy, two KernelBuilders ("q_init", data
    arg (q); "accum", data args (rec, q)) - a single CUDA __global__ has no
    portable grid-wide barrier the way two consecutive Taichi/Quadrants
    for-loops do, so "q_init" must be launched, and finish, before "accum".
    Either way: every node walks its receiver chain to the root, atomic-
    adding its own weight into each downstream node. Requires an acyclic
    receiver graph (run after depression handling) - a cycle degrades the
    result (the walk gives up once its guard counter reaches n_flat) rather
    than hanging.
  - "rake_compress" (ported to the new builder/frozen/bound/sequence stack,
    ../core/context/builder.py/frozen.py/bound.py/sequence.py): a
    SequenceBuilder (see _closure_accum.py's/_cupy_accum.py's
    build_rake_compress) plus its constituent KernelBuilders, keyed
    "zero_init", "reset_iteration", "decrement_iteration", "q_init",
    "receivers_to_donors", "rake_compress_accum", "fuse_accum_buffers" (plus
    "bump_iteration" on cupy only - see below). Composed sequence steps:
    "zero_init" -> "reset_iteration" -> "q_init" -> "receivers_to_donors" ->
    a loop over "rake_step" (the same rake_compress_accum kernel, `max_times
    = ceil(log2(n_flat)) + 2`) -> "decrement_iteration" (undoes the loop's
    last bump) -> "fuse_accum_buffers". On closure backends, the iteration
    bump is rake_compress_accum's own second top-level `for` loop, folded in
    rather than a separate single-thread kernel (two consecutive top-level
    `for` loops inside one compiled Taichi/Quadrants kernel are already
    separate offloaded tasks launched in order); on cupy, a single CUDA
    `__global__` gives no such guarantee, so the loop body is
    `["rake_step", "bump_iteration"]`, two real launches per round - see
    _cupy_accum.py's build_rake_compress. No `source`/`iteration_p` argument
    to this factory at all - `SOURCE`/`ITER` are bare wired PARAM slots (any
    mode), bound by the caller on the built sequence after `.build()`,
    exactly like make_receivers' `rand_unit.SEED`; see build_rake_compress's
    own docstring (per backend) for the exact addresses (four independent
    `ITER` addresses after rake_compress_accum's own `share()` collapses its
    two composed ping-pong helpers' ITER occurrences into its own, one
    `SOURCE` address).
  - "pointer_jump_push" (ported the same way): a SequenceBuilder (see
    build_pointer_jump_push) plus its constituent KernelBuilders, keyed
    "q_init", "copy_rec_to_work", and (closure backends)
    "accum_pointer_jump_push_step" or (cupy, split into two launches for a
    real barrier between the copy and the push - see _cupy_accum.py)
    "accum_pointer_jump_push_step_copy"/"accum_pointer_jump_push_step_core".
    The ping-pong between rounds is two independently-bound occurrences of
    the same step kernel ("step_a"/"step_b" - closure; "step_a_copy"/
    "step_a_core"/"step_b_copy"/"step_b_core" - cupy), alternated by a
    sequence loop of `rounds // 2` iterations (`rounds`, computed here,
    already rounded up to even) - no runtime swap() needed, unlike
    routine.py's old add_swap. Same no-`source`-argument contract as
    rake_compress; see build_pointer_jump_push's own docstring for its
    addresses.

Both factories return {"sequence": SequenceBuilder, **kernel_builders} - a
dict, not a compiled object (these factories export builders, not compiled
kernels - see CLAUDE.md). The caller `.freeze()`s (or lets `.compile()`
freeze implicitly - SequenceBuilder has no separate freeze() call exposed
here beyond what `.build()`/`.compile()` already do internally), `.build()`s,
binds every PARAM/DATA address named above and in each build_* docstring,
then `.compile(be)`. Every cupy kernel declares its own `domain`/`block`;
closure backends range over the template's own loop. The
compiled CompiledSequence takes no arguments; call it, then read
`last_trip_counts` if wanted (always exactly one loop entry per compiled
sequence here, so `last_trip_counts[0]` is the only entry, and is always the
full requested count - unlike depression routing's own use of the same loop
machinery, nothing here ever breaks out early via `until`).

Why a SequenceBuilder loop rather than routine.py's unroll-N-times idiom
(../ops/_closure_blocks.py's build_scan_routine, for its own log-depth
passes): a scan pass's kernel body differs every round (`stride` baked in as
a build-time constant), so unrolling costs nothing beyond the kernels
themselves; here the SAME kernel body runs unchanged every round, so
unrolling would only multiply the number of addresses a caller has to bind
(once per round) for no benefit - a loop keeps that count fixed regardless of
round count (`ceil(log2(n_flat))+2` rounds at 1024x1024 is ~22).

`n_flat` is REQUIRED for both, like `method="atomic"`'s own required `n_flat`
- `grid` is a bare `make_grid_group` FrozenGroup with no bound Parameter
values to read it off at build time (see ../grid/__init__.py's own module
docstring), so there is no way to read it from `grid` itself. `rake_compress`
also needs `n_neighbours` explicitly, for the same reason (sizing the fixed
per-node donor arrays) - `pointer_jump_push` needs neither `grid` nor
`n_neighbours` at all.

make_depressions: the depression-handling factory, on the builder/frozen/
bound stack. Two orthogonal build flags:

    ndep_p = TaichiParameter("NDEP", dtype="i32", mode="scalar", value=0, pool=pool)
    deps = make_depressions(be, grid, ndep_p, method="vanilla", reroute="carve", n_flat=n_flat)

`method` ("vanilla"|"optimized") picks how basins are labelled and, for
reroute="carve", how the carve itself runs; `reroute` ("carve"|"jump") picks
how a resolved basin's pit is reconnected to its outlet. All four
combinations build. `depression_counter_p` is a bare caller-allocated scalar
i32 Parameter, bound directly at the returned kernel's own `NDEP` PARAM
slot, post-`.build()` - there is no Need indirection anywhere in this stack;
the underlying Parameter is
not built here, since this factory takes no pool: every scratch buffer below
is a caller-supplied data arg, never a bound field Parameter.

Every buffer is n_flat-sized, since a per-basin array is indexed by basin id
and basin id = pit index + 1 (bid/basin_saddlenode/outlet range over the
same 0..n_flat-1 index space as every per-node buffer). Required data args,
by dict key:

  "ndep":                the `ndep_p` scalar Parameter itself (bag.ndep.read())
  "depression_counter":   closure: (rec,) - accumulates into ndep_p, bound
                          directly as a raw field. cupy: (rec, ndep) - ndep_p
                          is only ever reached through $...$ get() spans
                          there, which registers it read-only in the
                          constant block (see compile_cupy.py's
                          _register_ptr), so the caller instead
                          passes `ndep_p.handle().array` positionally, same as
                          `rec` - see build_depression_counter in
                          _cupy_depressions.py. Either way the caller must
                          ndep_p.set(0) before each launch (mirrors
                          ops.Reduce.run_sum).
  "copy_field":           (src, dst) i32        -> dst[:] = src[:]
  "label_basins" (vanilla): Routine, data names ("rec", "bid", "rec_jump")
  "label_basins" (optimized, closure): Kernel, data args (rec, rec_jump, bid)
  "label_basins" (optimized, cupy): Routine, data names ("rec", "rec_jump", "bid")
    - see _closure_depressions.py/_cupy_depressions.py's build_basin_labelling_* for
      why cupy needs three real launches where a closure backend needs one.
  "saddlesort":           Routine (6 kernels, unchanged by `method`), data
                          names ("bid", "z", "z_prime", "is_border",
                          "basin_saddle", "basin_saddlenode", "outlet") -
                          the six constituent KernelBuilders are also
                          exposed under "saddlesort_<name>".
  "reroute" (carve, vanilla): Routine, data names ("rec", "rec_work",
                          "rec_jump", "tag", "tag_alt", "bid",
                          "basin_saddlenode", "outlet", "rerouted")
  "reroute" (carve, optimized): Kernel, data args (rec, basin_saddlenode, outlet)
  "reroute" (jump, closure): Kernel, data args (rec, outlet, rerouted)
  "reroute" (jump, cupy): Routine, data names ("rec", "outlet", "rerouted")
    - split for the same real-launch-barrier reason as label_basins above.

Every "label_basins"/"reroute" constituent KernelBuilder is also exposed
under "label_basins_<name>"/"reroute_<name>" respectively, mirroring
make_accumulation's own routine+constituent-kernels convention.

`reroute_jump`'s pit write is deliberately `rec[i - 1]`, not `rec[i]`: the
loop is over basin ids and basin id = pit index + 1 - ported exactly as
legacy has it, not "fixed" - see _closure_depressions.py's build_reroute_jump.

`ops.bitpack` (pack/unpack_value/unpack_index) replaces legacy's
f32_i32_struct module for the lexicographic (elevation, target-basin) and
(elevation, node) argmins saddlesort's atomic_min passes need; on cupy,
i64 atomic_min is a CAS loop (CUDA has no native atomicMin over signed long
long) - see _cupy_depressions.py's build_atomic_min_ll.

make_depressions builds the routines/kernels only. make_depression_solver
builds the outer host-driven loop as a FrozenSequence. Callers bind live
DataHandles to the returned sequence and compile it for their Backend.

    depression_counter -> ndep;  ndep == 0 -> nothing to do
    loop max_times = ceil(log2(max(2, ndep))) + 2:
        <label basins>  <saddlesort>  <reroute>
        depression_counter -> ndep;  until ndep == 0

`max_times` returns 0 when the entry count is already 0, which is what
Sequence.add_loop's documented "max_times <= 0 runs the body zero times"
case is for, and matches legacy runtime.py's `if ndep == 0: return`.

Buffer naming: `rec` is the caller's authoritative receiver buffer, read at
entry and holding the resolved graph on return, in every combination. The
vanilla carve routine's own two receiver buffers are its internal working
copy (bound to `rec_scratch`) and the authoritative one (bound to `rec`) -
see build_reroute_carve_vanilla, whose last step copies the working buffer
back into the authoritative one.

make_fill_reconstruct/make_fill_reconstruct_solver: a standalone alternative
to make_depressions/make_depression_solver, not a `method` of it - grayscale
morphological reconstruction against elevation, converging `filled`/`parent`
(the receiver graph) directly with no basin ids, saddle search or outlet
routing at all. See that pair's own docstrings and the block modules'
section notes above build_fill_reconstruct_* for the algorithm, the
combined-buffer frontier ping-pong a Sequence's fixed-at-compile-time data
handles require, and the two closure-backend substitutions (no 3-arg
`range()`, `atomic_max` standing in for `atomicExch`).

Author: B.G (07/2026)
"""

import math
from importlib import import_module

from ..core import Backend, HostBlockBuilder, SequenceBuilder, require_backend
from ..noise import make_hash_u32

_MODES = frozenset({"steepest", "stochastic"})
_ACCUM_METHODS = frozenset({"atomic", "rake_compress", "pointer_jump_push"})


def _blocks_for(be: Backend, section: str):
    """
    The private block module implementing one flow section's device code
    for one backend name - e.g. ("cupy", "depressions") -> _cupy_depressions.

    `section` is one of "receivers", "accum", "depressions", "reconstruct" -
    each has its own {_cupy,_closure}_<section>.py module, kept separate
    since none of them share device code with each other.

    Author: B.G (07/2026)
    """
    if be.family == "closure":
        prefix = "_closure"
    elif be.family == "cupy":
        prefix = "_cupy"
    else:
        raise ValueError(f"unsupported backend family {be.family!r}")
    return import_module(f".{prefix}_{section}", __package__)


def make_receivers(
    be: Backend,
    grid,
    *,
    topology: str = "D8",
    mode: str = "steepest",
    diagonal_partition_correction: bool = False,
    h_aware: bool = False,
) -> dict:
    """
    Build one receivers dict: the `receivers` FrozenKernel (data args
    `(z, rec)` or `(z, h, rec)` depending on `h_aware`) plus the distance/
    slope FrozenHelpers it is made of, and `rand_unit` when
    mode="stochastic". See the module docstring for the full contract
    (`topology`, addressing, `SEED`).

    `mode` "steepest"|"stochastic" picks the kernel body variant.
    `diagonal_partition_correction`, `h_aware` and `topology` are documented
    in the module docstring.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    grid : FrozenGroup
        Grid structure this composes.
    topology : str, optional
        "D4" or "D8" (default).
    mode : str, optional
        "steepest" (default) or "stochastic".
    diagonal_partition_correction : bool, optional
        See the module docstring.
    h_aware : bool, optional
        See the module docstring.

    Returns
    -------
    dict
        {"receivers": FrozenKernel, ...distance/slope FrozenHelpers, and
        "rand_unit" when mode="stochastic"}.

    Raises
    ------
    ValueError
        If `mode` or `topology` is not a recognised value.

    Author: B.G (08/2026)
    """
    if mode not in _MODES:
        raise ValueError(f"make_receivers: mode must be one of {sorted(_MODES)}, got {mode!r}")
    if topology not in ("D4", "D8"):
        raise ValueError(f"make_receivers: topology must be 'D4' or 'D8', got {topology!r}")

    be = require_backend(be)
    blocks = _blocks_for(be, "receivers")
    hash_u32 = make_hash_u32(be) if mode == "stochastic" else None

    if be.family == "closure":
        backend_mod = be.module
        return blocks.build_receivers(
            backend=be.name,
            backend_mod=backend_mod,
            grid=grid,
            hash_u32=hash_u32,
            mode=mode,
            topology=topology,
            diagonal_partition_correction=diagonal_partition_correction,
            h_aware=h_aware,
        )
    return blocks.build_receivers(
        grid=grid,
        hash_u32=hash_u32,
        mode=mode,
        topology=topology,
        diagonal_partition_correction=diagonal_partition_correction,
        h_aware=h_aware,
    )


def make_accumulation(
    be: Backend,
    grid,
    source=None,
    *,
    method: str = "rake_compress",
    n_flat: int | None = None,
    n_neighbours: int | None = None,
    iteration_p=None,
    fr_stage: int = 2048,
    blocks_per_sm: int = 2,
    threads: int = 256,
) -> dict:
    """
    Build one accumulation structure for `method`
    "atomic"|"rake_compress"|"pointer_jump_push"|"persistent_mfd".

    method="atomic" (../core/context/builder.py/frozen.py/bound.py):
    `grid` is unused (atomic reads no grid at all - it only walks `rec`) and
    `source` is ignored; `n_flat` is REQUIRED - a `make_grid_group` result
    carries no Parameter values to read `nx`/`ny` from at build time (see
    ../grid/__init__.py's own module docstring on the structure/data
    split), so there is no way to compute it from `grid`. Returns
    {"accum": FrozenKernel} on Taichi/Quadrants (data args (rec, q); `SOURCE`
    is this kernel's own wired PARAM slot, bound after `.build()` like any
    other) or {"q_init": FrozenKernel, "accum": FrozenKernel} on cupy (see
    _cupy_accum.py's build_atomic for why cupy needs the second real
    launch): "q_init" (data arg (q,)) must be run, and finish, before
    "accum" (data args (rec, q)); both wire `SOURCE`.

    method="rake_compress"/"pointer_jump_push" (ported to the new builder/
    frozen/bound/sequence stack, ../core/context/builder.py/frozen.py/
    bound.py/sequence.py): `source`/`iteration_p` are accepted here for
    call-site parity with method="persistent_mfd" but are unused and
    ignored for these two - there is no Need indirection anywhere in this
    stack; the returned SequenceBuilder wires bare `SOURCE`/`ITER` PARAM
    slots the caller binds directly, post-`.build()` - see the module
    docstring and _closure_accum.py's/_cupy_accum.py's own docstrings for
    the exact addresses. `n_flat` is REQUIRED for both, and `rake_compress`
    additionally requires `n_neighbours` - `grid` is a bare `make_grid_group`
    FrozenGroup with no bound Parameter values to read either off at build
    time, the same reasoning method="atomic"'s own required `n_flat` already
    establishes; `grid` itself is unused by both (neither ever composes it -
    `rake_compress` only needs `n_neighbours` as a build-time int, never a
    device HELPER call, so there is nothing here for a FrozenGroup to supply
    that a plain int does not already cover).

    method="persistent_mfd" (ported to the new builder/frozen/bound stack,
    ../core/context/builder.py/frozen.py/bound.py, like atomic): `source`/
    `iteration_p` are accepted here for call-site parity with the other three
    methods but unused and ignored - there is no Need indirection in this
    stack; the returned "q_init" FrozenKernel wires a bare `SOURCE` PARAM
    slot (any mode) the caller binds directly, post-`.build()`, exactly like
    `method="atomic"`'s own "q_init".

    `method="persistent_mfd"` is cupy-only (raises for any other backend -
    see _cupy_mfd_accum.py's module docstring for why there is, and will
    never be, a closure-backend equivalent): a persistent-kernel,
    level-synchronous MFD accumulation over a caller-supplied receiver mask
    (`dirs`, u8) + dense per-direction weights (`mfd_w`, f32, this grid's
    n_neighbours values per cell) + `indegree` - this factory does not build
    MFD topology, only accumulates over one already built. Requires `n_flat`
    and `n_neighbours` explicitly (`grid` is a bare `make_grid_group`
    FrozenGroup with no bound Parameter values to read either off, same
    reasoning as `method="atomic"`'s own required `n_flat`). Returns a dict
    with "q_init" (data arg (accum,), FrozenKernel) and "accum" (data args
    (frontier0, frontier1, count, barrier, dirs, mfd_w, accum, indegree),
    FrozenKernel, composing its own `grid` occurrence for
    `ctx.grid.neighbour_raw` - launched with
    `persistent_grid_block(blocks_per_sm=blocks_per_sm, threads=threads)`'s
    fixed (grid, block), not n_flat-sized). `fr_stage`/`blocks_per_sm`/
    `threads` are only used by this method.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    grid : FrozenGroup
        Unused except by method="persistent_mfd" (composes `grid.neighbour_raw`).
    source : optional
        Unused - accepted only for call-site parity across methods.
    method : str, optional
        "atomic", "rake_compress" (default), "pointer_jump_push" or
        "persistent_mfd".
    n_flat : int, optional
        Required for every method.
    n_neighbours : int, optional
        Required for method="rake_compress" and "persistent_mfd".
    iteration_p : optional
        Unused - accepted only for call-site parity across methods.
    fr_stage, blocks_per_sm, threads : optional
        Used only by method="persistent_mfd".

    Returns
    -------
    dict
        Structure depends on `method` - see above.

    Raises
    ------
    ValueError
        If `method` is unrecognised, `n_flat`/`n_neighbours` is required and
        missing, or `method="persistent_mfd"` is requested on a non-cupy
        backend.

    Author: B.G (08/2026)
    """
    be = require_backend(be)
    if method == "atomic":
        if n_flat is None:
            raise ValueError(
                "make_accumulation: method='atomic' requires n_flat explicitly - "
                "grid is a bare FrozenGroup with no bound values to read it off"
            )
        blocks = _blocks_for(be, "accum")
        if be.family == "closure":
            return {"accum": blocks.build_atomic(backend=be.name, backend_mod=be.module, n_flat=int(n_flat))}
        return blocks.build_atomic(n_flat=int(n_flat))

    if method == "persistent_mfd":
        if be.family != "cupy":
            raise ValueError(
                f"make_accumulation: method='persistent_mfd' is cupy-only (got backend={be.name!r}) - "
                "see _cupy_mfd_accum.py's module docstring for why there is no closure-backend equivalent"
            )
        if n_flat is None:
            raise ValueError(
                "make_accumulation: method='persistent_mfd' requires n_flat explicitly - "
                "grid is a bare FrozenGroup with no bound values to read it off"
            )
        if n_neighbours is None:
            raise ValueError(
                "make_accumulation: method='persistent_mfd' requires n_neighbours explicitly - "
                "grid is a bare FrozenGroup with no bound values to read it off"
            )
        from . import _cupy_mfd_accum

        return _cupy_mfd_accum.build_persistent_mfd(
            grid=grid, n_flat=int(n_flat), n_neighbours=int(n_neighbours), fr_stage=fr_stage,
            blocks_per_sm=blocks_per_sm, threads=threads,
        )

    if method not in _ACCUM_METHODS:
        raise ValueError(f"make_accumulation: method must be one of {sorted(_ACCUM_METHODS)}, got {method!r}")

    # rake_compress/pointer_jump_push (ported to the new builder/frozen/
    # bound/sequence stack - ../core/context/builder.py, frozen.py, bound.py,
    # sequence.py): `source`/`iteration_p` are no longer accepted here at
    # all - there is no Need indirection in this stack (see _closure_accum.py/
    # _cupy_accum.py's module docstrings). Both factories return a
    # SequenceBuilder wiring bare `SOURCE`/`ITER` PARAM slots the caller binds
    # itself, at the addresses each module's build_rake_compress/
    # build_pointer_jump_push docstring enumerates, after `.build()` - exactly
    # like make_receivers' `rand_unit.SEED`.
    blocks = _blocks_for(be, "accum")
    if n_flat is None:
        raise ValueError(
            f"make_accumulation: method={method!r} requires n_flat explicitly - "
            "grid is a bare FrozenGroup with no bound values to read it off"
        )
    n_flat_resolved = int(n_flat)
    closure = be.family == "closure"

    logn = math.ceil(math.log2(n_flat_resolved)) + 1

    if method == "rake_compress":
        if n_neighbours is None:
            raise ValueError(
                "make_accumulation: method='rake_compress' requires n_neighbours explicitly - "
                "grid is a bare FrozenGroup with no bound values to read it off"
            )
        if closure:
            sb, kernels = blocks.build_rake_compress(
                backend=be.name, backend_mod=be.module, n_neighbours=int(n_neighbours), logn=logn,
            )
        else:
            sb, kernels = blocks.build_rake_compress(n_neighbours=int(n_neighbours), logn=logn, n_flat=n_flat_resolved)
    else:  # pointer_jump_push
        rounds = logn + 1
        if rounds % 2 != 0:
            rounds += 1
        if closure:
            sb, kernels = blocks.build_pointer_jump_push(backend=be.name, backend_mod=be.module, rounds=rounds)
        else:
            sb, kernels = blocks.build_pointer_jump_push(rounds=rounds, n_flat=n_flat_resolved)

    out = dict(kernels)
    out["sequence"] = sb
    return out


# ---------------------------------------------------------------------------
# depressions
# ---------------------------------------------------------------------------

_DEP_METHODS = frozenset({"vanilla", "optimized"})
_DEP_REROUTES = frozenset({"carve", "jump"})


def make_depressions(
    be: Backend,
    grid,
    depression_counter_p,
    *,
    method: str = "vanilla",
    reroute: str = "carve",
    n_flat: int,
) -> dict:
    """
    Build one depression-handling dict of unbuilt FrozenKernel/FrozenRoutine
    structures for `method` "vanilla"|"optimized" x `reroute` "carve"|"jump" -
    on the new builder/frozen/bound/routine stack (../core/context/
    builder.py, frozen.py, bound.py, routine.py). Keys: "ndep_p" (the
    caller's own Parameter, passed straight through), "copy_field",
    "depression_counter" (FrozenKernel, data args (rec, ndep) - `ndep` bound
    to `depression_counter_p.handle().array`, reset with `.set(0)` before each
    launch), "label_basins" (FrozenKernel for method="optimized" on closure
    backends, FrozenRoutine otherwise - see _closure_depressions.py's/
    _cupy_depressions.py's own build_basin_labelling_* docstrings for the
    exact step names/addresses each combination mints), "saddlesort"
    (FrozenRoutine, unchanged by `method`), "reroute" (FrozenKernel for
    reroute="jump" on closure backends or reroute="carve"+method="optimized"
    on either backend, FrozenRoutine otherwise). Every "label_basins_<name>"/
    "saddlesort_<name>"/"reroute_<name>" constituent FrozenKernel is also
    exposed, mirroring make_accumulation's own routine+constituent-kernels
    convention.

    `depression_counter_p` is a plain Parameter (i32, mode "scalar") the
    caller builds and owns - not built here (this factory takes no pool), no
    Need wrapper (there is no Need indirection in this stack; see the module
    docstring). `grid` is the caller's `make_grid_group` FrozenGroup (device
    structure only) - every site that reaches `can_out`/`neighbour`/
    `N_NEIGHBOURS` composes its OWN independent occurrence of it (build-phase
    `share()` only collapses occurrences within one KernelBuilder's own
    composed subtree, never across sibling routine/sequence steps - see
    _closure_depressions.py's module docstring), so a caller binds the same
    grid Parameter object at every one of those addresses after `.build()`-
    ing whatever contains them - `make_depression_solver` does this
    automatically by leaf name (`_bind_grid_everywhere`), below.

    `n_flat` is required, explicit - this factory takes no pool and reads no
    Parameter for it, and a bare `make_grid_group` FrozenGroup carries no
    bound values to read it off (same reasoning as make_accumulation's
    method="atomic").

    This builds the routines/kernels only; running them in the
    label -> saddlesort -> reroute -> recount loop the algorithm needs is
    make_depression_solver's job, not this one's.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    grid : FrozenGroup
        Grid structure, composed independently at each site that needs it.
    depression_counter_p : Parameter
        Caller-owned i32 scalar Parameter, bound to "depression_counter"'s
        `ndep` data arg.
    method : str, optional
        "vanilla" (default) or "optimized".
    reroute : str, optional
        "carve" (default) or "jump".
    n_flat : int
        Required explicitly - see above.

    Returns
    -------
    dict
        Keys documented above.

    Raises
    ------
    ValueError
        If `method` or `reroute` is not a recognised value.

    Author: B.G (08/2026)
    """
    if method not in _DEP_METHODS:
        raise ValueError(f"make_depressions: method must be one of {sorted(_DEP_METHODS)}, got {method!r}")
    if reroute not in _DEP_REROUTES:
        raise ValueError(f"make_depressions: reroute must be one of {sorted(_DEP_REROUTES)}, got {reroute!r}")

    be = require_backend(be)
    backend_mod = be.module
    blocks = _blocks_for(be, "depressions")
    n_flat_resolved = int(n_flat)
    closure = be.family == "closure"
    logn = math.ceil(math.log2(n_flat_resolved)) + 1

    from ..ops import make_bitpack_group

    bitpack = make_bitpack_group(be)

    out: dict = {"ndep_p": depression_counter_p}

    if closure:
        copy_field = blocks.build_copy_field(backend=be.name, backend_mod=backend_mod)
        depression_counter = blocks.build_depression_counter(backend=be.name, backend_mod=backend_mod, grid=grid)
    else:
        copy_field = blocks.build_copy_field(n_flat=n_flat_resolved)
        depression_counter = blocks.build_depression_counter(grid=grid, n_flat=n_flat_resolved)
    out["copy_field"] = copy_field
    out["depression_counter"] = depression_counter

    # basin labelling - route-based for both methods: basin identity carried
    # across rounds in basin_route (label from it, not the carved receivers) +
    # a merge kernel folding each resolved basin into its receiver. The full
    # logn+1 pointer-jump contraction every round is load-bearing: it is what
    # keeps the carved graph a forest, which the serial optimized carve's
    # saddle->pit walk requires to terminate. See _closure_depressions.py.
    if closure:
        lb_rb, lb_kernels = blocks.build_basin_labelling_route(
            backend=be.name, backend_mod=backend_mod, grid=grid, logn=logn,
        )
        out["merge_basin_route"] = blocks.build_merge_basin_route(
            backend=be.name, backend_mod=backend_mod, bitpack=bitpack,
        )
    else:
        lb_rb, lb_kernels = blocks.build_basin_labelling_route(
            grid=grid, n_flat=n_flat_resolved, logn=logn,
        )
        out["merge_basin_route"] = blocks.build_merge_basin_route(
            bitpack=bitpack, n_flat=n_flat_resolved,
        )
    out["label_basins"] = lb_rb.freeze()
    for name, kb in lb_kernels.items():
        out[f"label_basins_{name}"] = kb

    # saddlesort - shared, unchanged by `method`
    if closure:
        ss_rb, ss_kernels = blocks.build_saddlesort(backend=be.name, backend_mod=backend_mod, grid=grid, bitpack=bitpack)
    else:
        ss_rb, ss_kernels = blocks.build_saddlesort(grid=grid, bitpack=bitpack, n_flat=n_flat_resolved)
    out["saddlesort"] = ss_rb.freeze()
    for name, kb in ss_kernels.items():
        out[f"saddlesort_{name}"] = kb

    # reroute
    if reroute == "carve":
        if method == "vanilla":
            if closure:
                rr_rb, rr_kernels = blocks.build_reroute_carve_vanilla(
                    backend=be.name, backend_mod=backend_mod, bitpack=bitpack, copy_field=copy_field, logn=logn,
                )
            else:
                rr_rb, rr_kernels = blocks.build_reroute_carve_vanilla(
                    bitpack=bitpack, copy_field=copy_field, n_flat=n_flat_resolved, logn=logn,
                )
            out["reroute"] = rr_rb.freeze()
            for name, kb in rr_kernels.items():
                out[f"reroute_{name}"] = kb
        else:  # optimized
            if closure:
                out["reroute"] = blocks.build_reroute_carve_optimized(backend=be.name, backend_mod=backend_mod, bitpack=bitpack, n_flat=n_flat_resolved)
            else:
                out["reroute"] = blocks.build_reroute_carve_optimized(bitpack=bitpack, n_flat=n_flat_resolved)
    else:  # jump
        if closure:
            out["reroute"] = blocks.build_reroute_jump(backend=be.name, backend_mod=backend_mod, bitpack=bitpack)
        else:
            rr_rb, rr_kernels = blocks.build_reroute_jump(bitpack=bitpack, n_flat=n_flat_resolved)
            out["reroute"] = rr_rb.freeze()
            for name, kb in rr_kernels.items():
                out[f"reroute_{name}"] = kb

    return out


# ---------------------------------------------------------------------------
# depressions: the outer host-driven loop
# ---------------------------------------------------------------------------


def _bind_if_present(bound, addr: tuple, value) -> None:
    """Bind `addr` on `bound` iff it is one of its minted addresses - a no-op otherwise (this method/reroute combination never mints it)."""
    if addr in bound.addresses():
        bound.bind(addr, value)


def _bind_grid_everywhere(bound, grid_params: dict) -> None:
    """
    Bind every occurrence of a grid PARAM leaf (any address ending in
    `(..., "grid", <NAME>)` for `<NAME>` a `grid_params` key) to the matching
    Parameter - see make_depressions' own docstring for why grid is composed
    independently at every site that needs it, never build-phase-collapsed
    across routine/sequence steps, and so needs binding at every one of those
    addresses; this is the generic equivalent of make_accumulation's own
    multi-address `ITER`/`SOURCE` binding, applied by introspection rather
    than a hand-typed address list (the exact set of "grid" occurrences
    varies by `method`/`reroute`/backend - see _closure_depressions.py's/
    _cupy_depressions.py's own build_* docstrings).

    Not expressed via bound.py's own `bind_leaf`: the match
    here is "second-to-last segment is literally 'grid'", independent of
    address depth (`label_basins.grid.NX` and, in principle,
    some_step.some_helper.grid.NX alike) - `bind_leaf`'s `prefix` only
    restricts by leading segments and therefore cannot express "the
    second-to-last segment, whatever the leading depth". A plain unprefixed
    `bind_leaf(grid_params)` would happen to give the same result today
    (nothing outside a grid occurrence is ever named NX/NY/DX/N_NEIGHBOURS/
    NODATA_MASK/OUTLET_MASK), but that is a property of what currently
    exists, not something this function's own match should rely on.

    Author: B.G (08/2026)
    """
    for addr in bound.addresses():
        if len(addr) >= 2 and addr[-2] == "grid" and addr[-1] in grid_params:
            bound.bind(addr, grid_params[addr[-1]])


def _require(label: str, **buffers):
    """Raise naming every buffer this combination needs that was left None."""
    missing = sorted(name for name, buf in buffers.items() if buf is None)
    if missing:
        raise ValueError(f"make_depression_solver: {label} requires {missing}")


def _build_depression_solver_sequence(
    be: Backend,
    deps: dict,
    grid_params: dict,
    *,
    method: str = "vanilla",
    reroute: str = "carve",
    n_flat: int,
    block_size: int = 256,
):
    """
    Construct the inert depression-routing sequence for the public solver
    factory.
    make_depressions, as a compiled Sequence (sequence.py) - see the
    module docstring for its shape:

        zero_ndep(); depression_counter()
        loop max_times = entry_passes(ndep), until = resolved:
            label_basins; saddlesort; reroute; zero_ndep(); depression_counter()

    `method`/`reroute` must be the ones `deps` was built with; they decide
    which DATA addresses its explicit binding helper fills. The frozen result
    carries no live buffers.

    Live DATA and grid Parameters are supplied later to
    :func:`bind_depression_solver`; ``deps`` supplies every frozen child.
    grid_params : dict
        The make_grid_parameters() dict backing the same grid.
    method : str, optional
        Must match what `deps` was built with.
    reroute : str, optional
        Must match what `deps` was built with.
    rec, z, bid, rec_jump, z_prime, is_border, basin_saddle,
    basin_saddlenode, outlet : DataHandle
        Required in every combination - see above.
    rerouted, tag, tag_alt, rec_scratch : DataHandle, optional
        Required for some `method`/`reroute` combinations - see above.
    n_flat : int
        Required - sets cupy's launch dimensions.
    block_size : int, optional
        cupy CUDA block size (default 256); unused on taichi/quadrants.

    Returns
    -------
    CompiledSequence
        Takes no arguments; reports passes taken in `last_trip_counts[0]`.

    Raises
    ------
    ValueError
        If `method`/`reroute` is unrecognised, or a required buffer is
        missing for the given combination.

    Author: B.G (08/2026)
    """
    if method not in _DEP_METHODS:
        raise ValueError(f"make_depression_solver: method must be one of {sorted(_DEP_METHODS)}, got {method!r}")
    if reroute not in _DEP_REROUTES:
        raise ValueError(f"make_depression_solver: reroute must be one of {sorted(_DEP_REROUTES)}, got {reroute!r}")
    ndep_p = deps["ndep_p"]

    def _zero_ndep_tmpl(ctx):
        ctx.NDEP.set(0)

    def _entry_passes_tmpl(ctx):
        ndep = int(ctx.NDEP.read())
        if ndep == 0:
            return 0
        return math.ceil(math.log2(max(2, ndep))) + 2

    def _resolved_tmpl(ctx):
        return int(ctx.NDEP.read()) == 0

    zero_ndep_hb = HostBlockBuilder(_zero_ndep_tmpl).freeze()
    entry_passes_hb = HostBlockBuilder(_entry_passes_tmpl).freeze()
    resolved_hb = HostBlockBuilder(_resolved_tmpl).freeze()

    sb = SequenceBuilder()
    sb.add("zero_ndep", zero_ndep_hb)
    sb.add("depression_counter", deps["depression_counter"])
    sb.add("label_basins", deps["label_basins"])
    sb.add("saddlesort", deps["saddlesort"])
    sb.add("reroute", deps["reroute"])
    sb.add("entry_passes", entry_passes_hb)
    sb.add("resolved", resolved_hb)
    # carry basin identity across rounds (both methods): seed basin_route = rec
    # once, then merge each resolved basin into its receiver every round (OG
    # loop) - the forest guarantee both carve variants rely on.
    sb.add("init_basin_route", deps["copy_field"])
    sb.add("merge", deps["merge_basin_route"])

    sb.step("zero_ndep")
    sb.step("depression_counter")
    sb.step("init_basin_route")
    sb.loop(
        body=["label_basins", "saddlesort", "reroute", "merge", "zero_ndep", "depression_counter"],
        max_times="entry_passes",
        until="resolved",
    )

    return sb.freeze(), {"NDEP": ndep_p}


def bind_depression_solver(
    frozen, grid_params: dict, *, ndep_p, method: str, reroute: str,
    rec, z, bid, rec_jump, z_prime, is_border, basin_saddle,
    basin_saddlenode, outlet, rerouted=None, tag=None, tag_alt=None,
    rec_scratch=None, basin_route=None, b_rcv=None,
):
    """Bind live :class:`DataHandle` objects to a frozen depression sequence.

    This deliberately performs no compilation.  The caller owns the returned
    bound sequence and selects its launch configuration at ``.compile()``.
    """
    _require("every combination", rec=rec, z=z, bid=bid, rec_jump=rec_jump,
             z_prime=z_prime, is_border=is_border, basin_saddle=basin_saddle,
             basin_saddlenode=basin_saddlenode, outlet=outlet, b_rcv=b_rcv,
             basin_route=basin_route)
    if reroute == "jump" or (reroute == "carve" and method == "vanilla"):
        _require("reroute data", rerouted=rerouted)
    if reroute == "carve" and method == "vanilla":
        _require("vanilla carve data", tag=tag, tag_alt=tag_alt, rec_scratch=rec_scratch)
    bound = frozen.build()
    for name in ("zero_ndep", "entry_passes", "resolved"):
        bound.bind((name, "NDEP"), ndep_p)
    bound.bind(("depression_counter", "rec"), rec)
    bound.bind(("depression_counter", "ndep"), ndep_p.handle())
    bound.bind_leaf({"rec_jump": basin_route, "basin_route": basin_route, "bid": bid}, prefix=("label_basins",))
    bound.bind(("init_basin_route", "src"), rec)
    bound.bind(("init_basin_route", "dst"), basin_route)
    bound.bind_leaf({"outlet": outlet, "basin_route": basin_route}, prefix=("merge",))
    bound.bind_leaf({"bid": bid, "z": z, "z_prime": z_prime, "is_border": is_border,
                     "basin_saddle": basin_saddle, "basin_saddlenode": basin_saddlenode,
                     "outlet": outlet, "b_rcv": b_rcv}, prefix=("saddlesort",))
    if reroute == "carve" and method == "vanilla":
        bound.bind_leaf({"tag": tag, "tag_alt": tag_alt, "rec": rec_scratch,
                         "rec_work": rec, "bid": bid, "saddlenode": basin_saddlenode,
                         "outlet": outlet, "rerouted": rerouted, "rec_orig": rec_jump}, prefix=("reroute",))
        _bind_if_present(bound, ("reroute", "copy_recwork_to_rec", "src"), rec)
        _bind_if_present(bound, ("reroute", "copy_recwork_to_rec", "dst"), rec_scratch)
        _bind_if_present(bound, ("reroute", "copy_recwork_to_recjump", "src"), rec)
        _bind_if_present(bound, ("reroute", "copy_recwork_to_recjump", "dst"), rec_jump)
        _bind_if_present(bound, ("reroute", "copy_rec_to_recwork", "src"), rec_scratch)
        _bind_if_present(bound, ("reroute", "copy_rec_to_recwork", "dst"), rec)
    elif reroute == "carve":
        bound.bind_leaf({"rec": rec, "basin_saddlenode": basin_saddlenode, "outlet": outlet}, prefix=("reroute",))
    else:
        bound.bind_leaf({"rec": rec, "outlet": outlet, "rerouted": rerouted}, prefix=("reroute",))
    _bind_grid_everywhere(bound, grid_params)
    return bound


def depression_binding_plan(frozen, *, method: str, reroute: str) -> dict[str, str]:
    """Return Program's declarative binding map for a depression solver.

    This is the structural counterpart of :func:`bind_depression_solver`: it
    records program-value names, never receives live Parameters or handles and
    never compiles.  Program remains the sole bind/compile owner.
    """
    bound = frozen.build()
    plan: dict[str, str] = {}
    try:
        addrs = bound.addresses()

        def put(addr, value):
            if tuple(addr) in addrs:
                plan[".".join(addr)] = value

        def leaf(prefix, values):
            p = tuple(prefix)
            for addr in addrs:
                if addr[:len(p)] == p and addr[-1] in values:
                    plan[".".join(addr)] = values[addr[-1]]

        for name in ("zero_ndep", "entry_passes", "resolved"):
            put((name, "NDEP"), "ndep")
        put(("depression_counter", "rec"), "rec")
        put(("depression_counter", "ndep"), "ndep.handle")
        leaf(("label_basins",), {"rec_jump": "basin_route", "basin_route": "basin_route", "bid": "bid"})
        put(("init_basin_route", "src"), "rec")
        put(("init_basin_route", "dst"), "basin_route")
        leaf(("merge",), {"outlet": "outlet", "basin_route": "basin_route"})
        leaf(("saddlesort",), {"bid": "bid", "z": "z", "z_prime": "z_prime", "is_border": "is_border", "basin_saddle": "basin_saddle", "basin_saddlenode": "basin_saddlenode", "outlet": "outlet", "b_rcv": "b_rcv"})
        if reroute == "carve" and method == "vanilla":
            leaf(("reroute",), {"tag": "tag", "tag_alt": "tag_alt", "rec": "rec_scratch", "rec_work": "rec", "bid": "bid", "saddlenode": "basin_saddlenode", "outlet": "outlet", "rerouted": "rerouted", "rec_orig": "rec_jump"})
            put(("reroute", "copy_recwork_to_rec", "src"), "rec")
            put(("reroute", "copy_recwork_to_rec", "dst"), "rec_scratch")
            put(("reroute", "copy_recwork_to_recjump", "src"), "rec")
            put(("reroute", "copy_recwork_to_recjump", "dst"), "rec_jump")
            put(("reroute", "copy_rec_to_recwork", "src"), "rec_scratch")
            put(("reroute", "copy_rec_to_recwork", "dst"), "rec")
        elif reroute == "carve":
            leaf(("reroute",), {"rec": "rec", "basin_saddlenode": "basin_saddlenode", "outlet": "outlet"})
        else:
            leaf(("reroute",), {"rec": "rec", "outlet": "outlet", "rerouted": "rerouted"})
        for addr in addrs:
            if len(addr) >= 2 and addr[-2] == "grid":
                plan[".".join(addr)] = "grid"
        return plan
    finally:
        bound.close()


def bind_fill_reconstruct_solver(
    frozen, grid_params: dict, *, z, filled, parent, frontier, counters,
    queued_gen, pass_p, active_p,
):
    """Bind live :class:`DataHandle` objects to a frozen reconstruction sequence."""
    _require("fill reconstruction", z=z, filled=filled, parent=parent,
             frontier=frontier, counters=counters, queued_gen=queued_gen,
             pass_p=pass_p, active_p=active_p)
    bound = frozen.build()
    zpf = {"z": z, "filled": filled, "parent": parent}
    for step in ("init_filled", "sweep_row_lr", "sweep_row_rl", "sweep_col_tb", "sweep_col_bt"):
        bound.bind_leaf(zpf, prefix=(step,))
    bound.bind_leaf({"z": z, "filled": filled, "frontier": frontier, "counters": counters}, prefix=("frontier_init",))
    bound.bind_leaf({"z": z, "filled": filled, "parent": parent, "frontier": frontier,
                     "counters": counters, "queued_gen": queued_gen}, prefix=("relax",))
    bound.bind(("relax", "active"), active_p.handle())
    bound.bind(("relax", "P"), pass_p)
    bound.bind(("zero_pass", "P"), pass_p)
    bound.bind(("bump_pass", "P"), pass_p)
    bound.bind(("zero_active", "ACTIVE"), active_p)
    bound.bind(("converged", "ACTIVE"), active_p)
    _bind_grid_everywhere(bound, grid_params)
    return bound


def make_fill_reconstruct_solver(
    be: Backend, deps: dict, grid_params: dict, *, pass_p, active_p,
    n_flat: int, nx: int, ny: int, block_size: int = 256,
    max_passes: int | None = None,
):
    """Build the reconstruction solver and return ``(FrozenSequence, params)``.

    Callers own and bind all DATA handles; only the two host-loop Parameters
    are returned in ``params``.
    """
    be = require_backend(be)
    return _build_fill_reconstruct_sequence(
        be, deps, grid_params, pass_p=pass_p, active_p=active_p,
        n_flat=n_flat, nx=nx, ny=ny, block_size=block_size,
        max_passes=max_passes,
    )


def make_depression_solver(
    be: Backend, deps: dict, grid_params: dict, *, method: str = "vanilla",
    reroute: str = "carve", n_flat: int, block_size: int = 256,
):
    """Build the depression solver and return ``(FrozenSequence, params)``.

    DATA is deliberately absent from this factory: callers bind live
    ``DataHandle`` objects to the returned sequence before compiling it.
    The result contains no live DATA binding and can be reused safely.
    """
    be = require_backend(be)
    return _build_depression_solver_sequence(
        be, deps, grid_params, method=method, reroute=reroute,
        n_flat=n_flat, block_size=block_size,
    )


# ---------------------------------------------------------------------------
# fill by grayscale morphological reconstruction - a standalone alternative
# to make_depressions/make_depression_solver, not a variant of it: no basin
# ids, no saddle search, no outlet routing - one frontier-relaxation kernel
# converging `filled`/`parent` (the receiver graph) directly to a fixed
# point. See _cupy_reconstruct.py's/_closure_reconstruct.py's own section
# notes above build_fill_reconstruct_* for the algorithm and for this
# framework's Sequence-driven, data-args-fixed-at-compile-time shape.
# ---------------------------------------------------------------------------


def make_fill_reconstruct(
    be: Backend,
    grid,
    *,
    nx: int,
    ny: int,
) -> dict:
    """
    Build one reconstruction-fill dict: `init_filled`, `sweep_row_lr`,
    `sweep_row_rl`, `sweep_col_tb`, `sweep_col_bt`, `frontier_init`, `relax`
    (all FrozenKernels, data args per _cupy_reconstruct.py's/
    _closure_reconstruct.py's build_fill_reconstruct_* docstrings).

    `relax` wires its own `P` PARAM slot (any mode, though the solver needs
    "scalar" since it bumps it between passes via a host block) - a caller
    binds a Parameter there after `.build()`, exactly like make_accumulation's
    `ITER`; there is no Need indirection anywhere in this stack. `grid` is
    the caller's `make_grid_group` FrozenGroup, composed independently by
    `init_filled`/`relax` (see make_depressions' own docstring for why -
    identical reasoning).

    `nx`/`ny` are explicit, required build-time python ints - the row-
    length/row-count split the sweep kernels need is not derivable from a
    bare FrozenGroup (which carries no bound values); n_flat is `nx * ny`.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    grid : FrozenGroup
        Grid structure, composed independently by `init_filled`/`relax`.
    nx, ny : int
        Grid dimensions.

    Returns
    -------
    dict
        {"init_filled": ..., "sweep_row_lr": ..., "sweep_row_rl": ...,
        "sweep_col_tb": ..., "sweep_col_bt": ..., "frontier_init": ...,
        "relax": ...}, all FrozenKernels.

    Author: B.G (08/2026)
    """
    be = require_backend(be)
    backend_mod = be.module
    blocks = _blocks_for(be, "reconstruct")
    n_flat_resolved = int(nx) * int(ny)
    closure = be.family == "closure"

    if closure:
        init_filled = blocks.build_fill_reconstruct_init(backend=be.name, backend_mod=backend_mod, grid=grid)
        sweeps = blocks.build_fill_reconstruct_sweeps(backend=be.name, backend_mod=backend_mod, nx=nx, ny=ny)
        frontier_init = blocks.build_fill_reconstruct_frontier_init(backend=be.name, backend_mod=backend_mod)
        relax = blocks.build_fill_reconstruct_relax(
            backend=be.name, backend_mod=backend_mod, grid=grid, n_flat=n_flat_resolved,
        )
    else:
        init_filled = blocks.build_fill_reconstruct_init(grid=grid, n_flat=n_flat_resolved)
        sweeps = blocks.build_fill_reconstruct_sweeps(nx=nx, ny=ny)
        frontier_init = blocks.build_fill_reconstruct_frontier_init(n_flat=n_flat_resolved)
        relax = blocks.build_fill_reconstruct_relax(grid=grid, n_flat=n_flat_resolved)

    out: dict = {"init_filled": init_filled, "frontier_init": frontier_init, "relax": relax}
    for name, kb in sweeps.items():
        out[f"sweep_{name}"] = kb
    return out


def _build_fill_reconstruct_sequence(
    be: Backend,
    deps: dict,
    grid_params: dict,
    *,
    pass_p,
    active_p,
    n_flat: int,
    nx: int,
    ny: int,
    block_size: int = 256,
    max_passes: int | None = None,
):
    """
    Construct the inert reconstruction-fill outer loop over a dict from
    make_fill_reconstruct:

        init_filled; sweep_row_lr; sweep_row_rl; sweep_col_tb; sweep_col_bt;
        frontier_init -> counters[0]
        zero_pass(): pass_p = 0
        loop max_times = max_passes, until = converged:
            zero_active(): active_p = 0
            relax(pass_p) -> active_p += 1 per push
            bump_pass(): pass_p += 1

    Early stop, mirroring make_depression_solver's `ndep_p`/`resolved`
    pattern: `active_p` is a caller-allocated scalar i32 Parameter, zeroed by
    a host block before each `relax` call, and `relax` itself atomic-adds
    into its raw backing buffer for every node it pushes into the next
    pass's frontier (`_closure_reconstruct.py`/`_cupy_reconstruct.py`'s
    `build_fill_reconstruct_relax`) - the same "concurrently mutated is DATA
    by definition" classification `ndep_p`/`depression_counter` already use.
    A `converged` host block reads it back with `.read()` after each
    iteration; the loop stops once a pass pushes nothing.

    The binding helper receives caller-owned DataHandles: `z` (n_flat,),
    `filled` (n_flat,),
    `parent` (n_flat,) i32 - `filled`/`parent` need no caller-side init,
    `init_filled` seeds both; `frontier` (2*n_flat,) i32 - the two ping-pong
    halves combined into one buffer (see make_fill_reconstruct's module
    note), needs no caller-side init either; `counters` (max_passes+2,) i32,
    must be **zeroed once, by the caller, before the first call** - every
    pass writes a slot it never reuses (`counters[p+1]`), so it never needs
    zeroing again, the same trick `queued_gen` uses; `queued_gen` (n_flat,)
    i32, must be **filled with -1 once, by the caller, before the first
    call** for the same reason. `pass_p` is a caller-allocated scalar i32
    Parameter (mode "scalar") - bound to `relax`'s own wired `P` PARAM slot
    and bumped here by a host block between passes. `active_p` is a second
    caller-allocated scalar i32 Parameter (mode "scalar") - no caller-side
    init needed, `zero_active` resets it every iteration before `relax` runs.

    `n_flat`/`nx`/`ny` are required - `n_flat` sets cupy's launch dimensions
    (unused on taichi/quadrants); `nx`/`ny` are only used, if `max_passes` is
    not given, for the `4 * max(nx, ny)` default below.

    `max_passes` defaults to `4 * max(nx, ny)` - generous headroom over the
    measured ~0.6x that ratio in
    experimental/LM/fill_reconstruct_optimised.py.

    The result is an inert FrozenSequence; :func:`bind_fill_reconstruct_solver`
    supplies the live DATA and grid Parameter bindings.

    Author: B.G (09/2026)
    """
    if max_passes is None:
        max_passes = 4 * max(int(nx), int(ny))

    def _zero_pass_tmpl(ctx):
        ctx.P.set(0)

    def _bump_pass_tmpl(ctx):
        ctx.P.set(int(ctx.P.read()) + 1)

    def _zero_active_tmpl(ctx):
        ctx.ACTIVE.set(0)

    def _converged_tmpl(ctx):
        return int(ctx.ACTIVE.read()) == 0

    zero_pass_hb = HostBlockBuilder(_zero_pass_tmpl).freeze()
    bump_pass_hb = HostBlockBuilder(_bump_pass_tmpl).freeze()
    zero_active_hb = HostBlockBuilder(_zero_active_tmpl).freeze()
    converged_hb = HostBlockBuilder(_converged_tmpl).freeze()

    sb = SequenceBuilder()
    sb.add("init_filled", deps["init_filled"])
    sb.add("sweep_row_lr", deps["sweep_row_lr"])
    sb.add("sweep_row_rl", deps["sweep_row_rl"])
    sb.add("sweep_col_tb", deps["sweep_col_tb"])
    sb.add("sweep_col_bt", deps["sweep_col_bt"])
    sb.add("frontier_init", deps["frontier_init"])
    sb.add("zero_pass", zero_pass_hb)
    sb.add("relax", deps["relax"])
    sb.add("bump_pass", bump_pass_hb)
    sb.add("zero_active", zero_active_hb)
    sb.add("converged", converged_hb)

    sb.step("init_filled")
    sb.step("sweep_row_lr")
    sb.step("sweep_row_rl")
    sb.step("sweep_col_tb")
    sb.step("sweep_col_bt")
    sb.step("frontier_init")
    sb.step("zero_pass")
    sb.loop(body=["zero_active", "relax", "bump_pass"], max_times=int(max_passes), until="converged")

    return sb.freeze(), {"P": pass_p, "ACTIVE": active_p}
