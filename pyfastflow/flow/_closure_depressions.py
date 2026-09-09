"""
Taichi/Quadrants (closure) block templates behind make_depressions/
make_depression_solver: copy_field, both basin labelling variants,
saddlesort, both carve variants, jump reroute, and the depression counter -
on the new builder/frozen/bound/routine/sequence stack (../core/context/
builder.py, frozen.py, bound.py, routine.py, sequence.py).

See _closure_receivers.py/_closure_accum.py/_closure_reconstruct.py for
the other flow algorithms. Based on ../../flow/flow_reroute_kernels.py,
`pack`/`unpack_value`/`unpack_index`
composed from ops.make_bitpack_group (a FrozenGroup, `ctx.bitpack.pack(f,
i)`) in place of legacy's f32_i32_struct module. Every array here (rec, bid,
tag, basin_saddle, outlet, ...) is n_flat-sized, basin id = pit index + 1, so
a per-basin array is safely indexed by any node index too - the same double
duty the legacy kernels rely on.

`grid` is the caller's FrozenGroup (../grid's make_grid_group result),
composed under the name "grid" onto whichever KernelBuilder needs
`ctx.grid.can_out(i)`/`ctx.grid.neighbour(i, k)`/`ctx.grid.N_NEIGHBOURS.
get(0)` - exactly the idiom _closure_receivers.py's build_receivers already
established; `ctx.grid.N_NEIGHBOURS.get(0)` inside `range(...)` resolves to a
plain python int at trace time when N_NEIGHBOURS is bound const-mode (the
make_grid_parameters default), unrolling the neighbour loop exactly as
receivers' own does. Every site here composes its own independent occurrence
of `grid` - basin_id_init/label_basins_walk/border_zprime/atomic_min_saddle/
find_saddlenode/atomic_min_outlet/depression_counter each mint their own
`{step}.grid.NX`/etc address (build-phase sharing, `share()`, only collapses
occurrences *within* one KernelBuilder's own composed subtree - never across
sibling steps of a routine/sequence, see bound.py's own module docstring and
_closure_accum.py's ITER note), so a caller binds the same grid Parameter
object at every one of those addresses - enumerated per build_* docstring
below, the same multi-address idiom make_accumulation's `ITER`/`SOURCE`
already established.

`n_flat`, where needed (label_basins_walk's/depression_counter's guard
bounds, saddlesort/reroute have none), is a plain build-time python int -
this factory takes no pool and reads no Parameter for it, consistent with
"atomic" (_closure_accum.py) requiring it explicitly rather than reading a
bare FrozenGroup's absent bound values.

A fixed, build-time-constant repeat (propagate_basin_iter's `logn+1` rounds
inside vanilla basin labelling) is unrolled as `logn+1` distinct routine
compose() names for the SAME propagate_basin_iter FrozenKernel - the
instancing idiom routine.py's own module docstring documents ("composing
the same FrozenKernel object under two different step names... two
independently bindable slot sets") - not a SequenceBuilder loop: unlike
rake_compress_accum's own host-invisible bump, there is no per-round host
readback here, the round count is a plain python int fixed at build time, so
there is nothing a loop's host-side bookkeeping buys over a flat unroll (the
same choice ../ops/_closure_blocks.py's build_scan_routine makes for its own
log-depth passes, contrasted with _closure_accum.py's own SequenceBuilder
loop for rake_compress_accum's per-round-identical body).

Author: B.G (08/2026)
"""

from ..core import KernelBuilder, RoutineBuilder
from ._closure_shared import _tensor_annotation


def build_copy_field(*, backend: str, backend_mod):
    """
    dst[i] = src[i] over a whole flat buffer - the generic copy used
    everywhere a depression pass needs one buffer snapshotted into another
    (rec -> rec_jump, rec_work -> rec, ...), reused across every routine that
    needs it (composed under a distinct name at each use site - routine.py's
    instancing) rather than rebuilt per call site.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def copy_field_tmpl(ctx, src: T, dst: T):
        for i in src:
            dst[i] = src[i]

    return KernelBuilder(copy_field_tmpl).freeze()


def build_basin_id_init(*, backend: str, backend_mod, grid):
    """
    bid[i] = 0 on a can_out node, i+1 otherwise - the seed for vanilla basin
    labelling's pointer-jump propagation. Data arg (bid,); composes its own
    `grid` occurrence (`ctx.grid.can_out(i)`).

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    grid : FrozenGroup

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def basin_id_init_tmpl(ctx, bid: T):
        for i in bid:
            bid[i] = 0 if ctx.grid.can_out(i) else (i + 1)

    return KernelBuilder(basin_id_init_tmpl).compose("grid", grid).freeze()


def build_propagate_basin_iter(*, backend: str, backend_mod):
    """
    One pointer-jump step over `rec_jump`, halving path length to the root
    each call. Data arg (rec_jump,).

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def propagate_basin_iter_tmpl(ctx, rec_jump: T):
        for i in rec_jump:
            if rec_jump[i] != rec_jump[rec_jump[i]]:
                rec_jump[i] = rec_jump[rec_jump[i]]

    return KernelBuilder(propagate_basin_iter_tmpl).freeze()


def build_propagate_basin_final(*, backend: str, backend_mod):
    """
    bid[i] = bid[root(i)] - finalizes vanilla basin labelling once
    `rec_jump` has been pointer-jumped down to (near-)roots. Data args (bid,
    rec_jump).

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def propagate_basin_final_tmpl(ctx, bid: T, rec_jump: T):
        for i in bid:
            bid[i] = bid[rec_jump[i]]

    return KernelBuilder(propagate_basin_final_tmpl).freeze()


def build_basin_labelling_vanilla(*, backend: str, backend_mod, grid, copy_field, logn: int):
    """
    RoutineBuilder (routine) for vanilla basin labelling: basin_id_init(bid);
    copy_field(rec -> rec_jump); logn+1 unrolled propagate_basin_iter(rec_jump)
    rounds (composed under "propagate_iter_0".."propagate_iter_{logn}" - see
    the module docstring for the unroll-vs-loop choice); propagate_basin_final
    (bid, rec_jump).

    Composed step names: "basin_id_init", "copy_rec_to_recjump",
    "propagate_iter_0" .. "propagate_iter_{logn}", "propagate_basin_final".
    Data addresses: "basin_id_init.bid", "copy_rec_to_recjump.src"/".dst"
    (bound to rec/rec_jump), "propagate_iter_K.rec_jump" for every K in
    0..logn (bound to the same rec_jump buffer at each), "propagate_basin_
    final.bid"/".rec_jump". PARAM address: "basin_id_init.grid.*" (NX/NY/DX/
    N_NEIGHBOURS[, NODATA_MASK/OUTLET_MASK]) - one occurrence, since only
    basin_id_init reaches grid in this routine.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    grid : FrozenGroup
    copy_field : KernelBuilder
        Composed under "copy_rec_to_recjump".
    logn : int

    Returns
    -------
    tuple[RoutineBuilder, dict]
        kernels_dict holds "basin_id_init", "propagate_basin_iter" (the
        shared, unrolled kernel), "propagate_basin_final" (not
        "copy_field" - the caller's own to keep track of, shared across
        every routine that needs a copy).

    Author: B.G (08/2026)
    """
    basin_id_init = build_basin_id_init(backend=backend, backend_mod=backend_mod, grid=grid)
    propagate_basin_iter = build_propagate_basin_iter(backend=backend, backend_mod=backend_mod)
    propagate_basin_final = build_propagate_basin_final(backend=backend, backend_mod=backend_mod)

    kernels = {
        "basin_id_init": basin_id_init,
        "propagate_basin_iter": propagate_basin_iter,
        "propagate_basin_final": propagate_basin_final,
    }

    rb = RoutineBuilder()
    rb.step("basin_id_init", basin_id_init)
    rb.step("copy_rec_to_recjump", copy_field)
    for k in range(logn + 1):
        rb.step(f"propagate_iter_{k}", propagate_basin_iter)
    rb.step("propagate_basin_final", propagate_basin_final)

    return rb, kernels


def build_basin_labelling_optimized(*, backend: str, backend_mod, grid, n_flat: int):
    """
    label_basins_walk KernelBuilder - one launch: copy rec -> rec_jump,
    per-thread path-halving to the root under a `guard < n_flat` bound
    (races are benign - entries only ever move rootward), then
    bid[i] = 0 if can_out(root) else root + 1.

    Data args (rec, rec_jump, bid). Composes its own `grid` occurrence
    (`ctx.grid.can_out(root)`), one PARAM address group per grid slot at
    "grid.*" (this kernel's own top level, since it is the only occurrence).

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    grid : FrozenGroup
    n_flat : int

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NFLAT = n_flat

    def label_basins_walk_tmpl(ctx, rec: T, rec_jump: T, bid: T):
        for i in rec_jump:
            rec_jump[i] = rec[i]
        for i in rec_jump:
            guard = 0
            while rec_jump[i] != rec_jump[rec_jump[i]] and guard < NFLAT:
                rec_jump[i] = rec_jump[rec_jump[i]]
                guard += 1
        for i in bid:
            root = rec_jump[i]
            bid[i] = 0 if ctx.grid.can_out(root) else root + 1

    return KernelBuilder(label_basins_walk_tmpl).compose("grid", grid).freeze()


def build_label_from_route(*, backend: str, backend_mod, grid):
    """
    bid[i] = 0 if the node basin_route reaches can output, else root + 1.

    Labels each node by the basin it reaches through `basin_route` (contracted
    to a fixed point beforehand), which carries basin identity across rounds
    via accumulated merges - unlike re-deriving from the carved receivers.
    Data args (bid, basin_route); composes `grid` for `can_out`.

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def label_from_route_tmpl(ctx, bid: T, basin_route: T):
        for i in bid:
            root = basin_route[i]
            bid[i] = 0 if ctx.grid.can_out(root) else root + 1

    return KernelBuilder(label_from_route_tmpl).compose("grid", grid).freeze()


def build_basin_labelling_route(*, backend: str, backend_mod, grid, logn: int):
    """
    RoutineBuilder for basin labelling from the carried `basin_route`: logn+1
    unrolled pointer-jump contractions of `basin_route` (composed under
    "contract_0".."contract_{logn}", the same instancing idiom vanilla
    labelling uses), then label_from_route(bid, basin_route).

    Composed step names: "contract_0".."contract_{logn}", "label_from_route".
    Data addresses: "contract_K.rec_jump" for K in 0..logn (all bound to the
    basin_route buffer), "label_from_route.bid"/".basin_route". PARAM address:
    "label_from_route.grid.*".

    Returns (routine_builder, kernels_dict) with keys "propagate_basin_iter"
    (the shared unrolled kernel) and "label_from_route".

    Author: B.G (08/2026)
    """
    propagate_basin_iter = build_propagate_basin_iter(backend=backend, backend_mod=backend_mod)
    label_from_route = build_label_from_route(backend=backend, backend_mod=backend_mod, grid=grid)

    kernels = {"propagate_basin_iter": propagate_basin_iter, "label_from_route": label_from_route}

    rb = RoutineBuilder()
    for k in range(logn + 1):
        rb.step(f"contract_{k}", propagate_basin_iter)
    rb.step("label_from_route", label_from_route)

    return rb, kernels


def build_merge_basin_route(*, backend: str, backend_mod, bitpack):
    """
    Fold every kept basin into the outlet node it drains through:
    basin_route[pit] = outlet_node (the original FastFlow update_basin_route),
    so contraction next round folds the whole resolved basin into its receiver
    (or the boundary) and the basin graph strictly shrinks. Data args (outlet,
    basin_route); composes `bitpack` for `unpack_index`. basin id = pit + 1, so
    basin i's pit is node i - 1.

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def merge_basin_route_tmpl(ctx, outlet: T, basin_route: T):
        invalid_m = ctx.bitpack.pack(1e8, 42)
        for i in outlet:
            if i == 0 or outlet[i] == invalid_m:
                continue
            p_rcv = ctx.bitpack.unpack_index(outlet[i])
            basin_route[i - 1] = p_rcv

    return KernelBuilder(merge_basin_route_tmpl).compose("bitpack", bitpack).freeze()


def build_saddlesort(*, backend: str, backend_mod, grid, bitpack):
    """
    RoutineBuilder (routine) for the six saddlesort passes: border/z_prime
    detection, saddle/outlet/saddlenode init, bitpacked atomic-min saddle
    search, saddlenode identification, bitpacked atomic-min outlet search,
    basin-graph 2-cycle break. Shared unchanged by both `method`s.

    `bitpack` is the FrozenGroup ops.make_bitpack_group returns
    (`ctx.bitpack.pack`/`.unpack_value`/`.unpack_index`) - each KernelBuilder
    below composes its own occurrence, only where it actually calls one of
    those three. `grid` is likewise composed independently at every site
    that needs `can_out`/`neighbour`.

    Returns (routine_builder, kernels_dict) - kernels_dict keys
    "border_zprime", "init_saddle_outlet", "atomic_min_saddle",
    "find_saddlenode", "atomic_min_outlet", "break_cycle".

    Composed step names: same as the kernels_dict keys. Data addresses:
    "border_zprime.bid"/".z"/".z_prime"/".is_border",
    "init_saddle_outlet.basin_saddle"/".outlet"/".basin_saddlenode",
    "atomic_min_saddle.bid"/".is_border"/".z_prime"/".basin_saddle",
    "find_saddlenode.bid"/".is_border"/".z_prime"/".basin_saddle"/
    ".basin_saddlenode", "atomic_min_outlet.bid"/".basin_saddle"/
    ".basin_saddlenode"/".z"/".outlet", "break_cycle.bid"/".outlet"/
    ".basin_saddle"/".basin_saddlenode". PARAM addresses: "border_zprime.
    grid.*", "atomic_min_saddle.grid.*", "find_saddlenode.grid.*",
    "atomic_min_outlet.grid.*" - four independent grid occurrences, same
    Parameter bound at each (see the module docstring).

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    grid : FrozenGroup
    bitpack : FrozenGroup
        ops.make_bitpack_group's result.

    Returns
    -------
    tuple[RoutineBuilder, dict]

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def border_zprime_tmpl(ctx, bid: T, z: T, z_prime: T, is_border: T):
        for i in z:
            if ctx.grid.can_out(i):
                z_prime[i] = z[i]
                continue
            is_border[i] = 0
            z_prime[i] = 1e9
            zn = 1e9
            for k in range(ctx.grid.N_NEIGHBOURS.get(0)):
                j = ctx.grid.neighbour(i, k)
                if j != -1 and bid[j] != bid[i]:
                    is_border[i] = 1
                    zn = min(zn, z[j])
            if is_border[i]:
                z_prime[i] = max(z[i], zn)

    def init_saddle_outlet_tmpl(ctx, basin_saddle: T, outlet: T, basin_saddlenode: T, b_rcv: T):
        invalid_i = ctx.bitpack.pack(1e8, 42)
        for i in basin_saddle:
            basin_saddle[i] = invalid_i
            outlet[i] = invalid_i
            basin_saddlenode[i] = -1
            b_rcv[i] = 0

    def atomic_min_saddle_tmpl(ctx, bid: T, is_border: T, z_prime: T, basin_saddle: T):
        invalid_a = ctx.bitpack.pack(1e8, 42)
        for i in bid:
            if not is_border[i]:
                continue
            tbid = bid[i]
            res = invalid_a
            for k in range(ctx.grid.N_NEIGHBOURS.get(0)):
                j = ctx.grid.neighbour(i, k)
                if j != -1 and bid[j] != tbid:
                    candidate = ctx.bitpack.pack(z_prime[i], bid[j])
                    res = min(res, candidate)
            if res != invalid_a:
                ctx.bk.atomic_min(basin_saddle[tbid], res)

    def find_saddlenode_tmpl(ctx, bid: T, is_border: T, z_prime: T, basin_saddle: T, basin_saddlenode: T):
        for i in bid:
            if not is_border[i] or bid[i] == 0:
                continue
            target_z = ctx.bitpack.unpack_value(basin_saddle[bid[i]])
            target_b = ctx.bitpack.unpack_index(basin_saddle[bid[i]])
            is_here = False
            for k in range(ctx.grid.N_NEIGHBOURS.get(0)):
                j = ctx.grid.neighbour(i, k)
                if j != -1 and bid[j] == target_b and z_prime[i] == target_z:
                    is_here = True
            if is_here:
                basin_saddlenode[bid[i]] = i

    def atomic_min_outlet_tmpl(ctx, bid: T, basin_saddle: T, basin_saddlenode: T, z: T, outlet: T, b_rcv: T):
        invalid_o = ctx.bitpack.pack(1e8, 42)
        for i in bid:
            if i == 0 or basin_saddle[i] == invalid_o:
                continue
            node = basin_saddlenode[i]
            best_z = 1e9
            best_b = 2147483647
            rec_out = -1
            for k in range(ctx.grid.N_NEIGHBOURS.get(0)):
                j = ctx.grid.neighbour(node, k)
                if j != -1 and bid[j] != i:
                    cz = max(z[node], z[j])
                    bj = bid[j]
                    if cz < best_z or (cz == best_z and bj < best_b):
                        best_z = cz
                        best_b = bj
                        rec_out = j
            if rec_out > -1:
                outlet[i] = ctx.bitpack.pack(best_z, rec_out)
                b_rcv[i] = bid[rec_out]

    def set_keep_tmpl(ctx, bid: T, b_rcv: T, outlet: T, basin_saddle: T, basin_saddlenode: T):
        invalid_c = ctx.bitpack.pack(1e8, 42)
        for i in bid:
            if i == 0 or outlet[i] == invalid_c:
                continue
            brd = b_rcv[i]
            if brd != 0 and b_rcv[brd] == i and brd > i:
                outlet[i] = invalid_c
                basin_saddle[i] = invalid_c
                basin_saddlenode[i] = -1

    border_zprime = KernelBuilder(border_zprime_tmpl).compose("grid", grid).freeze()
    init_saddle_outlet = KernelBuilder(init_saddle_outlet_tmpl).compose("bitpack", bitpack).freeze()
    atomic_min_saddle = KernelBuilder(atomic_min_saddle_tmpl).compose("grid", grid).compose("bitpack", bitpack).freeze()
    find_saddlenode = KernelBuilder(find_saddlenode_tmpl).compose("grid", grid).compose("bitpack", bitpack).freeze()
    atomic_min_outlet = KernelBuilder(atomic_min_outlet_tmpl).compose("grid", grid).compose("bitpack", bitpack).freeze()
    set_keep = KernelBuilder(set_keep_tmpl).compose("bitpack", bitpack).freeze()

    kernels = {
        "border_zprime": border_zprime,
        "init_saddle_outlet": init_saddle_outlet,
        "atomic_min_saddle": atomic_min_saddle,
        "find_saddlenode": find_saddlenode,
        "atomic_min_outlet": atomic_min_outlet,
        "break_cycle": set_keep,
    }

    rb = RoutineBuilder()
    rb.step("border_zprime", border_zprime)
    rb.step("init_saddle_outlet", init_saddle_outlet)
    rb.step("atomic_min_saddle", atomic_min_saddle)
    rb.step("find_saddlenode", find_saddlenode)
    rb.step("atomic_min_outlet", atomic_min_outlet)
    rb.step("break_cycle", set_keep)

    return rb, kernels


def build_reroute_carve_vanilla(*, backend: str, backend_mod, bitpack, copy_field, logn: int):
    """
    RoutineBuilder (routine) for carve+vanilla reroute: init_reroute_carve
    (tag, tag_alt, basin_saddlenode); copy_field(rec_work -> rec);
    copy_field(rec_work -> rec_jump); logn+1 unrolled iteration_reroute_carve
    (tag, tag_alt, rec, rec_work, bid) rounds; finalise_reroute_carve(rec,
    rec_jump, tag, basin_saddlenode, outlet, rerouted); copy_field(rec ->
    rec_work).

    finalise's second data arg is bound to `rec_jump` - the snapshot taken
    of `rec_work` *before* the repeat block ran, i.e. the original,
    unjumped receiver chain - not the pointer-jumped `rec` the repeat block
    produced. The repeat block's pointer jumping is only used internally to
    propagate `tag` quickly; the actual edge reversal in finalise operates
    on the original chain, which is why finalise's own first statement
    resets `rec` from that original snapshot before reversing anything -
    ported exactly as legacy's flow_reroute_kernels.py has it.

    Composed step names: "init_reroute_carve", "copy_recwork_to_rec",
    "copy_recwork_to_recjump", "iteration_carve_0".."iteration_carve_{logn}",
    "finalise_reroute_carve", "copy_rec_to_recwork". Data addresses: "rec",
    "rec_work", "rec_jump", "tag", "tag_alt", "bid", "basin_saddlenode",
    "outlet", "rerouted" under each step's own name (see each template's
    signature above); `finalise_reroute_carve` composes `bitpack`.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    bitpack : FrozenGroup
    copy_field : KernelBuilder
    logn : int

    Returns
    -------
    tuple[RoutineBuilder, dict]

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def init_reroute_carve_tmpl(ctx, tag: T, tag_alt: T, saddlenode: T):
        for i in tag:
            tag[i] = 0
        for i in tag:
            if saddlenode[i] != -1:
                tag[saddlenode[i]] = 1
        for i in tag:
            tag_alt[i] = tag[i]

    def iteration_reroute_carve_tmpl(ctx, tag: T, tag_alt: T, rec: T, rec_work: T, bid: T):
        for i in tag:
            if bid[i] == 0:
                continue
            if tag[i] and rec[i] != i:
                tag_alt[rec[i]] = 1
            rec_work[i] = rec[i]
        for i in tag:
            if bid[i] == 0:
                continue
            if rec_work[i] != i:
                rec[i] = rec_work[rec_work[i]]
            tag[i] = tag_alt[i]

    def finalise_reroute_carve_tmpl(ctx, rec: T, rec_orig: T, tag: T, saddlenode: T, outlet: T, rerouted: T):
        invalid = ctx.bitpack.pack(1e8, 42)
        for i in rec:
            rec[i] = rec_orig[i]
        for i in rec:
            if tag[rec_orig[i]] and tag[i] and i != rec_orig[i]:
                rec[rec_orig[i]] = i
                rerouted[rec_orig[i]] = 1
        for i in rec:
            if outlet[i] != invalid:
                node = ctx.bitpack.unpack_index(outlet[i])
                rec[saddlenode[i]] = node
                rerouted[saddlenode[i]] = 1

    init_reroute_carve = KernelBuilder(init_reroute_carve_tmpl).freeze()
    iteration_reroute_carve = KernelBuilder(iteration_reroute_carve_tmpl).freeze()
    finalise_reroute_carve = KernelBuilder(finalise_reroute_carve_tmpl).compose("bitpack", bitpack).freeze()

    kernels = {
        "init_reroute_carve": init_reroute_carve,
        "iteration_reroute_carve": iteration_reroute_carve,
        "finalise_reroute_carve": finalise_reroute_carve,
    }

    rb = RoutineBuilder()
    rb.step("init_reroute_carve", init_reroute_carve)
    rb.step("copy_recwork_to_rec", copy_field)
    rb.step("copy_recwork_to_recjump", copy_field)
    for k in range(logn + 1):
        rb.step(f"iteration_carve_{k}", iteration_reroute_carve)
    rb.step("finalise_reroute_carve", finalise_reroute_carve)
    rb.step("copy_rec_to_recwork", copy_field)

    return rb, kernels


def build_reroute_carve_optimized(*, backend: str, backend_mod, bitpack, n_flat: int):
    """
    carve_basins_serial KernelBuilder - one launch, one serial thread per
    basin walking `rec` from the saddle node to the pit reversing links,
    then saddle -> outlet.

    The saddle->pit walk is bounded by a `guard < n_flat` counter: distinct
    basins' chains are expected node-disjoint (so the reversals never race),
    but the shared saddlesort's `find_saddlenode` is non-atomic on ties, and a
    racy saddlenode can produce overlapping/non-terminating chains. The guard
    turns that rare case into an incomplete carve (a pit survives to the next
    round) rather than an unbounded in-kernel hang - the same bound
    label_basins_walk already applies to its own per-thread walk. (`vanilla`'s
    pointer-jump carve is inherently bounded and needs no such guard.)

    Data args (rec, basin_saddlenode, outlet). Composes `bitpack`.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    bitpack : FrozenGroup
    n_flat : int
        Walk guard bound.

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NFLAT = n_flat

    def carve_basins_serial_tmpl(ctx, rec: T, basin_saddlenode: T, outlet: T):
        invalid = ctx.bitpack.pack(1e8, 42)
        for b in basin_saddlenode:
            s = basin_saddlenode[b]
            if s == -1 or outlet[b] == invalid:
                continue
            out_node = ctx.bitpack.unpack_index(outlet[b])
            node = s
            nxt = rec[node]
            rec[node] = out_node
            guard = 0
            while nxt != node and guard < NFLAT:
                nnxt = rec[nxt]
                rec[nxt] = node
                node = nxt
                nxt = nnxt
                guard += 1

    return KernelBuilder(carve_basins_serial_tmpl).compose("bitpack", bitpack).freeze()


def build_reroute_jump(*, backend: str, backend_mod, bitpack):
    """
    reroute_jump KernelBuilder - one launch, pit points straight at the
    outlet. Shared unchanged by both `method`s: called with the currently
    resolved receiver buffer bound to its `rec` argument, whichever buffer
    that is for a given method.

    The write is deliberately `rec[i - 1]`, not `rec[i]`: the loop is over
    basin ids (`i` ranges over `outlet`'s own index space) and basin id =
    pit index + 1, so `i - 1` is the pit node. Ported exactly as legacy has
    it - see _cupy_depressions.py's build_reroute_jump and make_depressions'
    docstring for the same note.

    Data args (rec, outlet, rerouted). Composes `bitpack`.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    bitpack : FrozenGroup

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def reroute_jump_tmpl(ctx, rec: T, outlet: T, rerouted: T):
        invalid = ctx.bitpack.pack(1e8, 42)
        for i in rerouted:
            rerouted[i] = 0
        for i in rec:
            if outlet[i] != invalid:
                rrec = ctx.bitpack.unpack_index(outlet[i])
                rec[i - 1] = rrec
                rerouted[i - 1] = 1

    return KernelBuilder(reroute_jump_tmpl).compose("bitpack", bitpack).freeze()


def build_depression_counter(*, backend: str, backend_mod, grid):
    """
    depression_counter KernelBuilder, data args (rec, ndep): atomic-adds 1
    into `ndep[None]` for every self-receiving node that cannot drain. The
    caller must reset the backing scalar Parameter to 0 (`.set(0)`) before
    each launch - this kernel only accumulates, mirroring ops.Reduce.
    run_sum's own reset-then-launch pattern. `ndep` is wired as DATA (the raw
    backing field, `ndep_p.handle().array`), not PARAM - a genuinely concurrent
    atomic accumulate needs the raw field, the same "concurrently mutated is
    DATA by definition" classification make_accumulation's own atomic `q`
    and ops.Reduce's own accumulators already use; PARAM access stays strict
    get()/set_node(). Composes its own `grid` occurrence (`ctx.grid.
    can_out(i)`).

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    grid : FrozenGroup

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def depression_counter_tmpl(ctx, rec: T, ndep: T):
        for i in rec:
            if rec[i] == i and not ctx.grid.can_out(i) and not ctx.grid.nodata(i):
                ctx.bk.atomic_add(ndep[None], 1)

    return KernelBuilder(depression_counter_tmpl).compose("grid", grid).freeze()
