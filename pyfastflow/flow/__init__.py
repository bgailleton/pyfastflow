"""GPU flow-routing and depression-handling building blocks.

The factories provide receivers, drainage accumulation, depression routing,
and fill-and-reconstruct solvers. They are composed with a grid group, then
bound to the models elevation, receiver, and working arrays.
"""

import math
from importlib import import_module

from ..core import Backend, HostBlockBuilder, SequenceBuilder, require_backend
from ..noise import make_hash_u32

_MODES = frozenset({"steepest", "stochastic"})
_ACCUM_METHODS = frozenset({"atomic", "rake_compress", "pointer_jump_push"})
_MFD_TOPOLOGY_METHODS = frozenset(
    {"surface", "cordonnier_rank", "cordonnier_fill"}
)


def _blocks_for(be: Backend, section: str):
    """Return the backend module implementing one flow section."""
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
    """Return a receiver-routing kernel and its helpers.

    ``mode`` selects steepest-descent or stochastic routing.  Set
    ``h_aware`` to route on ``z + h``; choose D4 or D8 topology.

    Parameters
    ----------
    be : Backend
        Target backend.
    grid : FrozenGroup
        Grid topology helpers.
    topology : {"D4", "D8"}
        Neighbourhood used for routing.
    mode : {"steepest", "stochastic"}
        Receiver selection rule.
    diagonal_partition_correction : bool
        Apply diagonal-distance correction on D8 grids.
    h_aware : bool
        Include water depth in the routed surface.

    Returns
    -------
    dict
        ``receivers`` and the distance/slope helpers it composes.
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


def make_mfd_topology(
    be: Backend,
    grid,
    *,
    method: str = "surface",
    n_flat: int,
    topology: str = "D8",
    diagonal_partition_correction: bool = False,
    quantized_weight: bool = False,
) -> dict:
    """Return CuPy kernels that construct a persistent-MFD topology.

    ``surface`` routes on a depression-filled elevation plus a flat-distance
    field. ``cordonnier_rank`` instead consumes an SFD receiver graph after
    Cordonnier carving: carved nodes keep their single receiver, while all
    other raw-downslope MFD links are gated by decreasing receiver rank.
    ``cordonnier_fill`` converts the carved receiver paths to their maximum
    elevation and uses receiver distance only as an epsilon ordering on flats;
    MFD is otherwise independent of the carved receiver graph.

    For ``cordonnier_rank``, run ``snapshot_receivers`` before modifying the
    receiver array, then run ``receiver_rank`` and ``dirs_weights`` after the
    carve. In either mode, run ``indegree_reset`` and ``indegree_count``
    before :func:`make_accumulation` with ``method="persistent_mfd"``.
    Set ``quantized_weight=True`` to emit max-normalized ``uint8`` scores;
    the accumulation factory must receive the same option and the caller must
    bind an unsigned-byte weight buffer instead of a float buffer.
    """
    be = require_backend(be)
    if be.family != "cupy":
        raise ValueError(
            f"make_mfd_topology is cupy-only (got backend={be.name!r})"
        )
    if method not in _MFD_TOPOLOGY_METHODS:
        raise ValueError(
            f"make_mfd_topology: method must be one of "
            f"{sorted(_MFD_TOPOLOGY_METHODS)}, got {method!r}"
        )
    if topology not in ("D4", "D8"):
        raise ValueError(
            f"make_mfd_topology: topology must be 'D4' or 'D8', got {topology!r}"
        )
    if int(n_flat) < 1:
        raise ValueError("make_mfd_topology: n_flat must be positive")

    from . import _cupy_mfd_topology

    if method == "surface":
        build = _cupy_mfd_topology.build_surface_mfd_topology
    elif method == "cordonnier_rank":
        build = _cupy_mfd_topology.build_ranked_mfd_topology
    else:
        build = _cupy_mfd_topology.build_filled_rank_mfd_topology
    out = build(
        grid=grid,
        n_flat=int(n_flat),
        topology=topology,
        diagonal_partition_correction=diagonal_partition_correction,
        quantized_weight=bool(quantized_weight),
    )
    if method == "cordonnier_fill":
        out["receiver_fill"] = _cupy_mfd_topology.build_receiver_fill(
            n_flat=int(n_flat)
        )
    return out


def bind_mfd_receiver_rank(
    frozen, *, rec, ancestor, ancestor_alt, rank, rank_alt
):
    """Bind the ping-pong buffers of a ``receiver_rank`` sequence.

    The completed rank and ancestor arrays are ``rank`` and ``ancestor``;
    the ``*_alt`` arrays are scratch space of the same shape and dtype.
    """
    bound = frozen.build()
    bound.bind_leaf(
        {"rec": rec, "ancestor": ancestor, "rank": rank},
        prefix=("init",),
        strict=True,
    )
    bound.bind_leaf(
        {
            "ancestor_in": ancestor,
            "rank_in": rank,
            "ancestor_out": ancestor_alt,
            "rank_out": rank_alt,
        },
        prefix=("forward",),
        strict=True,
    )
    bound.bind_leaf(
        {
            "ancestor_in": ancestor_alt,
            "rank_in": rank_alt,
            "ancestor_out": ancestor,
            "rank_out": rank,
        },
        prefix=("backward",),
        strict=True,
    )
    return bound


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
    quantized_weight: bool = False,
) -> dict:
    """Return structures for one drainage-accumulation method.

    ``atomic`` follows a single-flow receiver tree. ``rake_compress`` and
    ``pointer_jump_push`` return host-driven sequences. ``persistent_mfd``
    is CuPy-only and accumulates a supplied MFD topology. ``n_flat`` is
    required; ``rake_compress`` and ``persistent_mfd`` need ``n_neighbours``.

    Parameters
    ----------
    be : Backend
        Target backend.
    grid : FrozenGroup
        Grid helpers, used by ``persistent_mfd``.
    method : {"atomic", "rake_compress", "pointer_jump_push", "persistent_mfd"}
        Accumulation algorithm.
    n_flat : int
        Number of raster cells.
    n_neighbours : int, optional
        Number of neighbours for rake-compress or MFD accumulation.
    fr_stage, blocks_per_sm, threads : int
        CuPy persistent-MFD launch settings.
    quantized_weight : bool
        For persistent MFD, consume max-normalized ``uint8`` scores and
        renormalize their sum per node. Must match the topology builder.

    Returns
    -------
    dict
        An accumulation kernel or a sequence and its constituent kernels.
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
                "no equivalent is available for closure backends"
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
            quantized_weight=bool(quantized_weight),
        )

    if method not in _ACCUM_METHODS:
        raise ValueError(f"make_accumulation: method must be one of {sorted(_ACCUM_METHODS)}, got {method!r}")

    # These methods return a sequence whose PARAM slots are bound by the caller.
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
    """Return unbound kernels for labelling and rerouting depressions.

    ``method`` selects vanilla or optimized basin labelling; ``reroute``
    selects carving or jumping. Use :func:`make_depression_solver` to build
    the complete label-sort-reroute loop.

    Parameters
    ----------
    be : Backend
        Target backend.
    grid : FrozenGroup
        Grid topology helpers.
    depression_counter_p : Parameter
        Scalar integer parameter used to count unresolved depressions.
    method : {"vanilla", "optimized"}
        Basin-labelling method.
    reroute : {"carve", "jump"}
        Depression rerouting method.
    n_flat : int
        Number of raster cells.

    Returns
    -------
    dict
        Frozen labelling, saddle-sorting, and rerouting components.
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

    # Keep basin identities on their original receiver routes across rounds.
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
    """Bind an address when this method combination created it."""
    if addr in bound.addresses():
        bound.bind(addr, value)


def _bind_grid_everywhere(bound, grid_params: dict) -> None:
    """Bind matching grid parameters at every nested grid address."""
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
    """Build the repeated label, saddle-sort, and reroute sequence.

    Parameters
    ----------
    deps : dict
        Components returned by :func:`make_depressions`.
    method, reroute : str
        Choices used to create ``deps``.
    n_flat : int
        Number of raster cells.

    Returns
    -------
    tuple
        A frozen sequence and its depression-counter parameter.
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
    # Seed basin routes once, then merge resolved basins at each iteration.
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
    """Bind live data to a frozen depression sequence."""
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
    """Return the named bindings required by a depression solver."""
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
    """Build a frozen fill-reconstruction sequence and its loop parameters."""
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
    """Build a frozen depression solver and its counter parameter."""
    be = require_backend(be)
    return _build_depression_solver_sequence(
        be, deps, grid_params, method=method, reroute=reroute,
        n_flat=n_flat, block_size=block_size,
    )


# Fill by grayscale morphological reconstruction.


def make_fill_reconstruct(
    be: Backend,
    grid,
    *,
    nx: int,
    ny: int,
) -> dict:
    """Return kernels for filling depressions by morphological reconstruction.

    ``nx`` and ``ny`` are build-time grid dimensions; bind the working arrays
    with :func:`bind_fill_reconstruct_solver`.

    Parameters
    ----------
    be : Backend
        Target backend.
    grid : FrozenGroup
        Grid topology helpers.
    nx, ny : int
        Raster dimensions.

    Returns
    -------
    dict
        Initialisation, sweep, frontier, and relaxation kernels.
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
    """Build the repeated frontier-relaxation sequence for terrain filling.

    Parameters
    ----------
    deps : dict
        Components returned by :func:`make_fill_reconstruct`.
    pass_p, active_p : Parameter
        Scalar parameters tracking the current pass and active frontier.
    n_flat, nx, ny : int
        Raster dimensions.
    max_passes : int, optional
        Maximum number of relaxation passes. Defaults to ``4 * max(nx, ny)``.

    Returns
    -------
    tuple
        A frozen sequence and its loop parameters.
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
