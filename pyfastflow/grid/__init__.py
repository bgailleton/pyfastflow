"""
make_grid_group / make_grid_parameters: the GridContext-equivalent factory
pair, built on the new builder/frozen/bound stack (core/context/builder.py,
frozen.py, bound.py; see parameter.py for Parameter, unchanged).

There is no stateful Context class here, and no Bag: `make_grid_group`
returns a FrozenGroup (frozen.py) - a navigable, non-callable composite a
caller compose()s under a name (`kb.compose("grid", group)`) and reaches in a
template as `ctx.grid.neighbour(i, k)` (a composed HELPER call) or
`ctx.grid.NX.get(0)` (a PARAM leaf reached straight through the group, one
level in) - uniform by name whatever the backend and whatever the grid's own
topology/boundary/nodata/outlet config.

Two separate calls, structure vs. data
-----------------------------------------
`make_grid_group` returns structure: `topology`/`boundary`/`nodata`/`outlet`
pick which variant of each private block gets composed (see
_closure_blocks.py / _cupy_blocks.py), and every value a device template
reads (nx, ny, dx, n_neighbours, the two masks) is left as an unbound PARAM
slot - a FrozenGroup carries no Parameter objects of its own (frozen.py: a
frozen object is pure structure, nothing here is bind-phase data).
`make_grid_parameters` returns data: the concrete owned Parameters
(pool-backed scalar/field storage or const values) a caller then binds into
every kernel that composes this grid. The two calls must agree on
`topology`/`nodata`/`outlet` (boundary does not affect any Parameter's value,
only which block variant a device call resolves to) - passing a mismatched
pair silently produces a working-but-wrong grid, nothing here cross-checks
the two.

This two-call split is forced by the architecture, not a style choice: a
Parameter is always supplied by a caller at bind time, for every address it
fills, on every kernel that reaches it - there is no object that is
simultaneously a compose()-able recipe and a live data binding. Every later
factory (noise/ops/flow) that owns Parameters internally the way this one
owns nx/ny/dx is expected to split the same way: one function returning the
build-phase composable (whatever shape that factory's own device surface
needs - a FrozenGroup here, possibly something else elsewhere), one
returning the caller-owned Parameters that composable's PARAM slots need
bound.

Build-phase sharing collapses the duplicate addresses
---------------------------------------------------------
A device template can only call what is composed directly onto its own
scope (builder.py's module docstring) - never a sibling's. Grid's `row`/
`col` private blocks are each needed by several public helpers
(neighbour_raw, is_on_edge, which_edge, dist_between_nodes, ...), so each of
those composes its own copy of the same FrozenHelper object (shared by
identity, per frozen.py) under its own local name - which, left alone, would
mint one independent PARAM address per occurrence at build() time (bound.py)
for what is conceptually one nx/ny/dx/mask value. `share_leaf` (core/context/builder.py) is
this module's own use of GroupBuilder.share() (builder.py): after
`blocks.build_group()` composes every public helper onto the group, it walks
the group's own already-composed subtree, finds every PARAM slot literally
named "NX"/"NY"/"DX"/"NODATA_MASK"/"OUTLET_MASK" wherever it occurs, and
declares each occurrence shared with the group's own top-level slot of the
same name - one `share()` call per canonical, an explicit, itemized list
computed once here rather than hand-typed (grid's own private-block count
makes hand-typing them impractical, not a hint that a name-based mechanism
belongs in the framework itself - see bound.py's own module docstring for
why this stays declared-by-the-author, never inferred by matching name
strings across independently-authored composites). The result: a caller
composing a D8 grid sees one `grid.NX`, one `grid.NY`, one `grid.DX`, not one
per private occurrence - `bind()` once, and every reader inside the group's
own device code sees it, via bound.py's build-time redirect. A caller who
genuinely needs one occurrence to read a *different* value opts it back out
with compose()'s own `split=` (builder.py), independently of every other
occurrence.

Author: B.G (08/2026)
"""

import numpy as np

from ..core import Backend, FrozenGroup, GroupBuilder, require_backend, share_leaf

_TOPOLOGIES = {"D4": 4, "D8": 8}
_BOUNDARIES = frozenset({"normal", "periodic_EW", "periodic_NS"})
_OUTLETS = frozenset({"edge", "mask"})


def _blocks_for(be: Backend):
    """
    The private block module implementing make_grid_group's device code for
    one backend name: the closure blocks (shared by Taichi and Quadrants) or
    the cupy blocks.

    Author: B.G (08/2026)
    """
    if be.family == "closure":
        from . import _closure_blocks as blocks
    elif be.family == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"make_grid_group: unsupported backend family {be.family!r}")
    return blocks


def _check_config(topology: str, boundary: str, outlet: str) -> None:
    if topology not in _TOPOLOGIES:
        raise ValueError(f"make_grid_group: topology must be one of {sorted(_TOPOLOGIES)}, got {topology!r}")
    if boundary not in _BOUNDARIES:
        raise ValueError(f"make_grid_group: boundary must be one of {sorted(_BOUNDARIES)}, got {boundary!r}")
    if outlet not in _OUTLETS:
        raise ValueError(f"make_grid_group: outlet must be one of {sorted(_OUTLETS)}, got {outlet!r}")


def make_grid_group(
    be: Backend,
    *,
    topology: str = "D8",
    boundary: str = "normal",
    nodata: bool = False,
    outlet: str = "edge",
) -> FrozenGroup:
    """
    Build one grid's structure: a FrozenGroup wiring NX/NY/DX/N_NEIGHBOURS
    (plus NODATA_MASK if `nodata`, OUTLET_MASK if `outlet == "mask"`) as its
    own top-level PARAM slots, composing the neighbour/distance/edge public
    helper surface - uniform by name regardless of backend or config - and
    then declaring every private occurrence of each of those PARAM names
    build-phase-shared with the group's own top-level slot (`share_leaf`,
    see the module docstring), so a caller composing this group into a
    kernel binds one `grid.NX` rather than one per private block that reads
    it. Returns structure only, no Parameter objects - make_grid_parameters
    is the companion that builds those (see the module docstring for why
    the two are separate calls).

    Parameters
    ----------
    be : Backend
        Backend selecting the closure or cupy block family.
    topology : str, optional
        "D4" or "D8" (default). Picks block variants at build time.
    boundary : str, optional
        "normal" (default), "periodic_EW" or "periodic_NS".
    nodata : bool, optional
        Wires NODATA_MASK wherever a block needs it.
    outlet : str, optional
        "edge" (default) or "mask" - wires OUTLET_MASK when "mask".

    Returns
    -------
    FrozenGroup

    Raises
    ------
    ValueError
        If `topology`, `boundary` or `outlet` is not a recognised value.

    Author: B.G (08/2026)
    """
    be = require_backend(be)
    _check_config(topology, boundary, outlet)
    blocks = _blocks_for(be)

    group = GroupBuilder()
    group.param("NX")
    group.param("NY")
    group.param("DX")
    group.param("N_NEIGHBOURS")
    if nodata:
        group.param("NODATA_MASK")
    if outlet == "mask":
        group.param("OUTLET_MASK")

    blocks.build_group(group, topology=topology, boundary=boundary, nodata=nodata, outlet=outlet)

    share_leaf(group, "NX")
    share_leaf(group, "NY")
    share_leaf(group, "DX")
    if nodata:
        share_leaf(group, "NODATA_MASK")
    if outlet == "mask":
        share_leaf(group, "OUTLET_MASK")

    return group.freeze()


def make_grid_parameters(
    be: Backend,
    pool,
    nx: int,
    ny: int,
    dx: float,
    *,
    topology: str = "D8",
    nodata: bool = False,
    outlet: str = "edge",
    nx_mode: str = "const",
    ny_mode: str = "const",
    dx_mode: str = "const",
) -> dict:
    """
    Build the concrete, caller-owned Parameter objects one grid's structural
    PARAM slots need bound: {"NX": ..., "NY": ..., "DX": ...,
    "N_NEIGHBOURS": ...}, plus "NODATA_MASK"/"OUTLET_MASK" when
    `nodata`/`outlet == "mask"`. Keys match exactly the top-level PARAM slot
    names make_grid_group()'s FrozenGroup wires - and, since that group
    declares every private occurrence of each of those names build-phase-
    shared with its own top-level slot (see make_grid_group's own
    docstring), binding `"grid." + key` once is now everything a caller
    needs to do; there is no separate deep address left to also bind unless
    a caller explicitly `split`s one out (builder.py's `compose(...,
    split=...)`).

    `topology`/`nodata`/`outlet` must match whatever was passed to
    make_grid_group() for the grid this backs - see the module docstring.

    `nx_mode`/`ny_mode` default "const", may be overridden to "scalar".
    `dx_mode` defaults "const", may be overridden to "scalar" or "field" - a
    field-mode dx is allocated (one cell per node, caller fills it) but the
    public helpers that read dx (dist_from_k, dist_between_nodes) only ever
    read index 0: neither's signature carries a node to key a per-node value
    off, so a genuinely spatially-varying dx is not wired through those two
    helpers as things stand - only reachable by reading grid.DX.get(i)
    directly in a caller's own template.

    Parameters
    ----------
    be : Backend
        Backend whose Parameter class owns the returned values.
    pool : Pool
        Device-buffer pool backing scalar/field-mode Parameters.
    nx, ny : int
        Grid dimensions.
    dx : float
        Cell size.
    topology : str, optional
        Must match the value passed to make_grid_group().
    nodata : bool, optional
        Must match make_grid_group(); allocates NODATA_MASK if set.
    outlet : str, optional
        Must match make_grid_group(); allocates OUTLET_MASK if "mask".
    nx_mode, ny_mode : str, optional
        "const" (default) or "scalar".
    dx_mode : str, optional
        "const" (default), "scalar" or "field".

    Returns
    -------
    dict
        {"NX": ..., "NY": ..., "DX": ..., "N_NEIGHBOURS": ...}, plus
        "NODATA_MASK"/"OUTLET_MASK" when applicable.

    Raises
    ------
    ValueError
        If `topology`/`outlet` or any `*_mode` is not a recognised value.

    Author: B.G (08/2026)
    """
    be = require_backend(be)
    _check_config(topology, "normal", outlet)
    if nx_mode not in ("const", "scalar"):
        raise ValueError(f"make_grid_parameters: nx_mode must be 'const' or 'scalar', got {nx_mode!r}")
    if ny_mode not in ("const", "scalar"):
        raise ValueError(f"make_grid_parameters: ny_mode must be 'const' or 'scalar', got {ny_mode!r}")
    if dx_mode not in ("const", "scalar", "field"):
        raise ValueError(f"make_grid_parameters: dx_mode must be 'const', 'scalar' or 'field', got {dx_mode!r}")

    ParamCls = be.ParameterCls
    n_flat = int(nx) * int(ny)

    nx_p = ParamCls("GRID_NX", dtype="i32", mode=nx_mode, value=int(nx), pool=pool)
    ny_p = ParamCls("GRID_NY", dtype="i32", mode=ny_mode, value=int(ny), pool=pool)

    if dx_mode == "field":
        dx_p = ParamCls(
            "GRID_DX",
            dtype="f32",
            mode="field",
            value=np.full(n_flat, dx, dtype=np.float32),
            pool=pool,
            shape=(n_flat,),
        )
    else:
        dx_p = ParamCls("GRID_DX", dtype="f32", mode=dx_mode, value=float(dx), pool=pool)

    n_neighbours_p = ParamCls(
        "GRID_NNEIGHBOURS", dtype="i32", mode="const", value=_TOPOLOGIES[topology], pool=pool
    )

    params = {"NX": nx_p, "NY": ny_p, "DX": dx_p, "N_NEIGHBOURS": n_neighbours_p}

    if nodata:
        params["NODATA_MASK"] = ParamCls(
            "GRID_NODATA_MASK",
            dtype="u8",
            mode="field",
            value=np.zeros(n_flat, dtype=np.uint8),
            pool=pool,
            shape=(n_flat,),
        )

    if outlet == "mask":
        params["OUTLET_MASK"] = ParamCls(
            "GRID_OUTLET_MASK",
            dtype="u8",
            mode="field",
            value=np.zeros(n_flat, dtype=np.uint8),
            pool=pool,
            shape=(n_flat,),
        )

    return params
