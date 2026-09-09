"""
Taichi/Quadrants (closure) block templates behind make_receivers, on the new
builder/frozen/bound stack (core/context/builder.py, frozen.py, bound.py).

Every private block is a plain python def, first parameter `ctx`, PICKED -
never branched on inside one function body - by build_receivers() according
to the caller's `mode` ("steepest"|"stochastic") and `h_aware` flag. The one
runtime branch that exists (which diagonal k values get the sqrt(2)
correction) is inside the *corrected* distance helper only - k is genuine
per-call device data, so it cannot be resolved by picking a function ahead of
time.

`dist_from_k_corrected`/`dist_between_nodes_corrected` always independently
compose their own `grid` (the caller's FrozenGroup, ../grid's own
make_grid_group result) rather than reusing one of grid's own already-
composed sub-helpers directly - uniform whether or not
`diagonal_partition_correction` is on, so `build_receivers` always has
exactly two independent `grid` occurrences to collapse (its own top-level
one, needed for can_out/N_NEIGHBOURS/neighbour, and the one nested under
`slope.dist_from_k_corrected`), never a variable number depending on the
flag - the same "always wrap, then share() collapses it" shape
../ops/_closure_blocks.py's build_slope_group uses for its own two nested
grid occurrences. `_find_param_paths`/`_share_leaf` are copied from there
(and from ../grid/__init__.py, where they first appear) rather than
imported - an explicit, itemized, per-factory declaration, never name-
matching across independently-authored composites.

`rand_unit(i, k)` mixes node index and neighbour direction separately
(mirroring noise's `_white_unit_tmpl` col/row mixing), so every (node, k)
candidate draws its own value - a node-keyed hash would scale every
candidate by the same factor and weaken the randomisation. It composes
`hash_u32` (../noise's public hash helper) rather than a private copy, and
wires its own `SEED` PARAM slot - a caller binds a Parameter there (any mode)
after `.build()`, exactly like any other PARAM slot; the host bumps it
between calls for a fresh draw.

Author: B.G (08/2026)
"""

import math

from ..core import HelperBuilder, KernelBuilder, SlotKind, share_leaf
from ._closure_shared import _tensor_annotation

_SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# distance/slope helpers
# ---------------------------------------------------------------------------


def _dist_from_k_tmpl(ctx, k):
    return ctx.grid.dist_from_k(k)


def _dist_from_k_corrected_tmpl(ctx, k):
    d = ctx.grid.dist_from_k(k)
    if k == 0 or k == 2 or k == 5 or k == 7:
        d = d / _SQRT2
    return d


def _dist_between_nodes_tmpl(ctx, i, j):
    return ctx.grid.dist_between_nodes(i, j)


def _dist_between_nodes_corrected_tmpl(ctx, i, j):
    d = ctx.grid.dist_between_nodes(i, j)
    if d > ctx.grid.DX.get(0) * 1.1:
        d = d / _SQRT2
    return d


def _slope_from_values_k_tmpl(ctx, zi, hi, zj, hj, k):
    # (zi-zj)+(hi-hj) rather than (zi+hi)-(zj+hj) - avoids float cancellation
    # when z dominates h in magnitude.
    return ((zi - zj) + (hi - hj)) / ctx.dist_from_k_corrected(k)


def _slope_between_nodes_tmpl(ctx, vi, vj, i, j):
    return (vi - vj) / ctx.dist_between_nodes_corrected(i, j)


# ---------------------------------------------------------------------------
# rand_unit(i, k): hash_u32 mixing node, neighbour direction and seed
# ---------------------------------------------------------------------------


def _rand_unit_tmpl(ctx, i, k):
    key = ctx.bk.u32(ctx.SEED.get(0))
    key ^= ctx.bk.u32(i) * ctx.bk.u32(374761393)
    key ^= ctx.bk.u32(k) * ctx.bk.u32(668265263)
    hashed = ctx.hash_u32(key)
    return float(hashed) / 4294967296.0


def build_distance_slope_helpers(grid, *, topology: str, diagonal_partition_correction: bool):
    """
    dist_from_k_corrected/dist_between_nodes_corrected/slope_from_values_k/
    slope_between_nodes, each composing its own occurrence of `grid` (see the
    module docstring). `diagonal_partition_correction` only changes anything
    when `topology == "D8"` - the "corrected" distance helpers otherwise
    simply call straight through to `grid`'s own dist_from_k/
    dist_between_nodes, no correction applied.

    Returns {name: HelperBuilder}.

    Author: B.G (08/2026)
    """
    d8 = topology == "D8"
    correct = diagonal_partition_correction and d8

    dist_from_k_tmpl = _dist_from_k_corrected_tmpl if correct else _dist_from_k_tmpl
    dist_between_tmpl = _dist_between_nodes_corrected_tmpl if correct else _dist_between_nodes_tmpl

    dist_from_k_corrected = HelperBuilder(dist_from_k_tmpl).compose("grid", grid).freeze()
    dist_between_nodes_corrected = HelperBuilder(dist_between_tmpl).compose("grid", grid).freeze()

    slope_from_values_k = (
        HelperBuilder(_slope_from_values_k_tmpl)
        .compose("dist_from_k_corrected", dist_from_k_corrected)
        .freeze()
    )
    slope_between_nodes = (
        HelperBuilder(_slope_between_nodes_tmpl)
        .compose("dist_between_nodes_corrected", dist_between_nodes_corrected)
        .freeze()
    )

    return {
        "dist_from_k_corrected": dist_from_k_corrected,
        "dist_between_nodes_corrected": dist_between_nodes_corrected,
        "slope_from_values_k": slope_from_values_k,
        "slope_between_nodes": slope_between_nodes,
    }


def build_rand_unit(hash_u32):
    """
    rand_unit(i, k) HelperBuilder, wiring its own `SEED` PARAM slot and
    composing the caller-supplied `hash_u32` (../noise's public hash helper)
    rather than a private copy - see the module docstring.

    Author: B.G (08/2026)
    """
    return HelperBuilder(_rand_unit_tmpl).compose("hash_u32", hash_u32).freeze()


def build_receivers(
    *,
    backend: str,
    backend_mod,
    grid,
    hash_u32,
    mode: str,
    topology: str,
    diagonal_partition_correction: bool,
    h_aware: bool,
):
    """
    Build one closure-backend `receivers` KernelBuilder (data args (z, rec) or
    (z, h, rec) depending on `h_aware`) plus the distance/slope (and, for
    mode="stochastic", rand_unit) HelperBuilders it is made of - picking one
    of four kernel body variants (mode x h_aware), never branching on either
    inside a single kernel body.

    `hash_u32` is the noise module's public hash_u32 FrozenHelper, reused
    here rather than re-implemented, so rand_unit and noise's own white_unit
    share the exact same integer hash. Required, and only used, when
    mode="stochastic".

    `receivers`'s own top-level PARAM slots are every name `grid` itself
    wires (NX/NY/DX/N_NEIGHBOURS, plus NODATA_MASK/OUTLET_MASK if `grid` has
    them), each build-phase-shared (`_share_leaf`) with both of `grid`'s own
    independent occurrences in this kernel's composed subtree (see the
    module docstring) - a caller binds e.g. `NX` once on the compiled
    receivers kernel, not once per occurrence.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    grid : FrozenGroup
    hash_u32 : FrozenHelper
        Required, and only used, when mode="stochastic".
    mode : str
        "steepest" or "stochastic".
    topology : str
        "D4" or "D8".
    diagonal_partition_correction : bool
    h_aware : bool

    Returns
    -------
    dict
        {name: HelperBuilder/KernelBuilder} - the distance/slope helpers
        plus "receivers", plus "rand_unit" when mode="stochastic".

    Author: B.G (08/2026)
    """
    out = build_distance_slope_helpers(grid, topology=topology, diagonal_partition_correction=diagonal_partition_correction)
    slope = out["slope_from_values_k"]
    T = _tensor_annotation(backend_mod, backend)

    if mode == "stochastic":
        out["rand_unit"] = build_rand_unit(hash_u32)

    if mode == "steepest" and not h_aware:

        def receivers_tmpl(ctx, z: T, rec: T):
            for i in z:
                if ctx.grid.can_out(i):
                    rec[i] = i
                    continue
                r = i
                sr = 0.0
                for k in range(ctx.grid.N_NEIGHBOURS.get(0)):
                    j = ctx.grid.neighbour(i, k)
                    valid = j != -1
                    tsr = -1.0
                    if valid:
                        tsr = ctx.slope(z[i], 0.0, z[j], 0.0, k)
                    better = valid and tsr > sr
                    sr = tsr if better else sr
                    r = j if better else r
                rec[i] = r

    elif mode == "steepest" and h_aware:

        def receivers_tmpl(ctx, z: T, h: T, rec: T):
            for i in z:
                if ctx.grid.can_out(i):
                    rec[i] = i
                    continue
                r = i
                sr = 0.0
                for k in range(ctx.grid.N_NEIGHBOURS.get(0)):
                    j = ctx.grid.neighbour(i, k)
                    valid = j != -1
                    tsr = -1.0
                    if valid:
                        tsr = ctx.slope(z[i], h[i], z[j], h[j], k)
                    better = valid and tsr > sr
                    sr = tsr if better else sr
                    r = j if better else r
                rec[i] = r

    elif mode == "stochastic" and not h_aware:

        def receivers_tmpl(ctx, z: T, rec: T):
            for i in z:
                if ctx.grid.can_out(i):
                    rec[i] = i
                    continue
                r = i
                sr = 0.0
                for k in range(ctx.grid.N_NEIGHBOURS.get(0)):
                    j = ctx.grid.neighbour(i, k)
                    valid = j != -1
                    tsr = -1.0
                    if valid:
                        tsr = ctx.slope(z[i], 0.0, z[j], 0.0, k)
                        if tsr > 0.0:
                            tsr = ctx.rand_unit(i, k) * ctx.bk.sqrt(tsr)
                    better = valid and tsr > sr
                    sr = tsr if better else sr
                    r = j if better else r
                rec[i] = r

    else:  # mode == "stochastic" and h_aware

        def receivers_tmpl(ctx, z: T, h: T, rec: T):
            for i in z:
                if ctx.grid.can_out(i):
                    rec[i] = i
                    continue
                r = i
                sr = 0.0
                for k in range(ctx.grid.N_NEIGHBOURS.get(0)):
                    j = ctx.grid.neighbour(i, k)
                    valid = j != -1
                    tsr = -1.0
                    if valid:
                        tsr = ctx.slope(z[i], h[i], z[j], h[j], k)
                        if tsr > 0.0:
                            tsr = ctx.rand_unit(i, k) * ctx.bk.sqrt(tsr)
                    better = valid and tsr > sr
                    sr = tsr if better else sr
                    r = j if better else r
                rec[i] = r

    kb = KernelBuilder(receivers_tmpl)
    grid_param_names = grid.slots.names(SlotKind.PARAM)
    for name in grid_param_names:
        kb.param(name)
    kb.compose("grid", grid)
    kb.compose("slope", slope)
    if mode == "stochastic":
        kb.compose("rand_unit", out["rand_unit"])

    for name in grid_param_names:
        share_leaf(kb, name)

    out["receivers"] = kb.freeze()
    return out
