"""Rules moving data between leaves when a quadtree refines or coarsens.

Each rule fills the leaves a tree update creates; values on kept leaves do not
change. Pass ``(field, rule)`` pairs to :meth:`DynamicQuadtree.adapt`:

=================  ===================================  =======================
Rule               Coarse to fine                       Fine to coarse
=================  ===================================  =======================
``Mean``           copy                                 mean
``Sum``            split evenly                         sum
``Min``, ``Max``   copy                                 min, max
``Linear``         limited linear reconstruction        mean
``FromRaster``     aggregate a source raster            aggregate a source raster
``Depth``          flat water surface, volume kept      mean
``Velocity``       copy                                 depth-weighted mean
``FaceDischarge``  split along each face                sum along each face
=================  ===================================  =======================

``Mean``, ``Sum``, ``Linear``, ``Depth`` and ``FaceDischarge`` conserve their
integral: area-weighted for ``Mean``, ``Linear`` and ``Depth``, plain for
``Sum`` and ``FaceDischarge``. ``Velocity`` with ``Depth`` conserves momentum.
``Mean``, ``Sum`` and ``FromRaster`` also work with
:meth:`DynamicQuadtree.regrid`.

Author: B.G (11/2026)
"""

import cupy as cp

from ._ops import owner, neighbour_rows, pad_to_levels, pyramid, side


class Transfer:
    """Base rule for one field."""

    per_link = False
    regrid_ok = False
    order = 0

    def check(self, tree, arr, targets):
        """Reject a field this rule cannot move."""
        if arr.dtype.kind != "f":
            raise TypeError(f"{type(self).__name__}: needs a float field, got {arr.dtype}")

    def apply(self, arr, step):
        """Rewrite ``arr`` for the new tree described by ``step``."""
        children = self.children(arr[: step.n0].copy(), step)
        parents = self.parents(arr[: step.n0].copy(), step)
        write_leaves(arr, step, children, parents)

    def children(self, old, step):
        """Return ``(4, n_split)`` values, ordered low x low y to high x high y."""
        raise NotImplementedError

    def parents(self, old, step):
        """Return one value per merged sibling group."""
        raise NotImplementedError


def write_leaves(arr, step, children, parents):
    """Assemble kept, child and parent values into the new leaf order."""
    old = arr[: step.n0]
    out = cp.empty(step.n1, dtype=arr.dtype)
    out[step.mapping[step.keep]] = old[step.keep]
    if step.n_split:
        out[step.child_ids] = children.reshape(-1).astype(arr.dtype)
    if step.groups.size:
        out[step.parent_ids] = parents.astype(arr.dtype)
    arr[: step.n1] = out
    arr[step.n1 :] = 0


def _siblings(old, step):
    return cp.stack([old[step.groups + d] for d in range(4)])


class Mean(Transfer):
    """Copy to children, average to a parent; for intensive values."""

    regrid_ok = True

    def children(self, old, step):
        return cp.tile(old[step.split_ids], (4, 1))

    def parents(self, old, step):
        return _siblings(old, step).mean(0)


class Sum(Transfer):
    """Split evenly to children, sum to a parent; for extensive values."""

    regrid_ok = True

    def children(self, old, step):
        return cp.tile(old[step.split_ids] * 0.25, (4, 1))

    def parents(self, old, step):
        return _siblings(old, step).sum(0)


class Min(Mean):
    """Copy to children, keep the smallest sibling in a parent."""

    regrid_ok = False

    def parents(self, old, step):
        return _siblings(old, step).min(0)


class Max(Mean):
    """Copy to children, keep the largest sibling in a parent."""

    regrid_ok = False

    def parents(self, old, step):
        return _siblings(old, step).max(0)


def _minmod(a, b):
    return cp.where(a * b > 0, cp.where(cp.abs(a) < cp.abs(b), a, b), 0)


class Linear(Mean):
    """Second-order prolongation: a minmod-limited gradient inside each parent.

    Children average exactly to their parent and stay within the values of the
    parent and its face neighbours. A domain edge gives a zero gradient on its
    axis. Restriction is the mean.
    """

    regrid_ok = False

    def children(self, old, step):
        ids = step.split_ids
        dom = step.dom
        table = neighbour_rows(dom, step.old_level, step.old_x, step.old_y, step.old_key, ids)
        first = (1, 4, 6, 9) if dom.d8 else (0, 2, 4, 6)
        centre = old[ids].astype(cp.float64)
        s = side(step.old_level[ids]).astype(cp.float64)
        slopes = []
        for f in range(4):
            a, b = table[:, first[f]], table[:, first[f] + 1]
            pair = cp.stack([a, b])
            filled = pair >= 0
            count = filled.sum(0)
            safe = cp.where(filled, pair, 0)
            total = (old[safe].astype(cp.float64) * filled).sum(0)
            size = (side(step.old_level[safe]).astype(cp.float64) * filled).sum(0)
            has = count > 0
            div = cp.where(has, count, 1)
            value, nbr_side = total / div, size / div
            distance = 0.5 * (s + nbr_side)
            sign = -1.0 if f in (0, 1) else 1.0
            slopes.append(cp.where(has, sign * (value - centre) / distance, cp.nan))
        def axis(lo, hi):
            lo_ok, hi_ok = ~cp.isnan(lo), ~cp.isnan(hi)
            return cp.where(lo_ok & hi_ok, _minmod(cp.nan_to_num(lo), cp.nan_to_num(hi)), 0.0)
        gy = axis(slopes[0], slopes[3])
        gx = axis(slopes[1], slopes[2])
        q = 0.25 * s
        return cp.stack([centre - gx * q - gy * q, centre + gx * q - gy * q,
                         centre - gx * q + gy * q, centre + gx * q + gy * q])


class FromRaster(Transfer):
    """Aggregate a finest-resolution raster over each new leaf.

    ``op`` is ``mean``, ``min``, ``max`` or ``sum``. A bed elevation keeps the
    detail of its source DEM this way, whatever the update history.
    """

    regrid_ok = True

    def __init__(self, raster, op="mean"):
        if op not in ("mean", "min", "max", "sum"):
            raise ValueError(f"FromRaster: unknown op {op!r}")
        self.raster, self.op = raster, op
        self._pyramid = None

    def levels_of(self, tree):
        if self._pyramid is None:
            arr = cp.asarray(self.raster)
            if arr.shape != (tree.ny, tree.nx):
                raise ValueError(f"FromRaster: shape {arr.shape} differs from ({tree.ny}, {tree.nx})")
            self._pyramid = pyramid(pad_to_levels(arr, tree._dom, 0), self.op, tree.levels)
        return self._pyramid

    def sample(self, tree, level, x, y, dtype):
        """Return the aggregate over the given leaves."""
        pyr = self.levels_of(tree)
        out = cp.empty(level.size, dtype=dtype)
        for l in range(tree.levels + 1):
            ids = cp.nonzero(level == l)[0]
            if ids.size:
                out[ids] = pyr[l][y[ids] >> l, x[ids] >> l].astype(dtype)
        return out

    def check(self, tree, arr, targets):
        super().check(tree, arr, targets)
        self.levels_of(tree)

    def apply(self, arr, step):
        old = arr[: step.n0]
        out = cp.empty(step.n1, dtype=arr.dtype)
        out[step.mapping[step.keep]] = old[step.keep]
        ids = step.new_ids
        out[ids] = self.sample(step.tree, step.level[ids], step.x[ids], step.y[ids], arr.dtype)
        arr[: step.n1] = out
        arr[step.n1 :] = 0


def water_fill(volume_per_child, beds):
    """Return the depths of the flat water surface over ``(4, m)`` child beds.

    ``volume_per_child`` is the parent depth, so the four depths sum to four
    times it: a dry child holds nothing and the others share the volume.
    """
    z = cp.sort(beds, axis=0)
    total = 4.0 * volume_per_child
    counts = cp.arange(1, 5, dtype=cp.float64).reshape(4, 1)
    surfaces = (total + cp.cumsum(z, axis=0)) / counts
    wet = (surfaces > z).sum(0)
    surface = cp.where(wet > 0, surfaces[cp.maximum(wet - 1, 0), cp.arange(beds.shape[1])], -cp.inf)
    return cp.maximum(surface - beds, 0.0)


class Depth(Transfer):
    """Water depth: keeps volume and a flat water surface inside each parent.

    Children share the parent's volume over their own bed; a lake at rest stays
    at rest when the bed obeys ``mean(children) == parent``, as ``FromRaster``
    with ``mean`` does. A parent takes the mean depth, which conserves volume.
    ``bed`` must also be listed in the same update.
    """

    order = 1

    def __init__(self, bed):
        self.bed = bed

    def check(self, tree, arr, targets):
        super().check(tree, arr, targets)
        from ._tree import _raw

        self._bed = _raw(self.bed)
        if self._bed.data.ptr not in targets:
            raise ValueError("Depth: its bed field must be transferred in the same update")

    def apply(self, arr, step):
        old = arr[: step.n0].copy()
        children = cp.zeros((4, step.n_split), dtype=cp.float64)
        if step.n_split:
            beds = self._bed[step.child_ids].reshape(4, -1).astype(cp.float64)
            children = water_fill(cp.maximum(old[step.split_ids], 0).astype(cp.float64), beds)
        parents = _siblings(old, step).mean(0)
        write_leaves(arr, step, children, parents)


class Velocity(Transfer):
    """Velocity component: with :class:`Depth`, keeps the momentum of a leaf.

    Children take the velocity of their parent, so their momentum sums to the
    parent's whatever depths they get. A parent takes the depth-weighted mean of
    its children, zero when they are all dry. ``depth`` must be listed in the
    same update, with :class:`Depth`, which runs after this rule.
    """

    def __init__(self, depth):
        self.depth = depth

    def check(self, tree, arr, targets):
        super().check(tree, arr, targets)
        from ._tree import _raw

        self._depth = _raw(self.depth)
        if self._depth.data.ptr not in targets:
            raise ValueError("Velocity: its depth field must be transferred in the same update")

    def children(self, old, step):
        return cp.tile(old[step.split_ids], (4, 1))

    def parents(self, old, step):
        depth = _siblings(self._depth[: step.n0], step).astype(cp.float64)
        total = depth.sum(0)
        moment = (depth * _siblings(old, step)).sum(0)
        return cp.where(total > 0, moment / cp.where(total > 0, total, 1.0), 0.0)


class FaceDischarge(Transfer):
    """Discharge per directed link, in volume per time through the whole face.

    The field has ``adj_capacity`` entries, ordered like ``ADJ``. A new face
    takes its share of each old face it overlaps, proportional to length; a
    face inside a former leaf starts at zero; sibling faces sum. Divergence is
    kept on every kept leaf and in total over each changed area. D4 only: a D8
    corner link has no face to split.
    """

    per_link = True

    def check(self, tree, arr, targets):
        super().check(tree, arr, targets)
        if tree.n_neighbours != 8:
            raise ValueError("FaceDischarge: D4 only, a corner link has no face to split")

    def apply(self, arr, step):
        old = arr[: step.n_links0].copy()
        out = cp.empty(step.n_links1, dtype=arr.dtype)
        out[step.same_new] = old[step.same_old]
        out[step.dirty_pos] = self._dirty_links(old, step).astype(arr.dtype)
        arr[: step.n_links1] = out
        arr[step.n_links1 :] = 0

    @staticmethod
    def _dirty_links(old, step):
        a, b, k = step.dirty_rows, step.dirty_nbr, step.dirty_k
        face = k // 2
        vertical = (face == 1) | (face == 2)
        sa, sb = side(step.level[a]), side(step.level[b])
        f = cp.minimum(sa, sb)
        xa, ya, xb, yb = step.x[a], step.y[a], step.x[b], step.y[b]
        start = cp.where(vertical, cp.maximum(ya, yb), cp.maximum(xa, xb))
        high = (face == 2) | (face == 3)
        inner = cp.where(vertical, xa, ya) + cp.where(high, sa - 1, 0)
        outer = inner + cp.where(high, 1, -1)
        dom = step.dom
        extent = cp.where(vertical, dom.nx, dom.ny)
        periodic = cp.where(vertical, dom.periodic_x, dom.periodic_y)
        outer = cp.where(periodic, outer % extent, outer)
        total = cp.zeros(a.size, dtype=cp.float64)
        for half in (0, 1):
            along = start + half * (f >> 1)
            share = cp.where(f > 1, f // 2, half == 0).astype(cp.float64)
            px, py = cp.where(vertical, inner, along), cp.where(vertical, along, inner)
            qx, qy = cp.where(vertical, outer, along), cp.where(vertical, along, outer)
            A = owner(step.old_key, px, py)
            B = owner(step.old_key, qx, qy)
            old_face = cp.minimum(side(step.old_level[A]), side(step.old_level[B])).astype(cp.float64)
            found = cp.zeros(a.size, dtype=cp.float64)
            begin, end = step.ptr0[A], step.ptr0[A + 1]
            for t in range(int(step.dom.n_slots)):
                pos = begin + t
                ok = pos < end
                safe = cp.where(ok, pos, 0)
                hit = ok & (step.adj0[safe] == B) & (A != B)
                found = cp.where(hit, old[safe].astype(cp.float64), found)
            total += found * share / old_face
        return total


_NAMED = {"mean": Mean, "sum": Sum, "min": Min, "max": Max, "linear": Linear}


def resolve(rule):
    """Return a rule object from a rule or a name."""
    if isinstance(rule, Transfer):
        return rule
    if rule in _NAMED:
        return _NAMED[rule]()
    raise ValueError(f"unknown field transfer {rule!r}; expected a Transfer or one of {sorted(_NAMED)}")
