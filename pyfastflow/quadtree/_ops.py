"""GPU array operations building and updating a Morton-sorted leaf list.

A leaf is ``(level, x, y)``: ``x, y`` is its low corner in finest-cell units,
``1 << level`` its side. Leaves are kept sorted by the Morton key of ``(x, y)``,
so the leaf holding a point is one ``searchsorted`` away.

Neighbour slots, with ``reverse(k) = n_slots - 1 - k`` on every link:

- D4, 8 slots: N 0 1, W 2 3, E 4 5, S 6 7.
- D8, 12 slots: NW 0, N 1 2, NE 3, W 4 5, E 6 7, SW 8, S 9 10, SE 11.

A face holds up to two neighbours. On the N and W faces the first probe along
the face takes the first slot; on the E and S faces it takes the second. A
corner slot holds a leaf only when that leaf shares exactly this corner vertex.

Author: B.G (11/2026)
"""

from dataclasses import dataclass

import cupy as cp

OUT_BIT = 1
NODATA_BIT = 2
MAX_EXTENT = 65536


@dataclass(frozen=True)
class Domain:
    """Fixed geometry and connectivity of one tree."""

    nx: int
    ny: int
    levels: int
    d8: bool
    periodic_x: bool
    periodic_y: bool

    @property
    def n_slots(self) -> int:
        return 12 if self.d8 else 8


def _spread(v):
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    return (v | (v << 1)) & 0x55555555


def morton(x, y):
    """Return the u32 Morton key of integer coordinates."""
    return _spread(x.astype(cp.uint32)) | (_spread(y.astype(cp.uint32)) << 1)


def side(level):
    """Return the leaf side in finest-cell units."""
    return cp.left_shift(cp.ones_like(level), level)


def owner(keys, px, py):
    """Return the sorted position of the leaf holding each in-domain point."""
    return cp.searchsorted(keys, morton(px, py), side="right") - 1


def clip_probe(dom, px, py):
    """Wrap periodic probes; return ``(kept positions, px, py)`` of in-domain ones."""
    ok = cp.ones_like(px, dtype=bool)
    if dom.periodic_x:
        px = px % dom.nx
    else:
        ok &= (px >= 0) & (px < dom.nx)
    if dom.periodic_y:
        py = py % dom.ny
    else:
        ok &= (py >= 0) & (py < dom.ny)
    kept = cp.nonzero(ok)[0]
    return kept, px[kept], py[kept]


def _pool2(a, op):
    h, w = a.shape
    return getattr(a.reshape(h // 2, 2, w // 2, 2), op)((1, 3))


def pad_to_levels(a, dom, fill):
    """Pad a raster on its high edges to a multiple of the coarsest side."""
    s = 1 << dom.levels
    h = (dom.ny + s - 1) // s * s
    w = (dom.nx + s - 1) // s * s
    if (h, w) == (dom.ny, dom.nx):
        return a
    out = cp.full((h, w), fill, dtype=a.dtype)
    out[: dom.ny, : dom.nx] = a
    return out


def pyramid(a, op, levels):
    """Return ``levels + 1`` rasters pooled by ``op`` (``min``, ``max``, ``mean``, ``sum``, ``any``, ``all``)."""
    out = [a]
    for _ in range(levels):
        out.append(_pool2(out[-1], op))
    return out


def _dilate(a, dom):
    d = a.copy()
    d[1:] |= a[:-1]
    d[:-1] |= a[1:]
    if dom.periodic_y:
        d[0] |= a[-1]
        d[-1] |= a[0]
    e = d.copy()
    e[:, 1:] |= d[:, :-1]
    e[:, :-1] |= d[:, 1:]
    if dom.periodic_x:
        e[:, 0] |= d[:, -1]
        e[:, -1] |= d[:, 0]
    return e


def leaves_from_level_map(level_map, dom):
    """Return the 2:1-balanced leaves ``(level, x, y, key)`` honouring a level map.

    A leaf is refined until its level is at most the minimum of ``level_map``
    over its footprint, then as far as 2:1 balance across faces and corners
    requires. Leaves overlapping the area outside the domain are refined until
    they lie inside it.
    """
    L = dom.levels
    required = cp.clip(cp.asarray(level_map, dtype=cp.int32), 0, L)
    required = pad_to_levels(required, dom, -1)
    mins = pyramid(required, "min", L)
    maxs = pyramid(required, "max", L)
    outside = [m < 0 for m in maxs]
    refine = [cp.zeros_like(outside[0])] + [(mins[l] < l) & ~outside[l] for l in range(1, L + 1)]
    for l in range(1, L):
        refine[l + 1] |= _pool2(_dilate(refine[l], dom), "any") & ~outside[l + 1]

    levels, xs, ys = [], [], []
    for l in range(L + 1):
        present = cp.ones_like(refine[L]) if l == L else cp.repeat(cp.repeat(refine[l + 1], 2, 0), 2, 1)
        y, x = cp.nonzero(present & ~refine[l] & ~outside[l])
        levels.append(cp.full(y.size, l, dtype=cp.int32))
        xs.append(x.astype(cp.int32) << l)
        ys.append(y.astype(cp.int32) << l)
    level, x, y = cp.concatenate(levels), cp.concatenate(xs), cp.concatenate(ys)
    key = morton(x, y)
    order = cp.argsort(key)
    return level[order], x[order], y[order], key[order]


def neighbour_rows(dom, level, x, y, keys, ids):
    """Return the ``(ids.size, n_slots)`` i32 neighbour table of leaves ``ids``."""
    li, xi, yi = level[ids], x[ids], y[ids]
    sz = side(li)
    half = sz >> 1
    table = cp.full((ids.size, dom.n_slots), -1, dtype=cp.int32)
    first = (1, 4, 6, 9) if dom.d8 else (0, 2, 4, 6)
    faces = (
        (xi, yi - 1, xi + half, yi - 1, xi),
        (xi - 1, yi, xi - 1, yi + half, yi),
        (xi + sz, yi, xi + sz, yi + half, yi),
        (xi, yi + sz, xi + half, yi + sz, xi),
    )
    for f, (ax, ay, bx, by, along) in enumerate(faces):
        kept, ax, ay = clip_probe(dom, ax, ay)
        if kept.size == 0:
            continue
        _, bx, by = clip_probe(dom, bx, by)
        q0 = owner(keys, ax, ay)
        lq = level[q0]
        a = li[kept]
        low = f < 2
        s0 = first[f] + (0 if low else 1)
        s1 = first[f] + (1 if low else 0)

        same = cp.nonzero(lq == a)[0]
        table[kept[same], s0] = q0[same].astype(cp.int32)

        fine = cp.nonzero(lq < a)[0]
        if fine.size:
            table[kept[fine], s0] = q0[fine].astype(cp.int32)
            table[kept[fine], s1] = owner(keys, bx[fine], by[fine]).astype(cp.int32)

        coarse = cp.nonzero(lq > a)[0]
        if coarse.size:
            second = (along[kept[coarse]] >> a[coarse]) & 1
            slot = first[f] + (second if low else 1 - second)
            table[kept[coarse], slot] = q0[coarse].astype(cp.int32)

    if dom.d8:
        def wrapped(px, py):
            if dom.periodic_x:
                px = px % dom.nx
            if dom.periodic_y:
                py = py % dom.ny
            return px, py

        corners = (
            (0, xi - 1, yi - 1, True, True),
            (3, xi + sz, yi - 1, False, True),
            (8, xi - 1, yi + sz, True, False),
            (11, xi + sz, yi + sz, False, False),
        )
        for slot, px, py, west, north in corners:
            kept, px, py = clip_probe(dom, px, py)
            if kept.size == 0:
                continue
            q = owner(keys, px, py)
            sq = side(level[q])
            vx = x[q] + sq if west else x[q]
            vy = y[q] + sq if north else y[q]
            tx = xi[kept] if west else xi[kept] + sz[kept]
            ty = yi[kept] if north else yi[kept] + sz[kept]
            vx, vy = wrapped(vx, vy)
            tx, ty = wrapped(tx, ty)
            hit = cp.nonzero((vx == tx) & (vy == ty))[0]
            table[kept[hit], slot] = q[hit].astype(cp.int32)
    return table


def slot_mask(table, dtype):
    """Return the bitmask of filled slots per row."""
    bits = cp.left_shift(cp.ones(table.shape[1], dtype=cp.int32), cp.arange(table.shape[1], dtype=cp.int32))
    return ((table >= 0).astype(cp.int32) * bits).sum(1).astype(dtype)


def csr_from_table(table):
    """Return ``(ptr, adj)`` with each row's neighbours in ascending slot order."""
    filled = table >= 0
    ptr = cp.zeros(table.shape[0] + 1, dtype=cp.int32)
    ptr[1:] = cp.cumsum(filled.sum(1))
    return ptr, table[filled]


_RIPPLE = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))


def ripple_refinement(dom, level, x, y, keys, seed):
    """Return the mask of leaves to split so that splitting ``seed`` keeps 2:1 balance."""
    mark = cp.zeros(level.size, dtype=bool)
    mark[seed] = True
    frontier = seed
    while frontier.size:
        lf, xf, yf = level[frontier], x[frontier], y[frontier]
        sz = side(lf)
        found = []
        for ox, oy in _RIPPLE:
            px = xf - 1 if ox < 0 else (xf + sz if ox > 0 else xf)
            py = yf - 1 if oy < 0 else (yf + sz if oy > 0 else yf)
            kept, px, py = clip_probe(dom, px, py)
            if kept.size == 0:
                continue
            q = owner(keys, px, py)
            found.append(q[level[q] > lf[kept]])
        cand = cp.concatenate(found) if found else cp.zeros(0, dtype=cp.int64)
        cand = cp.unique(cand[~mark[cand]]) if cand.size else cand
        if cand.size:
            mark[cand] = True
        frontier = cand
    return mark


def sibling_groups(dom, level, keys, wanted, split):
    """Return first-child positions of complete sibling groups asking to merge.

    The four siblings are consecutive leaves of one level below the coarsest,
    all ``wanted`` and none in ``split``. :func:`blocked_groups` applies the
    balance rule on top of this.
    """
    n = level.size
    if n < 4:
        return cp.zeros(0, dtype=cp.int64)
    wide = keys.astype(cp.int64)
    width = cp.left_shift(cp.ones_like(wide), 2 * (level.astype(cp.int64) + 1))
    aligned = (wide & (width - 1)) == 0
    start = cp.nonzero(aligned[: n - 3] & wanted[: n - 3] & ~split[: n - 3] & (level[: n - 3] < dom.levels))[0]
    if start.size == 0:
        return start
    lv = level[start]
    ok = (level[start + 1] == lv) & (level[start + 2] == lv) & (level[start + 3] == lv)
    for d in (1, 2, 3):
        ok &= wanted[start + d] & ~split[start + d]
    span = cp.left_shift(cp.ones_like(wide[start]), 2 * lv.astype(cp.int64))
    ok &= wide[start + 3] == wide[start] + 3 * span
    return start[ok]


def blocked_groups(dom, level, x, y, keys, split, start):
    """Return a mask of groups whose parent would touch a leaf two levels finer."""
    blocked = cp.zeros(start.size, dtype=bool)
    lv, x0, y0 = level[start], x[start], y[start]
    s = side(lv)
    h = s >> 1
    P = 2 * s
    probes = []
    for t in (cp.zeros_like(s), h, s, s + h):
        probes += [(x0 + t, y0 - 1), (x0 + t, y0 + P), (x0 - 1, y0 + t), (x0 + P, y0 + t)]
    probes += [(x0 - 1, y0 - 1), (x0 + P, y0 - 1), (x0 - 1, y0 + P), (x0 + P, y0 + P)]
    for px, py in probes:
        kept, px, py = clip_probe(dom, px, py)
        if kept.size == 0:
            continue
        q = owner(keys, px, py)
        final = level[q] - split[q].astype(level.dtype)
        blocked[kept[final < lv[kept]]] = True
    return blocked
