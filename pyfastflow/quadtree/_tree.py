"""Host-side owners of a cell-level Morton CSR quadtree.

Author: B.G (11/2026)
"""

import cupy as cp
import numpy as np

from ..core import require_backend
from ..core.context.program import BorrowedParams
from types import SimpleNamespace

from . import transfer as _transfer
from ._ops import (
    MAX_EXTENT,
    NODATA_BIT,
    OUT_BIT,
    Domain,
    blocked_groups,
    csr_from_table,
    leaves_from_level_map,
    morton,
    neighbour_rows,
    owner,
    pad_to_levels,
    pyramid,
    ripple_refinement,
    sibling_groups,
    side,
    slot_mask,
)

_BOUNDARIES = frozenset({"normal", "periodic_EW", "periodic_NS", "periodic_EW_NS"})
_SLOTS = {"D4": 8, "D8": 12}


def _raw(field):
    """Return the flat device array behind a cupy array, DataHandle or accessor."""
    arr = getattr(field, "array", field)
    return arr.reshape(-1)


class Quadtree:
    """Static cell-level quadtree: one Morton-sorted leaf per entry, CSR links.

    Parameters
    ----------
    be : Backend
        Target backend; must be ``cupy``.
    pool : Pool
        Pool that owns the topology storage.
    level_map : array, shape (ny, nx)
        Finest-resolution raster of the coarsest level allowed per cell. Level
        0 is the finest; a level ``l`` leaf has side ``dx * 2**l``. Values are
        clipped to ``[0, levels]``. The tree is 2:1 balanced across faces and
        corners, so it may be finer than requested.
    dx : float
        Side of a level 0 leaf.
    levels : int
        Coarsest level.
    topology : {"D4", "D8"}, default="D8"
        Faces only (8 slots), or faces and corners (12 slots).
    boundary : {"normal", "periodic_EW", "periodic_NS", "periodic_EW_NS"}
        Periodic axes must be a multiple of ``2**levels``.
    nodata_mask, outlet_mask : array, shape (ny, nx), optional
        Non-zero marks a finest cell. A leaf is no-data when all its cells are,
        and an outlet when any of its cells is. Without ``outlet_mask``, leaves
        touching a non-periodic domain edge are outlets.
    dx_mode : {"const", "scalar"}, default="const"
        Storage of ``dx``.
    capacity, adj_capacity : int, optional
        Storage size in leaves and in links; default to the exact tree size.

    Attributes
    ----------
    group : FrozenGroup
        Device helpers; compose it into kernels as a grid.
    parameters : dict[str, Parameter]
        Parameters to bind to ``group``. The tree owns them: close every Program
        and kernel using them before :meth:`close`.
    n, capacity : int
        Current and maximum leaf count. Every per-cell field has ``capacity``
        entries. Entries from ``n`` on are isolated no-data outlets.
    """

    dynamic = False

    def __init__(self, be, pool, level_map, dx, levels, *, topology="D8", boundary="normal",
                 nodata_mask=None, outlet_mask=None, dx_mode="const",
                 capacity=None, adj_capacity=None):
        be = require_backend(be)
        if be.family != "cupy":
            raise ValueError(f"Quadtree: cupy only, got backend family {be.family!r}")
        if topology not in _SLOTS:
            raise ValueError(f"Quadtree: topology must be one of {sorted(_SLOTS)}, got {topology!r}")
        if boundary not in _BOUNDARIES:
            raise ValueError(f"Quadtree: boundary must be one of {sorted(_BOUNDARIES)}, got {boundary!r}")
        if dx_mode not in ("const", "scalar"):
            raise ValueError(f"Quadtree: dx_mode must be 'const' or 'scalar', got {dx_mode!r}")
        if level_map.ndim != 2:
            raise ValueError(f"Quadtree: level_map must be 2D, got shape {level_map.shape}")
        ny, nx = int(level_map.shape[0]), int(level_map.shape[1])
        if nx > MAX_EXTENT or ny > MAX_EXTENT:
            raise ValueError(f"Quadtree: raster extent is limited to {MAX_EXTENT}, got {ny}x{nx}")
        levels = int(levels)
        if levels < 0:
            raise ValueError(f"Quadtree: levels must be >= 0, got {levels}")
        periodic_x = boundary in ("periodic_EW", "periodic_EW_NS")
        periodic_y = boundary in ("periodic_NS", "periodic_EW_NS")
        if (periodic_x and nx % (1 << levels)) or (periodic_y and ny % (1 << levels)):
            raise ValueError(f"Quadtree: a periodic axis must be a multiple of {1 << levels}, got {ny}x{nx}")

        self._be, self._pool = be, pool
        self._dom = Domain(nx, ny, levels, topology == "D8", periodic_x, periodic_y)
        self.topology, self.boundary, self.dx0 = topology, boundary, float(dx)
        self._has_nodata = nodata_mask is not None
        self._has_outlet = outlet_mask is not None
        self._nodata_pyr = self._mask_pyramid(nodata_mask, "all", 1)
        self._outlet_pyr = self._mask_pyramid(outlet_mask, "any", 0)
        self._slot_dtype = "u32" if topology == "D8" else "u8"
        self._closed = False

        level, x, y, key = leaves_from_level_map(level_map, self._dom)
        ptr, adj, slots = self._full_topology(level, x, y, key)
        n = int(level.size)
        self.capacity = self._default_capacity(n) if capacity is None else int(capacity)
        self.adj_capacity = (self._default_adj_capacity(int(adj.size), n) if adj_capacity is None
                             else int(adj_capacity))
        self._check_room(n, int(adj.size))
        self._allocate(dx_mode, n)
        self._commit(level, x, y, key, ptr, adj, slots, self._cell_flags(level, x, y))

        from . import make_quadtree_group

        self.group = make_quadtree_group(be, topology=topology, nodata=self._has_nodata)

    def _default_capacity(self, n):
        return n

    def _default_adj_capacity(self, n_links, n):
        return n_links

    def _mask_pyramid(self, mask, op, fill):
        if mask is None:
            return None
        arr = cp.asarray(mask) != 0
        if arr.shape != (self._dom.ny, self._dom.nx):
            raise ValueError(f"Quadtree: mask shape {arr.shape} differs from ({self._dom.ny}, {self._dom.nx})")
        return pyramid(pad_to_levels(arr, self._dom, bool(fill)), op, self._dom.levels)

    def _allocate(self, dx_mode, n):
        cls = self._be.ParameterCls
        pool = self._pool
        cap = self.capacity

        def field(name, dtype, size, fill=0):
            value = np.full(size, fill, dtype=np.dtype({"i32": "int32", "u8": "uint8", "u32": "uint32"}[dtype]))
            return cls(f"TREE_{name}", dtype=dtype, mode="field", value=value, pool=pool, shape=(size,))

        self.parameters = BorrowedParams({
            "N": cls("TREE_N", dtype="i32", mode="scalar" if self.dynamic else "const", value=n, pool=pool),
            "CAPACITY": cls("TREE_CAPACITY", dtype="i32", mode="const", value=cap, pool=pool),
            "DX0": cls("TREE_DX0", dtype="f32", mode=dx_mode, value=self.dx0, pool=pool),
            "N_NEIGHBOURS": cls("TREE_NNEIGHBOURS", dtype="i32", mode="const", value=self._dom.n_slots, pool=pool),
            "LEVEL": field("LEVEL", "u8", cap),
            "FLAGS": field("FLAGS", "u8", cap, OUT_BIT | NODATA_BIT),
            "SLOTS": field("SLOTS", self._slot_dtype, cap),
            "PTR": field("PTR", "i32", cap + 1),
            "ADJ": field("ADJ", "i32", self.adj_capacity),
            "X": field("X", "i32", cap),
            "Y": field("Y", "i32", cap),
        })

    def _array(self, name):
        return self.parameters[name].handle().array

    def _check_room(self, n, n_links):
        if n > self.capacity:
            raise ValueError(f"Quadtree: {n} leaves exceed capacity {self.capacity}; tree left unchanged")
        if n_links > self.adj_capacity:
            raise ValueError(f"Quadtree: {n_links} links exceed adj_capacity {self.adj_capacity}; tree left unchanged")

    def _full_topology(self, level, x, y, key):
        table = neighbour_rows(self._dom, level, x, y, key, cp.arange(level.size))
        ptr, adj = csr_from_table(table)
        return ptr, adj, slot_mask(table, cp.uint32 if self._dom.d8 else cp.uint8)

    def _cell_flags(self, level, x, y):
        """Return outlet and no-data flags of the given leaves."""
        dom = self._dom
        flags = cp.zeros(level.size, dtype=cp.uint8)
        if self._outlet_pyr is None:
            sz = side(level)
            edge = cp.zeros(level.size, dtype=bool)
            if not dom.periodic_x:
                edge |= (x == 0) | (x + sz >= dom.nx)
            if not dom.periodic_y:
                edge |= (y == 0) | (y + sz >= dom.ny)
            flags[edge] |= OUT_BIT
        if self._outlet_pyr is not None or self._nodata_pyr is not None:
            for l in range(dom.levels + 1):
                ids = cp.nonzero(level == l)[0]
                if ids.size == 0:
                    continue
                by, bx = y[ids] >> l, x[ids] >> l
                if self._outlet_pyr is not None:
                    flags[ids[self._outlet_pyr[l][by, bx]]] |= OUT_BIT
                if self._nodata_pyr is not None:
                    flags[ids[self._nodata_pyr[l][by, bx]]] |= NODATA_BIT
        return flags

    def _commit(self, level, x, y, key, ptr, adj, slots, flags):
        """Write one tree into the fixed topology storage."""
        n, m = int(level.size), int(adj.size)
        self._array("LEVEL")[:n] = level.astype(cp.uint8)
        self._array("LEVEL")[n:] = 0
        self._array("FLAGS")[:n] = flags
        self._array("FLAGS")[n:] = OUT_BIT | NODATA_BIT
        self._array("SLOTS")[:n] = slots
        self._array("SLOTS")[n:] = 0
        self._array("X")[:n] = x
        self._array("Y")[:n] = y
        self._array("PTR")[: n + 1] = ptr
        self._array("PTR")[n + 1 :] = m
        self._array("ADJ")[:m] = adj
        self.n, self.n_links = n, m
        self._level, self._x, self._y, self._key = level, x, y, key
        self._flags, self._slots = flags, slots
        if self.dynamic:
            self.parameters["N"].set(n)

    @property
    def n_neighbours(self):
        """Neighbour slots per leaf."""
        return self._dom.n_slots

    @property
    def dims(self):
        """Program dimensions of a per-cell field: ``capacity`` entries on one row."""
        return {"nx": self.capacity, "ny": 1}

    @property
    def nx(self):
        """Finest raster width."""
        return self._dom.nx

    @property
    def ny(self):
        """Finest raster height."""
        return self._dom.ny

    @property
    def levels(self):
        """Coarsest level."""
        return self._dom.levels

    @property
    def level(self):
        """Level of each current leaf, as a device array."""
        return self._level

    @property
    def x(self):
        """Low-corner x of each current leaf, in finest-cell units."""
        return self._x

    @property
    def y(self):
        """Low-corner y of each current leaf, in finest-cell units."""
        return self._y

    def from_raster(self, raster, *, mode="mean", dtype=None):
        """Return a ``capacity``-long device field aggregating a finest raster.

        ``mode`` is the aggregation over each leaf: ``mean``, ``sum``, ``min``
        or ``max``. Entries from ``n`` on are zero.
        """
        if mode not in ("mean", "sum", "min", "max"):
            raise ValueError(f"from_raster: unknown mode {mode!r}")
        arr = cp.asarray(raster)
        if arr.shape != (self.ny, self.nx):
            raise ValueError(f"from_raster: shape {arr.shape} differs from ({self.ny}, {self.nx})")
        pyr = pyramid(pad_to_levels(arr, self._dom, 0), mode, self.levels)
        dtype = arr.dtype if dtype is None else dtype
        out = cp.zeros(self.capacity, dtype=dtype)
        for l in range(self.levels + 1):
            ids = cp.nonzero(self._level == l)[0]
            if ids.size:
                out[ids] = pyr[l][self._y[ids] >> l, self._x[ids] >> l].astype(dtype)
        return out

    def to_raster(self, field):
        """Return a field as a host array on the finest raster, piecewise constant."""
        arr = _raw(field)
        gy, gx = cp.meshgrid(cp.arange(self.ny, dtype=cp.int32), cp.arange(self.nx, dtype=cp.int32), indexing="ij")
        return cp.asnumpy(arr[owner(self._key, gx.ravel(), gy.ravel())].reshape(self.ny, self.nx))

    def adapt(self, flags, fields=None):
        """Refine and coarsen leaves; unavailable on a static tree."""
        raise NotImplementedError("Quadtree is static; use DynamicQuadtree to refine or coarsen")

    def regrid(self, level_map, fields=None):
        """Rebuild the tree from a level map; unavailable on a static tree."""
        raise NotImplementedError("Quadtree is static; use DynamicQuadtree to regrid")

    def close(self):
        """Release the topology storage."""
        if self._closed:
            return
        for p in self.parameters.values():
            p.destroy()
        self._closed = True


class DynamicQuadtree(Quadtree):
    """Quadtree whose leaves can be refined and coarsened in place.

    Topology Parameters keep their storage, so compiled kernels stay valid;
    updates re-sort the leaves and rewrite that storage. ``capacity`` defaults
    to twice the initial leaf count and ``adj_capacity`` to 6 (D4) or 10 (D8)
    links per leaf of capacity. An update exceeding either raises and leaves
    the tree and fields unchanged.
    """

    dynamic = True

    def _default_capacity(self, n):
        return 2 * n

    def _default_adj_capacity(self, n_links, n):
        return max(n_links, self.capacity * (10 if self._dom.d8 else 6))

    def _check_fields(self, fields):
        """Return ``(array, rule)`` pairs in application order, or raise before any change."""
        pairs = [(_raw(field), _transfer.resolve(rule)) for field, rule in (fields or ())]
        targets = {arr.data.ptr for arr, _ in pairs}
        for arr, rule in pairs:
            size = self.adj_capacity if rule.per_link else self.capacity
            if arr.size != size:
                raise ValueError(f"{type(rule).__name__}: field has {arr.size} entries, expected {size}")
            rule.check(self, arr, targets)
        return sorted(pairs, key=lambda pair: pair[1].order)

    def adapt(self, flags, fields=None):
        """Apply per-leaf flags and return the new leaf count.

        ``flags`` holds, per leaf, ``+1`` to refine, ``-1`` to coarsen and ``0``
        to keep; only its first ``n`` entries are read. Refining ripples to
        coarser neighbours as 2:1 balance requires. Four siblings merge only
        when all four are flagged and the merge keeps 2:1 balance; refinement
        wins over coarsening.

        ``fields`` is a sequence of ``(field, rule)`` pairs, each field rewritten
        in place. A rule is one from :mod:`pyfastflow.quadtree.transfer`, or the
        name ``mean``, ``sum``, ``min``, ``max`` or ``linear``.
        """
        fields = self._check_fields(fields)
        dom = self._dom
        n0 = self.n
        level, x, y, key = self._level, self._x, self._y, self._key
        f = cp.asarray(flags).reshape(-1)[:n0]
        if f.size != n0:
            raise ValueError(f"adapt: flags have {f.size} entries, expected at least {n0}")

        seed = cp.nonzero((f > 0) & (level > 0))[0]
        split = ripple_refinement(dom, level, x, y, key, seed)
        groups = sibling_groups(dom, level, key, f < 0, split)
        if groups.size:
            fine = cp.nonzero(level[groups] > 0)[0]
            if fine.size:
                blocked = blocked_groups(dom, level, x, y, key, split, groups[fine])
                keep_group = cp.ones(groups.size, dtype=bool)
                keep_group[fine[blocked]] = False
                groups = groups[keep_group]
        parents = groups.size
        split_ids = cp.nonzero(split)[0]
        if split_ids.size == 0 and parents == 0:
            return n0

        merged = cp.zeros(n0, dtype=bool)
        for d in range(4):
            merged[groups + d] = True
        removed = split | merged
        keep = cp.nonzero(~removed)[0]

        lc = level[split_ids] - 1
        h = side(lc)
        xs, ys = x[split_ids], y[split_ids]
        new_level = cp.concatenate([level[keep], lc, lc, lc, lc, level[groups] + 1])
        new_x = cp.concatenate([x[keep], xs, xs + h, xs, xs + h, x[groups]])
        new_y = cp.concatenate([y[keep], ys, ys, ys + h, ys + h, y[groups]])
        n1 = int(new_level.size)
        if n1 > self.capacity:
            self._check_room(n1, 0)

        new_key = morton(new_x, new_y)
        order = cp.argsort(new_key)
        new_level, new_x, new_y, new_key = new_level[order], new_x[order], new_y[order], new_key[order]
        inverse = cp.empty(n1, dtype=cp.int32)
        inverse[order] = cp.arange(n1, dtype=cp.int32)

        nk = int(keep.size)
        mapping = cp.full(n0, -1, dtype=cp.int32)
        mapping[keep] = inverse[:nk]

        # Rows that change: new leaves and kept neighbours of removed leaves.
        ptr0 = self._array("PTR")[: n0 + 1]
        adj0 = self._array("ADJ")[: self.n_links]
        count0 = ptr0[1:] - ptr0[:-1]
        row0 = cp.repeat(cp.arange(n0, dtype=cp.int32), count0)
        touched = cp.zeros(n0, dtype=bool)
        touched[adj0[removed[row0]]] = True
        touched[removed] = False
        dirty = cp.concatenate([inverse[nk:], mapping[cp.nonzero(touched)[0]]])
        table = neighbour_rows(dom, new_level, new_x, new_y, new_key, dirty)
        filled = table >= 0
        row_count = filled.sum(1).astype(cp.int32)

        same = cp.nonzero(~(removed | touched))[0]
        count = cp.zeros(n1, dtype=cp.int32)
        count[mapping[same]] = count0[same]
        count[dirty] = row_count
        slots_dtype = self._slots.dtype
        new_slots = cp.empty(n1, dtype=slots_dtype)
        new_slots[mapping[same]] = self._slots[same]
        new_slots[dirty] = slot_mask(table, slots_dtype)

        new_ptr = cp.zeros(n1 + 1, dtype=cp.int32)
        new_ptr[1:] = cp.cumsum(count)
        n_links = int(new_ptr[-1])
        self._check_room(n1, n_links)

        new_adj = cp.empty(n_links, dtype=cp.int32)
        sel = cp.nonzero(~(removed | touched)[row0])[0]
        r = row0[sel]
        same_new = new_ptr[mapping[r]] + (sel - ptr0[r])
        new_adj[same_new] = mapping[adj0[sel]]
        row_sel = cp.repeat(cp.arange(dirty.size, dtype=cp.int32), row_count)
        first = cp.zeros(dirty.size, dtype=cp.int32)
        first[1:] = cp.cumsum(row_count)[:-1]
        dirty_pos = new_ptr[dirty[row_sel]] + (cp.arange(row_sel.size, dtype=cp.int32) - first[row_sel])
        new_adj[dirty_pos] = table[filled]

        new_flags = cp.empty(n1, dtype=cp.uint8)
        new_flags[mapping[keep]] = self._flags[keep]
        new_ids = inverse[nk:]
        new_flags[new_ids] = self._cell_flags(new_level[new_ids], new_x[new_ids], new_y[new_ids])

        m = int(split_ids.size)
        filled_rows, filled_slots = cp.nonzero(filled)
        step = SimpleNamespace(
            tree=self, dom=dom, n0=n0, n1=n1, n_split=m, keep=keep, mapping=mapping,
            split_ids=split_ids, groups=groups, new_ids=new_ids,
            child_ids=inverse[nk : nk + 4 * m], parent_ids=inverse[nk + 4 * m :],
            old_level=level, old_x=x, old_y=y, old_key=key,
            level=new_level, x=new_x, y=new_y,
            ptr0=ptr0, adj0=adj0, n_links0=self.n_links, n_links1=n_links,
            same_old=sel, same_new=same_new,
            dirty_rows=dirty[filled_rows], dirty_k=filled_slots.astype(cp.int32),
            dirty_nbr=table[filled], dirty_pos=dirty_pos,
        )
        for arr, rule in fields:
            rule.apply(arr, step)

        self._commit(new_level, new_x, new_y, new_key, new_ptr, new_adj, new_slots, new_flags)
        return n1

    def regrid(self, level_map, fields=None):
        """Replace the whole tree from a level map and return the new leaf count.

        ``fields`` is as in :meth:`adapt`, limited to ``Mean``, ``Sum`` and
        ``FromRaster``. Values move between any two trees with area-weighted
        means or conserved totals.
        """
        fields = self._check_fields(fields)
        for _, rule in fields:
            if not rule.regrid_ok:
                raise NotImplementedError(f"regrid: {type(rule).__name__} works with adapt only")
        if tuple(level_map.shape) != (self.ny, self.nx):
            raise ValueError(f"regrid: level_map shape {tuple(level_map.shape)} differs from ({self.ny}, {self.nx})")
        level, x, y, key = leaves_from_level_map(level_map, self._dom)
        ptr, adj, slots = self._full_topology(level, x, y, key)
        n1 = int(level.size)
        self._check_room(n1, int(adj.size))

        if fields:
            old_level = self._level
            n0 = self.n
            old_key = self._key.astype(cp.int64)
            new_key = key.astype(cp.int64)
            span = cp.left_shift(cp.ones_like(new_key), 2 * level.astype(cp.int64))
            lo = cp.searchsorted(old_key, new_key, side="right") - 1
            hi = cp.searchsorted(old_key, new_key + span, side="left")
            old_cells = cp.left_shift(cp.ones(n0, dtype=cp.int64), 2 * old_level.astype(cp.int64)).astype(cp.float64)
            new_cells = span.astype(cp.float64)
            coarser_old = old_level[lo] > level
            for arr, rule in fields:
                if isinstance(rule, _transfer.FromRaster):
                    arr[:n1] = rule.sample(self, level, x, y, arr.dtype)
                    arr[n1:] = 0
                    continue
                conserve_mean = isinstance(rule, _transfer.Mean)
                old = arr[:n0].astype(cp.float64)
                mass = old * old_cells if conserve_mean else old
                prefix = cp.zeros(n0 + 1, dtype=cp.float64)
                prefix[1:] = cp.cumsum(mass)
                gathered = prefix[hi] - prefix[lo]
                if conserve_mean:
                    out = cp.where(coarser_old, old[lo], gathered / new_cells)
                else:
                    out = cp.where(coarser_old, old[lo] * new_cells / old_cells[lo], gathered)
                arr[:n1] = out.astype(arr.dtype)
                arr[n1:] = 0

        self._commit(level, x, y, key, ptr, adj, slots, self._cell_flags(level, x, y))
        return n1
