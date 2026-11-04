"""Cell-level quadtree grids for CuPy, usable by kernels written against a grid.

:class:`Quadtree` is static; :class:`DynamicQuadtree` adds refinement and
coarsening behind the same interface. Compose ``tree.group`` into a feature and
bind ``tree.parameters``, as with :func:`pyfastflow.grid.make_grid_group`.

Author: B.G (11/2026)
"""

from ..core import Backend, FrozenGroup, GroupBuilder, require_backend, share_leaf

_LEAVES = ("N", "CAPACITY", "DX0", "N_NEIGHBOURS", "LEVEL", "FLAGS", "PTR", "ADJ", "SLOTS", "X", "Y")


def make_quadtree_group(be: Backend, *, topology: str = "D8", nodata: bool = False) -> FrozenGroup:
    """Return reusable quadtree helpers.

    Parameters
    ----------
    be : Backend
        Target backend; must be ``cupy``.
    topology : {"D4", "D8"}, default="D8"
        8 face slots, or 8 face and 4 corner slots (12).
    nodata : bool, default=False
        Make ``neighbour`` return -1 for links touching a no-data leaf.

    Returns
    -------
    FrozenGroup
        Helpers with ``N``, ``CAPACITY``, ``DX0``, ``N_NEIGHBOURS``, ``LEVEL``,
        ``FLAGS``, ``PTR``, ``ADJ``, ``SLOTS``, ``X`` and ``Y`` parameter slots.

        Shared with the regular grid: ``n()``, the index-space size and length
        of every per-cell field; ``neighbour(i, k)`` for
        ``k < N_NEIGHBOURS``, -1 on an empty slot; ``neighbour_raw(i, k)``, the
        same without the no-data test; ``reverse(k)``, the slot of i seen from
        neighbour(i, k); ``dx(i)``; ``area(i)``; ``dist(i, k)``, centre
        distance; ``width(i, k)``, link width for partitioning flow: the shorter
        side of the two leaves, times sqrt(2) on a corner slot;
        ``flow_width(i, k)``, width of the flow leaving i, for a depth closure:
        its own side, times sqrt(2) on a corner slot; ``dist_between_nodes(i, j)``, -1 unless
        linked; ``neighbour_and_distance(i, k, &j, &d)``; ``can_out(i)``;
        ``is_active(i)``; ``nodata(i)``.

        Tree only: ``n_leaves()``, the current leaf count, at most ``n()``;
        entries from there on are isolated no-data outlets, which a kernel may
        skip or process like any cell; ``level(i)``; ``cx(i)`` and ``cy(i)``, leaf centre;
        ``nbr_begin(i)``, ``nbr_end(i)`` and ``nbr_at(p)``, the direct link
        loop, which also visits no-data leaves.

        ``DX``, ``NX``, ``NY``, ``delta``, ``face``, ``is_on_edge`` and
        ``which_edge`` do not exist here: kernels relying on the raster fail
        at compile.
    """
    be = require_backend(be)
    if be.family != "cupy":
        raise ValueError(f"make_quadtree_group: cupy only, got backend family {be.family!r}")
    if topology not in ("D4", "D8"):
        raise ValueError(f"make_quadtree_group: topology must be 'D4' or 'D8', got {topology!r}")
    from . import _cupy_blocks as blocks

    group = GroupBuilder()
    for leaf in _LEAVES:
        group.param(leaf)
    blocks.build_group(group, topology=topology, nodata=nodata)
    for leaf in _LEAVES:
        share_leaf(group, leaf)
    return group.freeze()


from . import transfer
from ._tree import DynamicQuadtree, Quadtree

__all__ = ["make_quadtree_group", "Quadtree", "DynamicQuadtree", "transfer"]
