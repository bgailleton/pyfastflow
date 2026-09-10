"""Taichi and Quadrants templates for fill-and-reconstruct routing."""

from ..core import KernelBuilder
from ._closure_shared import _tensor_annotation

_POS_SENTINEL = 1.0e9


def build_fill_reconstruct_init(*, backend: str, backend_mod, grid):
    """
    init_filled KernelBuilder, data args (z, filled, parent): on a can_out
    node, filled[i] = z[i] and parent[i] = i (self-receiving, the base-level
    convention); elsewhere filled[i] = +inf sentinel, parent[i] = -1 (never
    yet claimed) - the seed state every sweep/relax pass decreases from.
    Composes its own `grid` occurrence.

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

    """
    T = _tensor_annotation(backend_mod, backend)

    def init_filled_tmpl(ctx, z: T, filled: T, parent: T):
        for i in z:
            if ctx.grid.can_out(i):
                filled[i] = z[i]
                parent[i] = i
            else:
                filled[i] = _POS_SENTINEL
                parent[i] = -1

    return KernelBuilder(init_filled_tmpl).compose("grid", grid).freeze()


def build_fill_reconstruct_sweeps(*, backend: str, backend_mod, nx: int, ny: int):
    """
    Four KernelBuilders, each data args (z, filled, parent) - one raster
    sweep per direction (row left-to-right, row right-to-left, column
    top-to-bottom, column bottom-to-top), one thread per row/column walking
    it serially - no atomics needed, since distinct rows/columns never touch
    the same cell. Keyed "row_lr", "row_rl", "col_tb", "col_bt".

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    nx, ny : int

    Returns
    -------
    dict
        {"row_lr": ..., "row_rl": ..., "col_tb": ..., "col_bt": ...}, all
        KernelBuilders.

    """
    T = _tensor_annotation(backend_mod, backend)
    NX = nx
    NY = ny

    def sweep_row_lr_tmpl(ctx, z: T, filled: T, parent: T):
        for r in range(NY):
            base = r * NX
            for c in range(1, NX):
                i = base + c
                left = i - 1
                cand = z[i] if z[i] > filled[left] else filled[left]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = left

    def sweep_row_rl_tmpl(ctx, z: T, filled: T, parent: T):
        for r in range(NY):
            base = r * NX
            for cc in range(NX - 1):
                c = NX - 2 - cc
                i = base + c
                right = i + 1
                cand = z[i] if z[i] > filled[right] else filled[right]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = right

    def sweep_col_tb_tmpl(ctx, z: T, filled: T, parent: T):
        for c in range(NX):
            for r in range(1, NY):
                i = r * NX + c
                up = i - NX
                cand = z[i] if z[i] > filled[up] else filled[up]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = up

    def sweep_col_bt_tmpl(ctx, z: T, filled: T, parent: T):
        for c in range(NX):
            for rr in range(NY - 1):
                r = NY - 2 - rr
                i = r * NX + c
                down = i + NX
                cand = z[i] if z[i] > filled[down] else filled[down]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = down

    def _kb(tmpl):
        return KernelBuilder(tmpl).freeze()

    return {
        "row_lr": _kb(sweep_row_lr_tmpl),
        "row_rl": _kb(sweep_row_rl_tmpl),
        "col_tb": _kb(sweep_col_tb_tmpl),
        "col_bt": _kb(sweep_col_bt_tmpl),
    }


def build_fill_reconstruct_frontier_init(*, backend: str, backend_mod):
    """
    frontier_init KernelBuilder, data args (z, filled, frontier, counters):
    every cell not yet sealed after the sweeps (filled[i] > z[i]) is pushed
    into `frontier`'s first half (indices [0, n_flat)) and counted into
    `counters[0]` - the seed frontier the relax loop's pass 0 reads.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.

    Returns
    -------
    KernelBuilder

    """
    T = _tensor_annotation(backend_mod, backend)

    def frontier_init_tmpl(ctx, z: T, filled: T, frontier: T, counters: T):
        for i in z:
            if filled[i] > z[i]:
                pos = ctx.bk.atomic_add(counters[0], 1)
                frontier[pos] = i

    return KernelBuilder(frontier_init_tmpl).freeze()


def build_fill_reconstruct_relax(*, backend: str, backend_mod, grid, n_flat: int):
    """
    relax KernelBuilder, data args (z, filled, parent, frontier, counters,
    queued_gen): one pass over the `counters[ctx.P.get(0)]`-sized input half
    of `frontier`, relaxing each active cell against its neighbours and
    pushing any neighbour whose candidate could still improve into the
    output half, deduplicated per pass via ``queued_gen`` and ``atomic_max``.
    The gate may schedule unnecessary work, but never omits a potential
    improvement.

    `P` is this kernel's own wired PARAM slot (mode "scalar" - the host
    bumps it between passes); composes its own `grid` occurrence.

    `active` is the raw backing field of a caller's scalar Parameter
    (`active_p.handle().array`, same "concurrently mutated is DATA by
    definition" classification as `counters`/`queued_gen` - see
    _closure_depressions.py's `build_depression_counter` for the identical
    pattern with `ndep`) - every push into the output frontier half also
    atomic-adds 1 into it, so a host block can read it back after this
    kernel returns to know whether the next pass has any work
    (make_fill_reconstruct_solver's early-stop `until`). The caller must
    reset it to 0 (`.set(0)`) before each launch, same as `ndep`.

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

    """
    T = _tensor_annotation(backend_mod, backend)
    NFLAT = n_flat

    def relax_tmpl(ctx, z: T, filled: T, parent: T, frontier: T, counters: T, queued_gen: T, active: T):
        p = ctx.P.get(0)
        par = p % 2
        in_base = par * NFLAT
        out_base = (1 - par) * NFLAT
        count = counters[p]
        for idx in range(count):
            i = frontier[in_base + idx]
            nk = ctx.grid.N_NEIGHBOURS.get(0)

            best = _POS_SENTINEL
            best_j = -1
            for k in range(nk):
                j = ctx.grid.neighbour(i, k)
                if j != -1:
                    v = filled[j]
                    if v < best:
                        best = v
                        best_j = j
            candidate = z[i] if z[i] > best else best

            if candidate < filled[i]:
                filled[i] = candidate
                parent[i] = best_j
                for k in range(nk):
                    j = ctx.grid.neighbour(i, k)
                    if j != -1:
                        cand_j = z[j] if z[j] > candidate else candidate
                        if cand_j < filled[j]:
                            old = ctx.bk.atomic_max(queued_gen[j], p)
                            if old != p:
                                pos = ctx.bk.atomic_add(counters[p + 1], 1)
                                frontier[out_base + pos] = j
                                ctx.bk.atomic_add(active[None], 1)

    return KernelBuilder(relax_tmpl).compose("grid", grid).freeze()
