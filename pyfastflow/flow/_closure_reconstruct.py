"""
Taichi/Quadrants (closure) block templates behind make_fill_reconstruct/
make_fill_reconstruct_solver, on the builder/frozen/bound stack (../core/
context/builder.py, frozen.py, bound.py) - see _cupy_reconstruct.py's own
module docstring for the algorithm and for why frontier_a/
frontier_b become one combined (2*n_flat,) buffer here, addressed by a
`p % 2` parity computed from the bound `P` Parameter - identical reasoning on
this backend, since a compiled Sequence step's data is bound once, at compile
time, on every backend, not just cupy.

See _closure_receivers.py/_closure_accum.py/_closure_depressions.py for
the other flow algorithms.

`P` is a plain PARAM slot (any mode) on `relax` - a
caller binds a Parameter there (mode "scalar", since the host bumps it
between passes) after `.build()`, exactly like make_accumulation's `ITER`;
there is no Need indirection anywhere in this stack. `grid` is the caller's
FrozenGroup (../grid's make_grid_group result), composed under "grid" -
`relax` reaches `ctx.grid.can_out`/`.neighbour`/`.N_NEIGHBOURS.get(0)`,
`init_filled` reaches `ctx.grid.can_out` only - each its own independent
occurrence (see _closure_depressions.py's module docstring for why these are
never build-phase-collapsed across different KernelBuilders).

Two closure-specific substitutions from the cupy version, both verified
directly against this Taichi/Quadrants install before use here:

- No 3-argument `range()` (reverse step) inside a kernel - confirmed
  Taichi rejects it ("Range should have 1 or 2 arguments"). The two
  right-to-left/bottom-to-top sweeps below instead drive a forward-
  counting loop variable and compute the descending index from it
  (`c = NX - 2 - cc`); still a single serial nested loop per thread, same
  execution order as a real reverse range.
- No `atomicExch` on Taichi (only Quadrants has `atomic_exchange`) - both
  backends use `ctx.bk.atomic_max(queued_gen[j], p)` instead, whose returned
  old value gives the identical "first writer this pass wins" dedup
  `atomicExch` does, because `p` only ever increases across passes: the
  first thread to touch queued_gen[j] this pass raises it from some
  earlier (smaller) value to p and gets that smaller value back; every
  later thread doing the same atomic_max this pass finds it already at p,
  contributes no change, and gets p back - confirmed empirically (a fresh
  -1-filled field, one atomic_max(..., p) per candidate, only the winner's
  returned old value differs from p) before relying on it here.

Author: B.G (08/2026)
"""

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

    Author: B.G (08/2026)
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

    Author: B.G (08/2026)
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

    Author: B.G (08/2026)
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
    output half, deduplicated per pass via `queued_gen` + atomic_max (see
    the module docstring). See
    ../../experimental/LM/fill_reconstruct_optimised.py's module docstring
    for the push-gate correctness argument.

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

    Author: B.G (08/2026)
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
