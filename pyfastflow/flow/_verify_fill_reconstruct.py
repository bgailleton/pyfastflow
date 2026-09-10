"""
Standalone, re-runnable verification of make_fill_reconstruct +
make_fill_reconstruct_solver, against an independent numpy/heapq
priority-flood reference (Barnes 2014) - a different algorithm computing the
same fixed point, not a second copy of the frontier-relaxation logic under
test.

What it checks, per backend (taichi/quadrants/cupy), per terrain: builds a
grid, runs the solver, and inspects `filled`/`parent` against the reference:

  - `filled` matches the numpy priority-flood reference to within a small f32
    tolerance, everywhere - the actual correctness question, independent of
    how the device got there;
  - `filled[i] >= z[i]` everywhere (never fills below true elevation) and
    `filled[i] == z[i]` at every can_out (root) node - the two boundary
    conditions the fixed point must satisfy;
  - the local fixed-point condition itself: `filled[i] == max(z[i],
    filled[parent[i]])` for every non-root i - if this fails while `filled`
    still matches the reference, relax converged to the right values through
    a parent graph that doesn't actually explain them;
  - `parent` never left at -1 (every node claimed) and never wanders outside
    self-or-a-D8-neighbour (a wild write during relax) - the same
    `bad_neighbour` check _verify_depressions.py uses for reroute="carve";
  - the receiver graph `parent` forms is acyclic and every node reaches a
    can_out root - one indegree peel, reused from _verify_depressions;
  - flow is conserved - a unit source accumulated over `parent` totals n_flat
    across the roots.

SIDE is smaller than _verify_depressions.py's 1024: the reference here is a
pure-python heapq priority flood (no numpy vectorization is possible for a
priority queue), so its own cost, not the GPU solver's, sets the practical
grid size for a script meant to be run once and read.

Run:
    python -m pyfastflow.flow._verify_fill_reconstruct taichi
    python -m pyfastflow.flow._verify_fill_reconstruct quadrants
    python -m pyfastflow.flow._verify_fill_reconstruct cupy

Author: B.G (07/2026)
"""

import heapq
import sys

import numpy as np

from ._verify_accum import make_smooth_terrain, numpy_topological_accum
from ._verify_depressions import check_neighbour, edge_mask, make_noisy_terrain, peel_all_reached

DX = 1.0
SEED = 2024
SIDE = 512
BLOCK = 256
FTOL = 1e-3


def numpy_priority_flood(z: np.ndarray, nx: int, ny: int, can_out: np.ndarray) -> np.ndarray:
    """
    Reference `filled` surface via Barnes (2014) priority-flood: a min-heap
    seeded with every can_out node at its own elevation, expanding to
    unvisited D8 neighbours at max(neighbour's own z, the popped value) -
    the same fixed point grayscale reconstruction converges to, computed by
    a completely different (serial, heap-ordered) algorithm.

    Author: B.G (07/2026)
    """
    n = nx * ny
    filled = np.full(n, np.inf, dtype=np.float64)
    visited = np.zeros(n, dtype=bool)
    heap = []
    for i in np.flatnonzero(can_out):
        filled[i] = float(z[i])
        visited[i] = True
        heap.append((filled[i], int(i)))
    heapq.heapify(heap)
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while heap:
        h, i = heapq.heappop(heap)
        if h > filled[i]:
            continue
        r, c = divmod(i, nx)
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx:
                j = rr * nx + cc
                if not visited[j]:
                    visited[j] = True
                    cand = max(float(z[j]), h)
                    filled[j] = cand
                    heapq.heappush(heap, (cand, j))
    return filled


def check_all(filled: np.ndarray, parent: np.ndarray, z: np.ndarray, ref: np.ndarray, nx: int, ny: int, can_out: np.ndarray) -> dict:
    """
    Every post-solve invariant plus the reference comparison, as a dict of
    measured counts/values (0 / small means the invariant holds).

    Author: B.G (07/2026)
    """
    n = filled.shape[0]
    out = {}
    diff = np.abs(filled.astype(np.float64) - ref)
    out["max_abs_diff_vs_reference"] = float(diff.max())
    out["never_set_parent"] = int(np.count_nonzero(parent == -1))
    out["below_z"] = int(np.count_nonzero(filled < z - FTOL))
    out["root_not_at_z"] = int(np.count_nonzero(can_out & (np.abs(filled - z) > FTOL)))
    out["bad_neighbour"] = check_neighbour(parent.astype(np.int64), nx, ny)

    is_root = parent == np.arange(n)
    fixed_point_ok = np.abs(filled - np.maximum(z, filled[np.clip(parent, 0, n - 1)])) <= FTOL
    out["fixed_point_violations"] = int(np.count_nonzero(~fixed_point_ok & ~is_root & (parent != -1)))

    in_range = bool(parent.min() >= 0 and parent.max() < n)
    acyclic, processed = peel_all_reached(parent.astype(np.int64)) if in_range else (False, 0)
    out["acyclic"] = acyclic
    out["unprocessed"] = n - processed
    out["bad_roots"] = int(np.count_nonzero(is_root & ~can_out))
    if acyclic:
        q = numpy_topological_accum(parent.astype(np.int64), np.ones(n, dtype=np.float32))
        out["conservation"] = float(q[is_root].sum())
    else:
        out["conservation"] = None
    return out


def run(backend: str):
    """
    Build the solver once, run it over two terrains, and return (n_flat, rows).

    Author: B.G (08/2026)
    """
    if backend == "taichi":
        import taichi as ti
        ti.init(arch=ti.gpu)
    elif backend == "quadrants":
        import quadrants as qd
        qd.init(arch=qd.gpu)
    elif backend != "cupy":
        raise ValueError(f"unknown backend {backend!r}")

    from ..core import Backend
    from ..grid import make_grid_group, make_grid_parameters
    from . import bind_fill_reconstruct_solver, make_fill_reconstruct, make_fill_reconstruct_solver

    _bk = Backend.from_name(backend); ParamCls, dtypes = _bk.ParameterCls, _bk.dtypes
    i32, f32 = dtypes["i32"], dtypes["f32"]

    nx = ny = SIDE
    n = nx * ny
    can_out = edge_mask(nx, ny)

    def upload(handle, arr):
        handle.from_numpy(arr)

    def download(handle):
        return handle.to_numpy()

    pool = _bk.pool()
    grid_group = make_grid_group(_bk, topology="D8", boundary="normal", outlet="edge")
    grid_params = make_grid_parameters(_bk, pool, nx, ny, DX, topology="D8", outlet="edge")

    z = pool.get_data(f32, (n,))
    filled = pool.get_data(f32, (n,))
    parent = pool.get_data(i32, (n,))
    frontier = pool.get_data(i32, (2 * n,))
    queued_gen = pool.get_data(i32, (n,))

    pass_p = ParamCls("PASS", dtype="i32", mode="scalar", value=0, pool=pool)
    active_p = ParamCls("ACTIVE", dtype="i32", mode="scalar", value=0, pool=pool)

    deps = make_fill_reconstruct(_bk, grid_group, nx=nx, ny=ny)
    max_passes = 4 * max(nx, ny)
    counters = pool.get_data(i32, (max_passes + 2,))

    frozen, _ = make_fill_reconstruct_solver(
        _bk, deps, grid_params, pass_p=pass_p, active_p=active_p,
        n_flat=n, nx=nx, ny=ny, block_size=BLOCK, max_passes=max_passes,
    )
    bound = bind_fill_reconstruct_solver(
        frozen, grid_params, z=z, filled=filled, parent=parent, frontier=frontier,
        counters=counters, queued_gen=queued_gen, pass_p=pass_p, active_p=active_p,
    )
    solver = bound.compile(_bk)
    bound.close()

    terrains = (
        ("smooth", make_smooth_terrain(nx, ny, SEED)),
        ("iid", make_noisy_terrain(nx, ny, SEED)),
    )

    rows = []
    for terrain_name, z_np in terrains:
        upload(z, z_np)
        upload(counters, np.zeros(max_passes + 2, dtype=np.int32))
        upload(queued_gen, np.full(n, -1, dtype=np.int32))

        solver()

        filled_np = download(filled).astype(np.float64)
        parent_np = download(parent).astype(np.int64)
        ref = numpy_priority_flood(z_np.astype(np.float64), nx, ny, can_out)

        checks = check_all(filled_np, parent_np, z_np.astype(np.float64), ref, nx, ny, can_out)
        checks["passes"] = list(solver.last_trip_counts)
        checks["active_final"] = int(active_p.read())
        rows.append((terrain_name, checks))

    solver.close()  # release the compiled sequence's hold on the Parameters (Unit 6)
    pass_p.destroy()
    active_p.destroy()
    for h in (z, filled, parent, frontier, queued_gen, counters):
        pool.release_data(h)
    return n, rows


if __name__ == "__main__":
    backend_arg = sys.argv[1]
    n_flat, rows = run(backend_arg)
    print(f"{backend_arg}: n_flat={n_flat} side={SIDE}")
    for terrain, checks in rows:
        detail = " ".join(f"{k}={v}" for k, v in checks.items())
        print(f"{backend_arg} {terrain:7s} {detail}")
