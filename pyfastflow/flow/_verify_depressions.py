"""Numerical verification for depression handling."""

import os
import sys
from collections import deque

import numpy as np

from ._verify_accum import make_smooth_terrain, numpy_topological_accum

DX = 1.0
SEED = 2024
SIDE = 1024
BLOCK = 256

COMBOS = (("vanilla", "carve"), ("vanilla", "jump"), ("optimized", "carve"), ("optimized", "jump"))


def _progress(message: str) -> None:
    """Emit opt-in phase markers when diagnosing a long GPU stress run."""
    if os.environ.get("PFF_VERIFY_PROGRESS"):
        print(f"[progress] {message}", file=sys.stderr, flush=True)


def make_noisy_terrain(nx: int, ny: int, seed: int) -> np.ndarray:
    """
    Plain i.i.d. uniform elevation, no spatial correlation: on a D8 grid a
    node is a local minimum whenever it is the smallest of the nine
    independent values in its own neighbourhood, so about 1/9 of the interior
    nodes are pits. Every basin is a handful of cells, and the outer loop has
    to resolve hundreds of thousands of them at once.

    """
    rng = np.random.default_rng(seed)
    return rng.random((ny, nx)).astype(np.float32).ravel()


def edge_mask(nx: int, ny: int) -> np.ndarray:
    """
    The can_out predicate of a grid built boundary="normal", outlet="edge":
    True on the four borders, False everywhere inside.

    """
    mask = np.zeros((ny, nx), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask.ravel()


def count_pits(rec: np.ndarray, can_out: np.ndarray) -> int:
    """
    Self-receiving nodes that cannot drain - the same quantity the
    depression_counter kernel accumulates, computed on the host.

    """
    return int(np.count_nonzero((rec == np.arange(rec.shape[0])) & ~can_out))


def peel_all_reached(rec: np.ndarray) -> tuple[bool, int]:
    """
    (every node processed, number processed) for one indegree-driven peel of
    the receiver forest, leaves first.

    A node only enters the queue once every donor of it has been processed,
    so a node on a cycle never does. Processing all n_flat nodes is therefore
    exactly "no cycle", and - since every node's chain then terminates at a
    self-receiver - also "every node reaches a root".

    """
    n = rec.shape[0]
    is_root = rec == np.arange(n)
    indeg = np.zeros(n, dtype=np.int64)
    np.add.at(indeg, rec[~is_root], 1)
    dq = deque(int(i) for i in np.flatnonzero(indeg == 0))
    processed = 0
    while dq:
        i = dq.popleft()
        processed += 1
        r = int(rec[i])
        if r != i:
            indeg[r] -= 1
            if indeg[r] == 0:
                dq.append(r)
    return processed == n, processed


def check_neighbour(rec: np.ndarray, nx: int, ny: int) -> int:
    """
    Number of nodes whose receiver is neither itself nor one of its eight
    grid neighbours - a D8 receiver is at most one row and one column away.

    """
    n = rec.shape[0]
    if rec.min() < 0 or rec.max() >= n:
        return int(np.count_nonzero((rec < 0) | (rec >= n)))
    i = np.arange(n)
    drow = np.abs(rec // nx - i // nx)
    dcol = np.abs(rec % nx - i % nx)
    return int(np.count_nonzero((drow > 1) | (dcol > 1)))


def check_all(rec: np.ndarray, nx: int, ny: int, can_out: np.ndarray) -> dict:
    """
    Every post-resolution invariant, as a dict of measured counts (0 means
    the invariant holds). `conservation` is skipped, and reported as None,
    when the graph has a cycle - accumulating over a cyclic graph is not
    defined.

    """
    n = rec.shape[0]
    in_range = bool(rec.min() >= 0 and rec.max() < n)
    out = {"pits": count_pits(rec, can_out), "bad_neighbour": check_neighbour(rec, nx, ny)}
    acyclic, processed = peel_all_reached(rec) if in_range else (False, 0)
    out["acyclic"] = acyclic
    out["unprocessed"] = n - processed
    roots = rec == np.arange(n)
    out["bad_roots"] = int(np.count_nonzero(roots & ~can_out))
    if acyclic:
        q = numpy_topological_accum(rec, np.ones(n, dtype=np.float32))
        out["conservation"] = float(q[roots].sum())
    else:
        out["conservation"] = None
    return out


def _label_only_sequence(backend, deps, grid_params, n_flat, rec, rec_jump, bid, basin_route=None):
    """
    A compiled Sequence (sequence.py) whose only block is `deps`'s
    basin-labelling block, so the two labelling implementations can be run
    over one and the same receiver graph and diffed.

    Reuses bound.py's own `bind_leaf` and flow/__init__.py's private
    `_bind_grid_everywhere` - the same leaf-name binding make_depression_solver
    uses for its own "label_basins" block - so what is measured here is the
    labelling block exactly as the solver itself drives it, not a
    re-implementation of the binding logic.

    """
    from ..core import SequenceBuilder
    from . import _bind_grid_everywhere, _bind_if_present

    # vanilla now labels from the carried basin_route (seeded here from rec),
    # not from rec directly - see make_depression_solver.
    is_route = "merge_basin_route" in deps

    sb = SequenceBuilder()
    if is_route:
        sb.add("seed", deps["copy_field"])
        sb.add("label_basins", deps["label_basins"])
        sb.step("seed")
        sb.step("label_basins")
    else:
        sb.add("label_basins", deps["label_basins"])
        sb.step("label_basins")
    frozen = sb.freeze()
    bound = frozen.build()

    if is_route:
        bound.bind(("seed", "src"), rec)
        bound.bind(("seed", "dst"), basin_route)
        bound.bind_leaf(
            {"rec_jump": basin_route, "basin_route": basin_route, "bid": bid},
            prefix=("label_basins",),
        )
    else:
        bound.bind_leaf({"rec": rec, "rec_jump": rec_jump, "bid": bid}, prefix=("label_basins",))
        _bind_if_present(bound, ("label_basins", "copy_rec_to_recjump", "src"), rec)
        _bind_if_present(bound, ("label_basins", "copy_rec_to_recjump", "dst"), rec_jump)
    _bind_grid_everywhere(bound, grid_params)

    compiled = bound.compile()
    bound.close()
    return compiled


def run(backend: str):
    """
    Build every combination's solver once, run each over both terrains, and
    return (n_flat, rows) - one row per (terrain, method, reroute) plus the
    vanilla-vs-optimized diff rows.

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
    from . import bind_depression_solver, make_depression_solver, make_depressions, make_receivers

    _bk = Backend.from_name(backend); ParamCls, dtypes = _bk.ParameterCls, _bk.dtypes
    i32, i64, f32, u8 = dtypes["i32"], dtypes["i64"], dtypes["f32"], dtypes["u8"]

    closure = backend in ("taichi", "quadrants")
    nx = ny = SIDE
    n = nx * ny
    can_out = edge_mask(nx, ny)

    def upload(handle, arr):
        handle.from_numpy(arr)

    def download(handle):
        return handle.to_numpy()

    pool = _bk.pool()
    grid = make_grid_group(_bk, topology="D8", boundary="normal", outlet="edge")
    grid_params = make_grid_parameters(_bk, pool, nx, ny, DX, topology="D8", outlet="edge")

    z = pool.get_data(f32, (n,))
    rec = pool.get_data(i32, (n,))
    rec_scratch = pool.get_data(i32, (n,))
    rec_jump = pool.get_data(i32, (n,))
    bid = pool.get_data(i32, (n,))
    z_prime = pool.get_data(f32, (n,))
    is_border = pool.get_data(u8, (n,))
    basin_saddle = pool.get_data(i64, (n,))
    basin_saddlenode = pool.get_data(i32, (n,))
    outlet = pool.get_data(i64, (n,))
    tag = pool.get_data(u8, (n,))
    tag_alt = pool.get_data(u8, (n,))
    rerouted = pool.get_data(u8, (n,))
    basin_route = pool.get_data(i32, (n,))
    b_rcv = pool.get_data(i32, (n,))

    ndep_p = ParamCls("NDEP", dtype="i32", mode="scalar", value=0, pool=pool)

    recv = make_receivers(_bk, grid, topology="D8", mode="steepest")
    recv_bound = recv["receivers"].build()
    recv_bound.bind_leaf(grid_params)
    recv_bound.bind("z", z)
    recv_bound.bind("rec", rec)
    recv_kernel = recv_bound.compile(_bk)

    buffers = dict(
        rec=rec, z=z, bid=bid, rec_jump=rec_jump, z_prime=z_prime,
        is_border=is_border, basin_saddle=basin_saddle, basin_saddlenode=basin_saddlenode,
        outlet=outlet, rerouted=rerouted, tag=tag, tag_alt=tag_alt,
        rec_scratch=rec_scratch, basin_route=basin_route, b_rcv=b_rcv,
    )

    solvers = {}
    labellers = {}
    for method, reroute in COMBOS:
        deps = make_depressions(_bk, grid, ndep_p, method=method, reroute=reroute, n_flat=n)
        frozen, _ = make_depression_solver(
            _bk, deps, grid_params, method=method, reroute=reroute, n_flat=n, block_size=BLOCK,
        )
        bound = bind_depression_solver(
            frozen, grid_params, ndep_p=ndep_p, method=method, reroute=reroute, **buffers,
        )
        solvers[(method, reroute)] = bound.compile(_bk)
        bound.close()
        if reroute == "carve":
            labellers[method] = _label_only_sequence(
                backend, deps, grid_params, n, rec, rec_jump, bid, basin_route
            )

    terrains = (
        ("smooth", make_smooth_terrain(nx, ny, SEED)),
        ("iid", make_noisy_terrain(nx, ny, SEED)),
    )

    rows = []
    for terrain_name, z_np in terrains:
        upload(z, z_np)
        recv_kernel()
        rec0 = download(rec).astype(np.int64)
        entry_pits = count_pits(rec0, can_out)

        # basin labelling, both methods, over the one unresolved graph
        bids = {}
        label_self = {}
        for method, seq in labellers.items():
            twice = []
            for _ in range(2):
                _progress(f"{backend} {terrain_name} label {method} run {len(twice) + 1}")
                upload(rec, rec0.astype(np.int32))
                seq()
                twice.append(download(bid).astype(np.int64))
            bids[method] = twice[0]
            label_self[method] = int(np.count_nonzero(twice[0] != twice[1]))
        label_diff = int(np.count_nonzero(bids["vanilla"] != bids["optimized"]))
        rows.append((terrain_name, "label_basins", "vanilla_vs_optimized", None,
                     {"bid_mismatch": label_diff, "self_vanilla": label_self["vanilla"],
                      "self_optimized": label_self["optimized"], "entry_pits": entry_pits}))

        resolved = {}
        for combo in COMBOS:
            method, reroute = combo
            runs = []
            for _ in range(2):
                _progress(f"{backend} {terrain_name} solve {method}/{reroute} run {len(runs) + 1}")
                upload(rec, rec0.astype(np.int32))
                upload(rerouted, np.zeros(n, dtype=np.uint8))
                solvers[combo]()
                runs.append(download(rec).astype(np.int64))
            got = runs[0]
            resolved[combo] = got
            checks = check_all(got, nx, ny, can_out)
            # same solver, same input graph, twice: any difference is this
            # combination's own run-to-run nondeterminism, which is what
            # separates a race from a genuine algorithmic disagreement
            # between the two methods.
            checks["self_mismatch"] = int(np.count_nonzero(runs[0] != runs[1]))
            checks["ndep_device"] = int(ndep_p.read())
            checks["entry_pits"] = entry_pits
            rows.append((terrain_name, method, reroute, solvers[combo].last_trip_counts, checks))

        for reroute in ("carve", "jump"):
            a = resolved[("vanilla", reroute)]
            b = resolved[("optimized", reroute)]
            rows.append((terrain_name, f"rec_{reroute}", "vanilla_vs_optimized", None,
                         {"rec_mismatch": int(np.count_nonzero(a != b))}))

    for _solver in solvers.values():  # release each compiled solver's hold on NDEP
        _solver.close()
    for _labeller in labellers.values():
        _labeller.close()
    recv_kernel.close()
    recv_bound.close()
    ndep_p.destroy()
    for h in (z, rec, rec_scratch, rec_jump, bid, z_prime, is_border, basin_saddle,
              basin_saddlenode, outlet, tag, tag_alt, rerouted, basin_route, b_rcv):
        pool.release_data(h)
    return n, rows


if __name__ == "__main__":
    backend_arg = sys.argv[1]
    n_flat, rows = run(backend_arg)
    print(f"{backend_arg}: n_flat={n_flat} side={SIDE}")
    for terrain, method, reroute, trips, checks in rows:
        trip_str = f" passes={list(trips)}" if trips else ""
        detail = " ".join(f"{k}={v}" for k, v in checks.items())
        print(f"{backend_arg} {terrain:7s} {method:14s} {reroute:19s}{trip_str} {detail}")
