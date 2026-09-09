"""
Tier 3: white-noise DEM -> resolve depressions -> assert none remain, over
every main grid combination.

Grid axes covered (topology fixed D8):
  boundary       normal / periodic_EW / periodic_NS
  nodata         off / on   (on: an interior rectangular nodata blob)
  outlet         edge / mask (mask: only the top row drains)
= 12 configs, each run on every backend present, each solved three ways:
  reconstruct, carve+vanilla, carve+optimized (reroute="jump" is skipped).

"No depression remains" is checked two ways on the resolved graph (`parent`
for reconstruct, `rec` for carve):
  - the device depression_counter kernel reports 0 (nodata/boundary-aware);
  - numpy: no non-outlet, non-nodata self-receiver; the graph is acyclic and
    every non-nodata node's chain ends on a can_out node.

Author: B.G (08/2026)
"""

import importlib

import numpy as np
import pytest

from pyfastflow.core.context.backends import Backend
from pyfastflow.flow._verify_depressions import make_noisy_terrain, peel_all_reached

DX = 1.0
SEED = 2024
SIDE = 128
BLOCK = 256

_BACKENDS = ("taichi", "quadrants", "cupy")

_CONFIGS = [
    (boundary, nodata, custom_outlet)
    for boundary in ("normal", "periodic_EW", "periodic_NS")
    for nodata in (False, True)
    for custom_outlet in (False, True)
]
_IDS = [f"{b}-{'nd' if nd else 'x'}-{'mask' if mo else 'edge'}" for b, nd, mo in _CONFIGS]


def _available(name: str) -> bool:
    try:
        mod = importlib.import_module(name)
    except Exception:
        return False
    if name == "cupy":
        try:
            return mod.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False
    return True


@pytest.fixture(params=_BACKENDS)
def backend(request):
    name = request.param
    if not _available(name):
        pytest.skip(f"{name} not available")
    if name == "taichi":
        import taichi as ti

        ti.init(arch=ti.gpu)
    elif name == "quadrants":
        import quadrants as qd

        qd.init(arch=qd.gpu)
    return name


def _pool_cls(name):
    if name == "taichi":
        from pyfastflow.core.pool.taichi_pool import TaichiPool as P
    elif name == "quadrants":
        from pyfastflow.core.pool.quadrants_pool import QuadrantsPool as P
    else:
        from pyfastflow.core.pool.cupy_pool import CupyPool as P
    return P


def _edge_can_out(boundary: str, nx: int, ny: int) -> np.ndarray:
    m = np.zeros((ny, nx), dtype=bool)
    if boundary in ("normal", "periodic_EW"):
        m[0, :] = True
        m[-1, :] = True
    if boundary in ("normal", "periodic_NS"):
        m[:, 0] = True
        m[:, -1] = True
    return m.ravel()


def _forest_roots(g: np.ndarray) -> np.ndarray:
    r = np.arange(g.shape[0])
    prev = None
    while prev is None or not np.array_equal(r, prev):
        prev = r
        r = g[r]
    return r


def _assert_resolved(g: np.ndarray, can_out: np.ndarray, nodata: np.ndarray, label: str) -> None:
    n = g.shape[0]
    live = nodata == 0
    self_recv = g == np.arange(n)
    pits = int(np.count_nonzero(self_recv & ~can_out & live))
    assert pits == 0, f"{label}: {pits} unresolved pit(s)"

    acyclic, processed = peel_all_reached(g)
    assert acyclic, f"{label}: receiver graph has a cycle ({processed}/{n} peeled)"

    roots = _forest_roots(g)
    stranded = int(np.count_nonzero(live & ~can_out[roots]))
    assert stranded == 0, f"{label}: {stranded} live node(s) drain to a non-outlet sink"


@pytest.mark.parametrize("boundary,nodata,custom_outlet", _CONFIGS, ids=_IDS)
def test_depressions_resolve(backend, boundary, nodata, custom_outlet):
    from pyfastflow.flow import (
        bind_depression_solver,
        make_depressions,
        make_fill_reconstruct,
        bind_fill_reconstruct_solver,
        make_depression_solver,
        make_fill_reconstruct_solver,
        make_receivers,
    )
    from pyfastflow.grid import make_grid_group, make_grid_parameters

    bk = Backend.from_name(backend)
    Param, dt = bk.ParameterCls, bk.dtypes
    i32, i64, f32, u8 = dt["i32"], dt["i64"], dt["f32"], dt["u8"]
    closure = backend in ("taichi", "quadrants")
    nx = ny = SIDE
    n = nx * ny
    outlet_cfg = "mask" if custom_outlet else "edge"
    launch = {} if closure else {"grid": ((n + BLOCK - 1) // BLOCK,), "block": (BLOCK,)}

    pool = _pool_cls(backend)()

    grid = make_grid_group(bk, topology="D8", boundary=boundary, nodata=nodata, outlet=outlet_cfg)
    gp = make_grid_parameters(bk, pool, nx, ny, DX, topology="D8", nodata=nodata, outlet=outlet_cfg)

    z_np = make_noisy_terrain(nx, ny, SEED).copy()
    nodata_np = np.zeros(n, dtype=np.uint8)
    if nodata:
        blob = np.zeros((ny, nx), dtype=np.uint8)
        blob[ny // 4 : ny // 2, nx // 4 : nx // 2] = 1
        nodata_np = blob.ravel()
        z_np[nodata_np == 1] = 9999.0
        gp["NODATA_MASK"].set(nodata_np)

    if custom_outlet:
        om = np.zeros((ny, nx), dtype=np.uint8)
        om[0, :] = 1
        outlet_np = om.ravel()
        gp["OUTLET_MASK"].set(outlet_np)
        can_out = outlet_np.astype(bool)
    else:
        can_out = _edge_can_out(boundary, nx, ny)

    z = pool.get_data(f32, (n,))
    z.from_numpy(z_np)
    rec = pool.get_data(i32, (n,))

    recv = make_receivers(bk, grid, topology="D8", mode="steepest")
    rb = recv["receivers"].build()
    rb.bind_leaf(gp)
    rb.bind("z", z)
    rb.bind("rec", rec)
    recv_kernel = rb.compile(backend)
    recv_kernel(**launch)
    recv_kernel.close()
    rb.close()
    rec0 = rec.to_numpy().astype(np.int32)

    ndep_p = Param("NDEP", dtype=i32, mode="scalar", value=0, pool=pool)

    # ---- reconstruct -------------------------------------------------------
    filled = pool.get_data(f32, (n,))
    parent = pool.get_data(i32, (n,))
    frontier = pool.get_data(i32, (2 * n,))
    queued_gen = pool.get_data(i32, (n,))
    max_passes = 4 * max(nx, ny)
    counters = pool.get_data(i32, (max_passes + 2,))
    counters.from_numpy(np.zeros(max_passes + 2, dtype=np.int32))
    queued_gen.from_numpy(np.full(n, -1, dtype=np.int32))
    pass_p = Param("PASS", dtype=i32, mode="scalar", value=0, pool=pool)
    active_p = Param("ACTIVE", dtype=i32, mode="scalar", value=0, pool=pool)

    fr = make_fill_reconstruct(bk, grid, nx=nx, ny=ny)
    fr_frozen, _ = make_fill_reconstruct_solver(
        bk, fr, gp, pass_p=pass_p, active_p=active_p,
        n_flat=n, nx=nx, ny=ny, block_size=BLOCK, max_passes=max_passes,
    )
    fr_bound = bind_fill_reconstruct_solver(
        fr_frozen, gp, z=z, filled=filled, parent=parent, frontier=frontier,
        counters=counters, queued_gen=queued_gen, pass_p=pass_p, active_p=active_p,
    )
    fr_solver = fr_bound.compile(backend, **launch)
    fr_bound.close()
    fr_solver()
    fr_solver.close()
    parent_np = parent.to_numpy().astype(np.int64)
    _assert_resolved(parent_np, can_out, nodata_np, "reconstruct")

    # device depression_counter on the reconstruct parent graph
    dc_deps = make_depressions(bk, grid, ndep_p, method="vanilla", reroute="carve", n_flat=n)
    parent_scratch = pool.get_data(i32, (n,))
    parent_scratch.from_numpy(parent_np.astype(np.int32))
    dc = dc_deps["depression_counter"].build()
    dc.bind_leaf(gp)
    dc.bind("rec", parent_scratch)
    dc.bind("ndep", ndep_p.handle())
    ndep_p.set(0)
    dc_kernel = dc.compile(backend)
    dc_kernel(**launch)
    dc_kernel.close()
    dc.close()
    assert int(ndep_p.read()) == 0, "reconstruct: device depression_counter != 0"

    # ---- carve, vanilla and optimized -----------------------------------
    carve_bufs = dict(
        rec=rec, z=z,
        bid=pool.get_data(i32, (n,)), rec_jump=pool.get_data(i32, (n,)),
        z_prime=pool.get_data(f32, (n,)), is_border=pool.get_data(u8, (n,)),
        basin_saddle=pool.get_data(i64, (n,)), basin_saddlenode=pool.get_data(i32, (n,)),
        outlet=pool.get_data(i64, (n,)), rerouted=pool.get_data(u8, (n,)),
        tag=pool.get_data(u8, (n,)), tag_alt=pool.get_data(u8, (n,)),
        rec_scratch=pool.get_data(i32, (n,)), basin_route=pool.get_data(i32, (n,)),
        b_rcv=pool.get_data(i32, (n,)),
    )

    for method in ("vanilla", "optimized"):
        rec.from_numpy(rec0)
        deps = make_depressions(bk, grid, ndep_p, method=method, reroute="carve", n_flat=n)
        deps_frozen, _ = make_depression_solver(
            bk, deps, gp, method=method, reroute="carve", n_flat=n, block_size=BLOCK,
        )
        deps_bound = bind_depression_solver(
            deps_frozen, gp, ndep_p=ndep_p, method=method, reroute="carve", **carve_bufs,
        )
        solver = deps_bound.compile(backend, **launch)
        deps_bound.close()
        solver()
        solver.close()
        got = rec.to_numpy().astype(np.int64)
        _assert_resolved(got, can_out, nodata_np, f"carve/{method}")
        assert int(ndep_p.read()) == 0, f"carve/{method}: device depression_counter != 0"

    pool.clear_all(force=True)
