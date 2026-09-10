"""Numerical verification for GraphFlood."""

import sys

import numpy as np
from scipy.ndimage import gaussian_filter

DX = 1.0
SEED = 2024
SIDE = 64
BLOCK = 256
N_STEPS = 200
RAIN = 1.0e-5
DT = 1.0
MANNING = 0.033
EXPO = 2.0 / 3.0


def make_terrain(nx: int, ny: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(ny, nx)).astype(np.float32)
    smooth = gaussian_filter(raw, sigma=max(1.0, 0.02 * nx), mode="nearest")
    row = np.arange(ny)[:, None]
    col = np.arange(nx)[None, :]
    slope = 0.01 * (row + col)
    return (smooth + slope).astype(np.float32).ravel()


def run(backend: str, fill_method: str) -> None:
    from ..core import Backend
    from ..grid import make_grid_group, make_grid_parameters
    from . import bind_graphflood, make_graphflood

    _bk = Backend.from_name(backend); ParamCls, dtypes = _bk.ParameterCls, _bk.dtypes
    i32, i64, f32 = dtypes["i32"], dtypes["i64"], dtypes["f32"]

    nx = ny = SIDE
    n = nx * ny
    pool = _bk.pool()

    grid_group = make_grid_group(_bk, topology="D8", boundary="normal", outlet="edge")
    grid_params = make_grid_parameters(_bk, pool, nx, ny, DX, topology="D8", outlet="edge")

    z_np = make_terrain(nx, ny, SEED)
    z = pool.get_data(f32, (n,))
    h = pool.get_data(f32, (n,))
    Q_in = pool.get_data(f32, (n,))
    Qo = pool.get_data(f32, (n,))
    z.from_numpy(z_np)
    h.from_numpy(np.zeros(n, dtype=np.float32))

    # SOURCE is Q (m^3/s per cell): rain rate * cell area, not a bare rate -
    # see apply_divergence's (Q_in - Qo)/area*dt (Qo is m^3/s). DX=1 here so
    # this is currently a no-op, but stays explicit rather than relying on
    # that (see graphflood_cli.py's own note - this exact omission was a
    # real, DX-masked bug there).
    source_p = ParamCls("SOURCE", dtype="f32", mode="const", value=RAIN * DX * DX, pool=pool)
    manning_p = ParamCls("MANNING", dtype="f32", mode="const", value=MANNING, pool=pool)
    expo_p = ParamCls("EXPO", dtype="f32", mode="const", value=EXPO, pool=pool)
    dt_p = ParamCls("DT", dtype="f32", mode="const", value=DT, pool=pool)
    boundary_h_p = ParamCls("BOUNDARY_H", dtype="f32", mode="const", value=0.0, pool=pool)
    gf_min_increment_p = ParamCls("GF_MIN_INCREMENT", dtype="f32", mode="const", value=0.0, pool=pool)

    structure_kwargs = dict(
        n_flat=n, nx=nx, ny=ny,
        fill_method=fill_method, block_size=BLOCK,
    )
    bindings = dict(
        z=z, h=h, Q_in=Q_in, Qo=Qo,
        source_p=source_p, manning_p=manning_p, friction_exponent_p=expo_p, dt_p=dt_p,
        boundary_h_p=boundary_h_p, gf_min_increment_p=gf_min_increment_p,
    )

    extra = {}
    if fill_method == "jump":
        rec = pool.get_data(i32, (n,))
        bid = pool.get_data(i32, (n,))
        rec_jump = pool.get_data(i32, (n,))
        z_prime = pool.get_data(f32, (n,))
        is_border = pool.get_data(i32, (n,))
        basin_saddle = pool.get_data(i64, (n,))
        basin_saddlenode = pool.get_data(i32, (n,))
        outlet_h = pool.get_data(i64, (n,))
        rerouted = pool.get_data(i32, (n,))
        basin_route = pool.get_data(i32, (n,))
        b_rcv = pool.get_data(i32, (n,))
        ndep_p = ParamCls("NDEP", dtype="i32", mode="scalar", value=0, pool=pool)
        extra = dict(
            rec=rec, ndep_p=ndep_p, bid=bid, rec_jump=rec_jump, z_prime=z_prime,
            is_border=is_border, basin_saddle=basin_saddle, basin_saddlenode=basin_saddlenode,
            outlet=outlet_h, rerouted=rerouted, b_rcv=b_rcv, basin_route=basin_route,
        )
    else:
        surface = pool.get_data(f32, (n,))
        filled = pool.get_data(f32, (n,))
        parent = pool.get_data(i32, (n,))
        frontier = pool.get_data(i32, (2 * n,))
        max_passes = 4 * max(nx, ny)
        counters = pool.get_data(i32, (max_passes + 2,))
        queued_gen = pool.get_data(i32, (n,))
        pass_p = ParamCls("P", dtype="i32", mode="scalar", value=0, pool=pool)
        active_p = ParamCls("ACTIVE", dtype="i32", mode="scalar", value=0, pool=pool)
        counters.from_numpy(np.zeros(max_passes + 2, dtype=np.int32))
        queued_gen.from_numpy(np.full(n, -1, dtype=np.int32))
        extra = dict(
            surface=surface, filled=filled, parent=parent, frontier=frontier,
            counters=counters, queued_gen=queued_gen, pass_p=pass_p, active_p=active_p,
        )
        structure_kwargs["max_passes"] = max_passes

    frozen, _ = make_graphflood(_bk, grid_group, **structure_kwargs)
    gf = bind_graphflood(frozen, grid_params, **bindings, **extra)

    outlet_mask = None
    total_in = 0.0
    total_out = 0.0
    for step in range(N_STEPS):
        gf.step()
        h_np = h.to_numpy()
        if not np.all(np.isfinite(h_np)):
            print(f"[{backend}/{fill_method}] step {step}: non-finite h - FAIL")
            return
        if h_np.min() < -1e-6:
            print(f"[{backend}/{fill_method}] step {step}: negative h (min={h_np.min():.3e}) - FAIL")
            return
        total_in += RAIN * n * DT
        if outlet_mask is None:
            row = np.arange(n) // nx
            col = np.arange(n) % nx
            outlet_mask = (row == 0) | (row == ny - 1) | (col == 0) | (col == nx - 1)
        total_out += float(Q_in.to_numpy()[outlet_mask].sum()) * DT

    h_final = h.to_numpy()
    stored = float(h_final.sum())
    print(
        f"[{backend}/{fill_method}] steps={N_STEPS} h_max={h_final.max():.4g} "
        f"stored_volume={stored:.4g} total_in={total_in:.4g} total_out~{total_out:.4g} "
        f"balance_residual~{total_in - stored - total_out:.4g}"
    )
    gf.close()


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "taichi"
    if backend == "taichi":
        import taichi as ti
        ti.init(arch=ti.gpu)
    elif backend == "quadrants":
        import quadrants as qd
        qd.init(arch=qd.gpu)
    elif backend != "cupy":
        raise ValueError(f"unknown backend {backend!r}")
    for fill_method in ("jump", "reconstruct"):
        run(backend, fill_method)


if __name__ == "__main__":
    main()
