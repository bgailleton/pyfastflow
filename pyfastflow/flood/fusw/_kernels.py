"""CuPy kernels of FuswProgram and the routine of one time step.

Faces: x faces (ny, nx + 1), face ``row (nx + 1) + col`` west of cell
(row, col); y faces (ny + 1, nx), face ``row nx + col`` north of it. A
periodic axis uses face 0 of the row (column) as the east (south) face of
its last cell and leaves the extra face at 0. Per face: mass flux f1,
x-momentum flux f2 and y-momentum flux f3 (m2/s, m3/s2) and, with order 2,
the face bed max(z_l, z_r) of the reconstructed beds of its two sides
(``zx``, ``zy``).

``flux`` (one thread per cell, its west and north faces, plus the east /
south face of the last column / row on a non-periodic axis): each side of a
face is the reconstruction of its cell (``_reconstruction.py``) or, beyond
a boundary face, the ghost state of the active cell (``_boundary.py``). The
hydrostatic reconstruction (Audusse et al. 2004) cuts the depths by the bed
step dz = z_r - z_l, h_l <- max(h_l - max(dz, 0), 0) and
h_r <- max(h_r - max(-dz, 0), 0), before the Riemann flux
(``_riemann.py``). The largest wave speed goes to a control slot.

``stage`` (one thread per cell) advances a state over dt (FullSWOF):

  h* = h - dt/dx (div f1) + (RAIN f_rain + SOURCE f_source / dx^2) dt,
  infiltration (``_infiltration.py``),
  q*_x = h u - dt/dx (f2_E - f2_W + g/2 ((h_W' - h_l)(h_W' + h_l)
         + (h_r - h_E')(h_r + h_E') + (h_l + h_r) dzc)) - dt/dx (f2_S - f2_N),
  q*_y likewise along y,

with h_l, h_r, dzc = z_r - z_l the cell's own reconstruction and h_W',
h_E' its hydrostatic depths on its west / east faces: its own reconstructed
depth cut by the bed step to the other side, max(dz, 0) = z_f - z_own with
z_f = max(z_l, z_r) the face bed (order 1: from z of the neighbour or the
ghost bed; order 2: stored by ``flux``), the same bits as in ``flux``; the
uncut depth on an inflow face. Then the friction
divisor (``_friction.py``, input velocities), u = q / h above HE_CA, and
the cuts h <= HE_CA -> dry, |u| <= VE_CA -> 0. Kinds:

  - ``euler`` (order 1): h, u, v in place;
  - ``first`` (order 2): h, u, v -> hs, us, vs;
  - ``second`` (order 2): hs, us, vs -> the Heun average with h, u, v,
    h <- (h + h*) / 2, q <- (h u + h* u*) / 2 (dry below HE_CA).

``euler`` and the ungated ``first`` take dt from the control: with
``time_step="cfl"`` dt = min(DT_MAX, CFL dx / (c_max g)) (DT_MAX <= 0 means
CFL dx, c_max the largest wave speed of the step-start fluxes, g = 1 for
``euler`` and for ``first`` the growth of the wave speed over the first
stage of the last step, at least 1, so that the intermediate state rarely
needs a smaller dt), with
``"fixed"`` dt = DT and a CFL error flag when c_max DT / dx > CFL; both
are cut at the end time. dt is evaluated in ``real`` by every thread and
stored by thread 0; ``check`` repeats the same expression so that equal
wave speeds give the same dt. Sinks absorb their water after ``euler`` and
``second``. Each stage adds its infiltrated and threshold volumes and its
boundary-face terms, weighted 1 / order, to the step budget (warp sums
skipped on warps where they are all zero); rain and source volumes, which
do not depend on the state, are added by ``finalize`` as dt times their
time factor times the rates of the active cells summed by ``prepare``.

``check`` (order 2, after the intermediate fluxes): with ``"cfl"``, when
min(dt, CFL dx / c_max) of the intermediate state is below dt and fewer
than ``max_restarts`` redos ran in the step, it sets that dt, clears the
step budget and raises the redo flag; the gated ``flux`` / ``first`` /
``flux`` / ``check`` that follow then recompute the first stage from the
step-start state; otherwise it stores the growth c_max(intermediate) /
c_max(step start) of the step for the next one. With ``"fixed"`` it raises
the CFL error flag.

``finalize`` (one thread per budget entry) commits the step budget plus the
rain and source volumes, then thread 0 advances time and the step counter,
clears the step flags and evaluates the time factors; ``record`` writes a
row (time, dt, probe h / u / v, budget) every RECORD_INTERVAL steps.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder, RoutineBuilder

from ._boundary import make_ghost, make_inflow_face
from ._common import (
    B_FACE, B_INFILTRATION, B_RAIN, B_SINK, B_SOURCE, B_THRESHOLD,
    BOUNDARY_CODES, C_DT, C_GROWTH, C_INFLOW, C_RAIN, C_RAIN_RATE, C_SOURCE,
    C_SOURCE_RATE, C_T_END, C_TIME,
    K_CFL_ERROR, K_CMAX0, K_CMAX1, K_REC_BASE, K_REC_OVERFLOW, K_REDO,
    K_RESTARTS, K_STEP, K_TOTAL_RESTARTS, N_BUDGET, N_FACE_TERMS,
    PRECISIONS, cupy_only, literal, warp_reduce,
)
from ._friction import make_friction, make_uniform
from ._infiltration import make_infiltration
from ._limiters import make_limiter
from ._reconstruction import make_axis, make_side
from ._riemann import make_flux
from ._diagnostics import ACC_CELL, ACC_FACE, ACC_IN, FROUDE_CELL, FROUDE_FACE
from ._shear import CELL_OUT, FACE_OUT, cell_kernel, face_kernel

FACES = ("fx1", "fx2", "fx3", "fy1", "fy2", "fy3", "zx", "zy")
STATE = ("h", "u", "v")
STAGE_STATE = ("hs", "us", "vs")

PARAM_BIND = {
    "GRAVITY": "gravity", "HE_CA": "he_ca", "VE_CA": "ve_ca", "IE_CA": "ie_ca",
    "EPSILON": "epsilon", "FRICTION": "friction_coefficient",
    "KS": "ks", "KC": "kc", "DTHETA": "dtheta", "PSI": "psi",
    "ZCRUST": "zcrust", "IMAX": "imax", "STAGE": "stage", "INFLOW": "inflow",
    "OUTLET_SLOPE": "outlet_slope", "OUTLET_MANNING": "outlet_manning",
    "RATING_A": "rating_a", "RATING_B": "rating_b", "RATING_ETA0": "rating_eta0",
    "NEWTON_ITERATIONS": "newton_iterations", "RAIN": "rain", "SOURCE": "source",
    "CFL": "cfl", "DT": "dt", "DT_MAX": "dt_max",
    "RECORD_INTERVAL": "record_interval", "RHO": "rho",
    "LOCAL_DT_RATIO": "local_dt_ratio",
}


def _lines(config):
    return int(config["inflow_lines"])


def _geometry(config):
    nx, ny = int(config["nx"]), int(config["ny"])
    return (nx, ny, nx * ny, config["boundary"] == "periodic_EW",
            config["boundary"] == "periodic_NS")


def _real(config):
    return PRECISIONS[config["precision"]]


def _helpers(bundles, config):
    """Ghost, side, axis, flux, friction and infiltration helpers."""
    real = _real(config)
    grid = bundles["grid"]
    ghost = make_ghost(real, config["inflow_unit"])
    side = make_side(grid, ghost, real)
    limiter = make_limiter(config["limiter"], real)
    axis = make_axis(int(config["order"]), grid, side, limiter, real)
    return {
        "side": side, "axis": axis, "flux": make_flux(config["flux"], real),
        "friction": make_friction(config["friction"], real),
        "infiltration": make_infiltration(config["infiltration"], real),
        "inflow": make_inflow_face(real),
        "uniform": make_uniform(config["friction"], real),
    }


def _compose(kb, bundles, helpers, names):
    kb = kb.compose("grid", bundles["grid"])
    for name in names:
        kb = kb.compose(name, helpers[name])
    return kb.freeze()


# ----------------------------------------------------------------------------
# face fluxes
# ----------------------------------------------------------------------------

def _flux_kernel(bundles, config, helpers, src, slot, gated):
    nx, ny, n, px, py = _geometry(config)
    real = _real(config)
    c = literal(real)
    h, u, v = src
    bed = int(config["order"]) == 2
    gate = f"if (control[{K_REDO}] == 0u) return;" if gated else ""
    reduce = (f'''
    cmax = fusw_warp_max(cmax);
    if ((threadIdx.x & 31) == 0 && cmax > {c(0)})
        atomicMax(&control[{slot}], __float_as_uint((float)cmax));'''
              if slot is not None else "")
    lines = _lines(config) > 0
    face_args = f"{h}, {u}, {v}, z, boundary_type, boundary_faces" + (
        ", inflow_q, inflow_depth" if lines else "")
    inflow_args = f", const {real}* iq, const {real}* idp" if lines else ""
    inflow_kernel_args = (f", const {real}* inflow_q, const {real}* inflow_depth"
                          if lines else "")
    inflow_text = f'''
    int ic = -1, isg = 0;
    if (a < 0 && code[b] == {BOUNDARY_CODES["inflow"]} && ((fmask[b] >> (axis == 0 ? 1 : 0)) & 1)) {{
        ic = b; isg = -1;
    }} else if (b < 0 && code[a] == {BOUNDARY_CODES["inflow"]} && ((fmask[a] >> (axis == 0 ? 2 : 3)) & 1)) {{
        ic = a; isg = 1;
    }}
    if (ic >= 0) {{
        {real} hc = isg < 0 ? hR : hL;
        {real} uc = isg < 0 ? nR : nL;
        {real} f1, fn, hb, cw;
        $ctx.inflow(h[ic], axis == 0 ? u[ic] : v[ic], isg, iq[ic], idp[ic], hc, uc,
                    &f1, &fn, &hb, &cw)$;
        *F1 = f1;
        *F2 = axis == 0 ? fn : {c(0)};
        *F3 = axis == 0 ? {c(0)} : fn;{" *Z = fmax(zL, zR);" if bed else ""}
        *C = cw;
        return;
    }}''' if lines else ""

    def face(a, b, axis, out, index):
        f1, f2, f3 = ("fx1", "fx2", "fx3") if axis == 0 else ("fy1", "fy2", "fy3")
        zf = ("zx" if axis == 0 else "zy") if bed else None
        return f'''fusw_face({face_args}, {a}, {b}, {axis}, dx, factor,
                      &F1, &F2, &F3,{" &Z," if bed else ""} &C);
            {f1}[{index}] = F1; {f2}[{index}] = F2; {f3}[{index}] = F3;{f" {zf}[{index}] = Z;" if bed else ""}
            cmax = fmax(cmax, C);'''

    def zero(axis, index):
        names = ("fx1", "fx2", "fx3") if axis == 0 else ("fy1", "fy2", "fy3")
        if bed:
            names += ("zx",) if axis == 0 else ("zy",)
        return " ".join(f"{name}[{index}] = {c(0)};" for name in names)

    west_raw = f"x > 0 ? i - 1 : {f'i + {nx - 1}' if px else '-1'}"
    north_raw = f"y > 0 ? i - {nx} : {f'i + {n - nx}' if py else '-1'}"
    east = (f"{zero(0, 'fxw + 1')}" if px else
            f"if (act) {{ {face('i', '-1', 0, 'x', 'fxw + 1')} }} else {{ {zero(0, 'fxw + 1')} }}")
    south = (f"{zero(1, f'fyn + {nx}')}" if py else
             f"if (act) {{ {face('i', '-1', 1, 'y', f'fyn + {nx}')} }} else {{ {zero(1, f'fyn + {nx}')} }}")
    text = warp_reduce(real) + f'''
__device__ void fusw_face(const {real}* h, const {real}* u, const {real}* v,
                          const {real}* z, const unsigned char* code,
                          const unsigned char* fmask{inflow_args}, int a, int b, int axis,
                          {real} dx, {real} factor,
                          {real}* F1, {real}* F2, {real}* F3,{f" {real}* Z," if bed else ""} {real}* C) {{
    {real} hL, nL, tL, zL, hR, nR, tR, zR, d0, d1, d2, d3;
    if (a >= 0) $ctx.axis(h, u, v, z, code, fmask, a, axis, dx, factor,
                          &d0, &hL, &d1, &nL, &d2, &tL, &d3, &zL)$;
    else $ctx.side(h, u, v, z, code, fmask, b, axis == 0 ? 1 : 0, dx, factor, &hL, &nL, &tL, &zL)$;
    if (b >= 0) $ctx.axis(h, u, v, z, code, fmask, b, axis, dx, factor,
                          &hR, &d0, &nR, &d1, &tR, &d2, &zR, &d3)$;
    else $ctx.side(h, u, v, z, code, fmask, a, axis == 0 ? 2 : 3, dx, factor, &hR, &nR, &tR, &zR)$;{inflow_text}
    {real} dz = zR - zL;{" *Z = fmax(zL, zR);" if bed else ""}
    {real} hl = fmax(hL - fmax(dz, {c(0)}), {c(0)});
    {real} hr = fmax(hR - fmax(-dz, {c(0)}), {c(0)});
    {real} f1, fn, ft, cfl;
    $ctx.flux(hl, nL, tL, hr, nR, tR, &f1, &fn, &ft, &cfl)$;
    *F1 = f1;
    *F2 = axis == 0 ? fn : ft;
    *F3 = axis == 0 ? ft : fn;
    *C = cfl;
}}

extern "C" __global__ void fusw_flux(
        const {real}* {h}, const {real}* {u}, const {real}* {v}, const {real}* z,
        const unsigned char* boundary_type, const unsigned char* boundary_faces{inflow_kernel_args},
        {real}* fx1, {real}* fx2, {real}* fx3, {real}* fy1, {real}* fy2, {real}* fy3,{f" {real}* zx, {real}* zy," if bed else ""}
        const double* clock, unsigned int* control) {{
    {gate}
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    {real} cmax = {c(0)};
    if (i < {n}) {{
        int x = i % {nx};
        int y = i / {nx};
        {real} dx = ({real})$ctx.grid.DX.get(0)$;
        {real} factor = ({real})clock[{C_INFLOW}];
        int fxw = y * {nx + 1} + x;
        int fyn = y * {nx} + x;
        bool act = $ctx.grid.is_active(i)$;
        {real} F1, F2, F3,{" Z," if bed else ""} C;
        int a = -1, b = -1;
        if (act) {{ b = i; a = $ctx.grid.neighbour(i, 1)$; }}
        else {{ int w = {west_raw}; if (w >= 0 && $ctx.grid.is_active(w)$) a = w; }}
        if (a >= 0 || b >= 0) {{ {face('a', 'b', 0, 'x', 'fxw')} }} else {{ {zero(0, 'fxw')} }}
        if (x == {nx - 1}) {{ {east} }}
        a = -1; b = -1;
        if (act) {{ b = i; a = $ctx.grid.neighbour(i, 0)$; }}
        else {{ int w = {north_raw}; if (w >= 0 && $ctx.grid.is_active(w)$) a = w; }}
        if (a >= 0 || b >= 0) {{ {face('a', 'b', 1, 'y', 'fyn')} }} else {{ {zero(1, 'fyn')} }}
        if (y == {ny - 1}) {{ {south} }}
    }}{reduce}
}}
'''
    return _compose(KernelBuilder(text, domain=n), bundles, helpers,
                    ("axis", "side", "flux") + (("inflow",) if lines else ()))


# ----------------------------------------------------------------------------
# inflow lines
# ----------------------------------------------------------------------------

LINE_BLOCK = 256


def _line_kernel(bundles, config, helpers):
    real = _real(config)
    n_lines = _lines(config)
    text = f'''
__device__ double fusw_block_reduce(double v, double* shared, bool take_min) {{
    for (int o = 16; o > 0; o >>= 1) {{
        double w = __shfl_down_sync(0xffffffffu, v, o);
        v = take_min ? fmin(v, w) : v + w;
    }}
    __syncthreads();
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = v;
    __syncthreads();
    double r = shared[0];
    for (int k = 1; k < {LINE_BLOCK // 32}; ++k) r = take_min ? fmin(r, shared[k]) : r + shared[k];
    return r;
}}

__device__ double fusw_line_q(const {real}* z, const int* cells, int a, int e,
                              double eta, double slope, double* shared) {{
    double sum = 0.0;
    for (int f = a + (int)threadIdx.x; f < e; f += {LINE_BLOCK}) {{
        int i = cells[f];
        sum += (double)$ctx.uniform(({real})fmax(eta - (double)z[i], 0.0), ({real})slope, i)$;
    }}
    return fusw_block_reduce(sum, shared, false);
}}

extern "C" __global__ void fusw_lines(
        const {real}* z, const int* line_cells, const int* line_offsets,
        const double* line_q, const double* line_slope, double* line_last,
        const double* clock, {real}* inflow_q, {real}* inflow_depth) {{
    __shared__ double shared[{LINE_BLOCK // 32}];
    int l = blockIdx.x;
    if (l >= {n_lines}) return;
    int a = line_offsets[l], e = line_offsets[l + 1];
    double q_total = line_q[l] * clock[{C_INFLOW}];
    if (q_total == line_last[l] || e <= a) return;
    double dx = (double)$ctx.grid.DX.get(0)$;
    double slope = line_slope[l];
    double zmin = 1.0e300;
    for (int f = a + (int)threadIdx.x; f < e; f += {LINE_BLOCK}) zmin = fmin(zmin, (double)z[line_cells[f]]);
    zmin = fusw_block_reduce(zmin, shared, true);
    double eta = zmin, total = 0.0;
    if (q_total > 0.0) {{
        double target = q_total / dx;
        double lo = zmin, hi = zmin + 1.0;
        for (int it = 0; it < 64 && fusw_line_q(z, line_cells, a, e, hi, slope, shared) < target; ++it) {{
            lo = hi;
            hi = zmin + 2.0 * (hi - zmin);
        }}
        for (int it = 0; it < 64; ++it) {{
            double mid = 0.5 * (lo + hi);
            if (fusw_line_q(z, line_cells, a, e, mid, slope, shared) < target) lo = mid; else hi = mid;
        }}
        eta = hi;
        total = fusw_line_q(z, line_cells, a, e, eta, slope, shared);
    }}
    for (int f = a + (int)threadIdx.x; f < e; f += {LINE_BLOCK}) {{
        int i = line_cells[f];
        double d = fmax(eta - (double)z[i], 0.0);
        double q = 0.0;
        if (q_total > 0.0)
            q = total > 0.0
                ? (double)$ctx.uniform(({real})d, ({real})slope, i)$ * q_total / (total * dx)
                : q_total / ((double)(e - a) * dx);
        inflow_q[i] = ({real})q;
        inflow_depth[i] = ({real})d;
    }}
    if (threadIdx.x == 0) line_last[l] = q_total;
}}
'''
    return (KernelBuilder(text, domain=n_lines * LINE_BLOCK, block=LINE_BLOCK)
            .compose("grid", bundles["grid"]).compose("uniform", helpers["uniform"]).freeze())


# ----------------------------------------------------------------------------
# stage update
# ----------------------------------------------------------------------------

def _dt_text(config, real, growth):
    """dt of the step from the step-start wave speed, times the predicted
    growth (``growth``), in ``real``; thread 0 stores it."""
    c = literal(real)
    if config["time_step"] == "cfl":
        scale = f" * fmax(({real})clock[{C_GROWTH}], {c(1)})" if growth else ""
        base = f'''{real} dt = dt_max;
    if (fabs(c0 * dt_max / dxd) >= eps) dt = fmin(dt_max, cfl * dxd / (c0{scale}));'''
        error = ""
    else:
        base = f"{real} dt = ({real})$ctx.DT.get(0)$;"
        error = f"if (c0 * dt / dxd > cfl) control[{K_CFL_ERROR}] = 1u;"
    return f'''
    {real} dxd = ({real})$ctx.grid.DX.get(0)$;
    {real} cfl = ({real})$ctx.CFL.get(0)$;
    {real} eps = ({real})$ctx.EPSILON.get(0)$;
    {real} dt_max = ({real})$ctx.DT_MAX.get(0)$;
    if (!(dt_max > {c(0)})) dt_max = cfl * dxd;
    {real} c0 = ({real})__uint_as_float(control[{K_CMAX0}]);
    {base}
    if (i == 0) {{ {error} }}
    double t_end = clock[{C_T_END}];
    if (t_end > 0.0) dt = fmax(fmin(dt, ({real})(t_end - clock[{C_TIME}])), {c(0)});
    if (i == 0) clock[{C_DT}] = (double)dt;'''


def _local_dt_text(config, real, kind, src):
    """Per-cell time step when the program declares ``time_stepping``
    "local": dt_i = min(cfl dx / c_i, LOCAL_DT_RATIO dt), at least dt (the
    global step), c_i the largest max(|u|, |v|) + sqrt(g h) over the cell and
    its neighbours at the step start; the first (or euler) stage stores it in
    ``dt_local``, the second stage reads it. Returns the extra kernel
    arguments and the text setting dt (both empty otherwise)."""
    if config.get("time_stepping", "global") != "local":
        return "", ""
    c = literal(real)
    if kind == "second":
        return f", const {real}* dt_local", "\n        dt = dt_local[i];"
    sh, su, sv = src
    return f", {real}* dt_local", f'''
        {{
            {real} g = ({real})$ctx.GRAVITY.get(0)$;
            {real} cl = {c(0)};
            for (int k = -1; k < 4; ++k) {{
                int j = k < 0 ? i : $ctx.grid.neighbour(i, k)$;
                if (j < 0) continue;
                {real} cw = sqrt(g * fmax({sh}[j], {c(0)}));
                cl = fmax(cl, fmax(fabs({su}[j]), fabs({sv}[j])) + cw);
            }}
            {real} cap = dt * ({real})$ctx.LOCAL_DT_RATIO.get(0)$;
            {real} dl = cl > {c(0)}
                ? fmin(({real})$ctx.CFL.get(0)$ * dx / cl, cap) : cap;
            dt = fmax(dt, dl);
            dt_local[i] = dt;
        }}'''


def _stage_kernel(bundles, config, helpers, kind, gated):
    nx, ny, n, px, py = _geometry(config)
    real = _real(config)
    c = literal(real)
    order = int(config["order"])
    infiltrate = config["infiltration"] != "none"
    src, dst = {"euler": (STATE, STATE), "first": (STATE, STAGE_STATE),
                "second": (STAGE_STATE, STATE)}[kind]
    sh, su, sv = src
    dh, du, dv = dst
    weight = "1.0" if kind == "euler" else "0.5"
    sink = BOUNDARY_CODES["sink"]

    state_args = (f"{real}* h, {real}* u, {real}* v" if kind == "euler" else
                  f"const {real}* {sh}, const {real}* {su}, const {real}* {sv}, "
                  f"{real}* {dh}, {real}* {du}, {real}* {dv}")
    infiltration_args = ""
    infiltration_text = ""
    if infiltrate:
        infiltration_args = {
            "euler": f", {real}* infiltrated",
            "first": f", const {real}* infiltrated, {real}* infiltrated_stage",
            "second": f", {real}* infiltrated, const {real}* infiltrated_stage",
        }[kind]
        base = "infiltrated_stage[i]" if kind == "second" else "infiltrated[i]"
        store = {
            "euler": "infiltrated[i] = base + inf;",
            "first": "infiltrated_stage[i] = base + inf;",
            "second": f"infiltrated[i] = {c(0.5)} * (infiltrated[i] + base + inf);",
        }[kind]
        infiltration_text = f'''
        {real} base = {base};
        {real} inf = $ctx.infiltration(hn, base, dt, i)$;
        hn -= inf;
        vol_inf += {weight} * (double)inf * (double)area;
        {store}'''

    state = f"{sh}, {su}, {sv}, z, boundary_type, boundary_faces"
    face_bed_args = f", const {real}* zx, const {real}* zy" if order == 2 else ""
    if order == 2:
        reconstruction = f'''
        {real} h1l, h1r, h2l, h2r, zl1, zr1, zl2, zr2, r0, r1, r2, r3;
        $ctx.axis({state}, i, 0, dx, factor, &h1l, &h1r, &r0, &r1, &r2, &r3, &zl1, &zr1)$;
        {real} dzc1 = zr1 - zl1;
        $ctx.axis({state}, i, 1, dx, factor, &h2l, &h2r, &r0, &r1, &r2, &r3, &zl2, &zr2)$;
        {real} dzc2 = zr2 - zl2;'''
    else:
        reconstruction = f'''
        {real} h1l = hi, h1r = hi, h2l = hi, h2r = hi;
        {real} zl1 = z[i], zr1 = zl1, zl2 = zl1, zr2 = zl1;
        {real} dzc1 = {c(0)}, dzc2 = {c(0)};'''

    def hydrostatic(name, k, face, own_h, own_z):
        """Depth ``name`` of the cell on its face k: its own depth cut by the
        bed step to the face bed zf = max(z_l, z_r), zf - own_z being max(dz,
        0) of ``flux`` exactly (order 1: zf from z of the neighbour or the
        ghost bed; order 2: the stored ``face``)."""
        inflow = (f'''
            if (j < 0 && boundary_type[i] == {BOUNDARY_CODES["inflow"]} && ((boundary_faces[i] >> {k}) & 1))
                {name} = {own_h};''' if _lines(config) > 0 else "")
        if order == 2:
            bed = f"{real} zf = {face};"
        else:
            bed = f'''{real} zf, r0, r1, r2;
            if (j >= 0) zf = z[j];
            else $ctx.side({state}, i, {k}, dx, factor, &r0, &r1, &r2, &zf)$;
            zf = fmax(zf, {own_z});'''
        return f'''
        {real} {name};
        {{
            int j = $ctx.grid.neighbour(i, {k})$;
            {bed}
            {name} = fmax({own_h} - (zf - {own_z}), {c(0)});{inflow}
        }}'''

    reconstruction += (hydrostatic("hwl", 1, "zx[fxw]", "h1l", "zl1")
                       + hydrostatic("hwr", 2, "zx[fxe]", "h1r", "zr1")
                       + hydrostatic("hnl", 0, "zy[fyn]", "h2l", "zl2")
                       + hydrostatic("hsr", 3, "zy[fys]", "h2r", "zr2"))

    average = ""
    if kind == "second":
        average = f'''
        {{
            {real} ho = {dh}[i], uo = {du}[i], vo = {dv}[i];
            {real} mean = {c(0.5)} * (ho + hn);
            if (mean >= he) {{
                {real} qa = {c(0.5)} * (ho * uo + hn * un);
                {real} qb = {c(0.5)} * (ho * vo + hn * vn);
                un = qa / mean; vn = qb / mean; hn = mean;
                if (fabs(un) <= ve) un = {c(0)};
                if (fabs(vn) <= ve) vn = {c(0)};
            }} else {{
                vol_thr += (double)mean * (double)area;
                hn = {c(0)}; un = {c(0)}; vn = {c(0)};
            }}
        }}'''
    sink_text = "" if kind == "first" else f'''
        if (boundary_type[i] == {sink}) {{
            atomicAdd(&budget_step[{B_SINK}], (double)hn * (double)area);
            hn = {c(0)}; un = {c(0)}; vn = {c(0)};
        }}'''

    gate = f"if (control[{K_REDO}] == 0u) return;" if gated else ""
    dt_text = (f"\n    {real} dt = ({real})clock[{C_DT}];"
               if gated or kind == "second" else _dt_text(config, real, kind == "first"))
    local_args, local_text = _local_dt_text(config, real, kind, src)
    reduce_inf = f'''
    if (__any_sync(0xffffffffu, vol_inf != 0.0)) {{
        vol_inf = fusw_warp_sum(vol_inf);
        if ((threadIdx.x & 31) == 0 && vol_inf != 0.0) atomicAdd(&budget_step[{B_INFILTRATION}], vol_inf);
    }}''' if infiltrate else ""
    east = f"(x == {nx - 1} ? y * {nx + 1} : fxw + 1)" if px else "fxw + 1"
    south = f"(y == {ny - 1} ? x : fyn + {nx})" if py else f"fyn + {nx}"

    text = warp_reduce(real) + f'''
extern "C" __global__ void fusw_stage(
        {state_args}, const {real}* z, const unsigned char* boundary_type,
        const unsigned char* boundary_faces,
        const {real}* fx1, const {real}* fx2, const {real}* fx3,
        const {real}* fy1, const {real}* fy2, const {real}* fy3{face_bed_args}{infiltration_args},
        double* clock, unsigned int* control, double* budget_step{local_args}) {{
    {gate}
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double vol_inf = 0.0, vol_thr = 0.0;{dt_text}
    if (i < {n}) {{
      if (!$ctx.grid.is_active(i)$) {{
        {dh}[i] = {c(0)}; {du}[i] = {c(0)}; {dv}[i] = {c(0)};
      }} else {{
        int x = i % {nx};
        int y = i / {nx};
        {real} dx = ({real})$ctx.grid.DX.get(0)$;
        {real} area = dx * dx;{local_text}
        {real} tx = dt / dx;
        {real} he = ({real})$ctx.HE_CA.get(0)$;
        {real} ve = ({real})$ctx.VE_CA.get(0)$;
        {real} gd = {c(0.5)} * ({real})$ctx.GRAVITY.get(0)$;
        {real} factor = ({real})clock[{C_INFLOW}];
        int fxw = y * {nx + 1} + x;
        int fxe = {east};
        int fyn = y * {nx} + x;
        int fys = {south};
        {real} hi = {sh}[i], ui = {su}[i], vi = {sv}[i];
        {real} rain = ({real})$ctx.RAIN.get(i)$ * ({real})clock[{C_RAIN}];
        {real} source = ({real})$ctx.SOURCE.get(i)$ * ({real})clock[{C_SOURCE}] / area;
        {real} hn = hi - tx * (fx1[fxe] - fx1[fxw]) - tx * (fy1[fys] - fy1[fyn])
                  + (rain + source) * dt;{infiltration_text}{reconstruction}
        {real} qx = hi * ui - tx * (fx2[fxe] - fx2[fxw]
            + gd * ((hwl - h1l) * (hwl + h1l) + (h1r - hwr) * (h1r + hwr) + (h1l + h1r) * dzc1))
            - tx * (fy2[fys] - fy2[fyn]);
        {real} qy = hi * vi - tx * (fx3[fxe] - fx3[fxw])
            - tx * (fy3[fys] - fy3[fyn]
            + gd * ((hnl - h2l) * (hnl + h2l) + (h2r - hsr) * (h2r + hsr) + (h2l + h2r) * dzc2));
        {real} den = $ctx.friction(hn, sqrt(ui * ui + vi * vi), dt, i)$;
        qx /= den;
        qy /= den;
        {real} un = {c(0)}, vn = {c(0)};
        if (hn > he) {{ un = qx / hn; vn = qy / hn; }}
        if (hn <= he) {{
            vol_thr += {weight} * (double)hn * (double)area;
            hn = {c(0)}; un = {c(0)}; vn = {c(0)};
        }}
        if (fabs(un) <= ve) un = {c(0)};
        if (fabs(vn) <= ve) vn = {c(0)};
        for (int k = 0; k < 4; ++k) {{
            if ($ctx.grid.neighbour(i, k)$ >= 0) continue;
            int code = ((boundary_faces[i] >> k) & 1) ? (int)boundary_type[i] : 0;
            double s = (k == 2 || k == 3) ? 1.0 : -1.0;
            {real} f1, fn, ft, hside, hhyd;
            if (k == 1) {{ f1 = fx1[fxw]; fn = fx2[fxw]; ft = fx3[fxw]; hside = h1l; hhyd = hwl; }}
            else if (k == 2) {{ f1 = fx1[fxe]; fn = fx2[fxe]; ft = fx3[fxe]; hside = h1r; hhyd = hwr; }}
            else if (k == 0) {{ f1 = fy1[fyn]; fn = fy3[fyn]; ft = fy2[fyn]; hside = h2l; hhyd = hnl; }}
            else {{ f1 = fy1[fys]; fn = fy3[fys]; ft = fy2[fys]; hside = h2r; hhyd = hsr; }}
            double scale = {weight} * (double)dt * (double)dx;
            double* bb = budget_step + {B_FACE} + {N_FACE_TERMS} * code;
            atomicAdd(bb, scale * s * (double)f1);
            atomicAdd(bb + 1, scale * (double)fn);
            atomicAdd(bb + 2, scale * s * (double)ft);
            atomicAdd(bb + 3, scale * (double)gd * (double)(hside * hside - hhyd * hhyd));
        }}{average}{sink_text}
        {dh}[i] = hn; {du}[i] = un; {dv}[i] = vn;
      }}
    }}
{reduce_inf}
    if (__any_sync(0xffffffffu, vol_thr != 0.0)) {{
        vol_thr = fusw_warp_sum(vol_thr);
        if ((threadIdx.x & 31) == 0 && vol_thr != 0.0) atomicAdd(&budget_step[{B_THRESHOLD}], vol_thr);
    }}
}}
'''
    names = ("friction", "infiltration") + (("axis",) if order == 2 else ("side",))
    return _compose(KernelBuilder(text, domain=n), bundles, helpers, names)


# ----------------------------------------------------------------------------
# control kernels
# ----------------------------------------------------------------------------

def _check_kernel(bundles, config, again):
    real = _real(config)
    gate = f"if (control[{K_REDO}] == 0u) return;" if again else ""
    if config["time_step"] == "cfl":
        body = f'''
    {real} dt2 = dt1;
    if (fabs(c1 * dt1 / dxd) >= eps) dt2 = fmin(dt1, cfl * dxd / c1);
    if (dt2 < dt1 && control[{K_RESTARTS}] < {int(config["max_restarts"])}u) {{
        clock[{C_DT}] = (double)dt2;
        control[{K_REDO}] = 1u;
        control[{K_RESTARTS}] += 1u;
        control[{K_CMAX1}] = 0u;
        for (int b = 0; b < {N_BUDGET}; ++b) budget_step[b] = 0.0;
    }} else {{
        control[{K_REDO}] = 0u;
        {real} c0 = ({real})__uint_as_float(control[{K_CMAX0}]);
        clock[{C_GROWTH}] = c0 > {literal(real)(0)} && c1 > c0 ? (double)(c1 / c0) : 1.0;
    }}'''
    else:
        body = f'''
    (void)eps;
    if (c1 * ({real})$ctx.DT.get(0)$ / dxd > cfl) control[{K_CFL_ERROR}] = 1u;
    control[{K_REDO}] = 0u;'''
    text = f'''
extern "C" __global__ void fusw_check(double* clock, unsigned int* control, double* budget_step) {{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    {gate}
    {real} dxd = ({real})$ctx.grid.DX.get(0)$;
    {real} cfl = ({real})$ctx.CFL.get(0)$;
    {real} eps = ({real})$ctx.EPSILON.get(0)$;
    {real} c1 = ({real})__uint_as_float(control[{K_CMAX1}]);
    {real} dt1 = ({real})clock[{C_DT}];
    (void)dt1;{body}
}}
'''
    return KernelBuilder(text, domain=1).compose("grid", bundles["grid"]).freeze()


def _series_text(config):
    linear = "1" if config["series_interp"] == "linear" else "0"
    return f'''
__device__ double fusw_series(const double* table, int n, double t) {{
    if (n <= 0) return 1.0;
    if (t <= table[0]) return table[1];
    for (int k = 1; k < n; ++k) {{
        if (t < table[2 * k]) {{
            if ({linear}) {{
                double w = (t - table[2 * k - 2]) / (table[2 * k] - table[2 * k - 2]);
                return table[2 * k - 1] + w * (table[2 * k + 1] - table[2 * k - 1]);
            }}
            return table[2 * k - 1];
        }}
    }}
    return table[2 * n - 1];
}}
'''


def _factors_text(config):
    return f'''
    double t = clock[{C_TIME}];
    clock[{C_RAIN}] = fusw_series(rain_table, {int(config["rain_series"])}, t);
    clock[{C_INFLOW}] = fusw_series(inflow_table, {int(config["inflow_series"])}, t);
    clock[{C_SOURCE}] = fusw_series(source_table, {int(config["source_series"])}, t);'''


_SERIES_ARGS = "const double* rain_table, const double* inflow_table, const double* source_table"


FINALIZE_BLOCK = 32 * ((N_BUDGET + 31) // 32)


def _finalize_kernel(config):
    text = _series_text(config) + f'''
extern "C" __global__ void fusw_finalize(
        double* clock, unsigned int* control, double* budget, double* budget_step,
        {_SERIES_ARGS}) {{
    int b = threadIdx.x;
    if (b < {N_BUDGET}) {{
        double add = budget_step[b];
        if (b == {B_RAIN}) add += clock[{C_DT}] * clock[{C_RAIN}] * clock[{C_RAIN_RATE}];
        if (b == {B_SOURCE}) add += clock[{C_DT}] * clock[{C_SOURCE}] * clock[{C_SOURCE_RATE}];
        budget[b] += add;
        budget_step[b] = 0.0;
    }}
    __syncthreads();
    if (b != 0) return;
    clock[{C_TIME}] += clock[{C_DT}];
    control[{K_STEP}] += 1u;
    control[{K_TOTAL_RESTARTS}] += control[{K_RESTARTS}];
    control[{K_RESTARTS}] = 0u;
    control[{K_REDO}] = 0u;
    control[{K_CMAX0}] = 0u;
    control[{K_CMAX1}] = 0u;{_factors_text(config)}
}}
'''
    return KernelBuilder(text, domain=FINALIZE_BLOCK, block=FINALIZE_BLOCK).freeze()


def _refresh_kernel(config):
    text = _series_text(config) + f'''
extern "C" __global__ void fusw_refresh(double* clock, unsigned int* control, {_SERIES_ARGS}) {{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    control[{K_CMAX0}] = 0u;
    control[{K_CMAX1}] = 0u;
    control[{K_REDO}] = 0u;
    control[{K_RESTARTS}] = 0u;
    clock[{C_RAIN_RATE}] = 0.0;
    clock[{C_SOURCE_RATE}] = 0.0;{_factors_text(config)}
}}
'''
    return KernelBuilder(text, domain=1).freeze()


def record_width(config):
    return 2 + 3 * int(config["n_probes"]) + N_BUDGET


def _record_kernel(config):
    real = _real(config)
    n_probes = int(config["n_probes"])
    capacity = int(config["record_capacity"])
    width = record_width(config)
    text = f'''
extern "C" __global__ void fusw_record(
        const {real}* h, const {real}* u, const {real}* v, const int* probe_index,
        const double* clock, unsigned int* control, const double* budget, double* records) {{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t > {n_probes}) return;
    int every = $ctx.RECORD_INTERVAL.get(0)$;
    unsigned int step = control[{K_STEP}];
    if (every <= 0 || step % (unsigned int)every != 0u) return;
    long long row = (long long)(step / (unsigned int)every) - (long long)control[{K_REC_BASE}] - 1;
    if (row < 0) return;
    if (row >= {capacity}) {{
        if (t == 0) control[{K_REC_OVERFLOW}] = 1u;
        return;
    }}
    double* r = records + row * {width};
    if (t == 0) {{
        r[0] = clock[{C_TIME}];
        r[1] = clock[{C_DT}];
        for (int b = 0; b < {N_BUDGET}; ++b) r[{2 + 3 * n_probes} + b] = budget[b];
    }} else {{
        int p = t - 1;
        int j = probe_index[p];
        r[2 + p] = (double)h[j];
        r[{2 + n_probes} + p] = (double)u[j];
        r[{2 + 2 * n_probes} + p] = (double)v[j];
    }}
}}
'''
    return KernelBuilder(text, domain=n_probes + 1).freeze()


def _prepare_kernel(bundles, config):
    real = _real(config)
    c = literal(real)
    n = int(config["nx"]) * int(config["ny"])
    text = warp_reduce(real) + f'''
extern "C" __global__ void fusw_prepare({real}* h, {real}* u, {real}* v, double* clock) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double rain = 0.0, source = 0.0;
    if (i < {n}) {{
        {real} he = ({real})$ctx.HE_CA.get(0)$;
        {real} ve = ({real})$ctx.VE_CA.get(0)$;
        bool act = $ctx.grid.is_active(i)$;
        if (!act || h[i] <= he) {{
            h[i] = {c(0)}; u[i] = {c(0)}; v[i] = {c(0)};
        }} else {{
            if (fabs(u[i]) <= ve) u[i] = {c(0)};
            if (fabs(v[i]) <= ve) v[i] = {c(0)};
        }}
        if (act) {{
            double dx = (double)$ctx.grid.DX.get(0)$;
            rain = (double)$ctx.RAIN.get(i)$ * dx * dx;
            source = (double)$ctx.SOURCE.get(i)$;
        }}
    }}
    if (__any_sync(0xffffffffu, rain != 0.0 || source != 0.0)) {{
        rain = fusw_warp_sum(rain);
        source = fusw_warp_sum(source);
        if ((threadIdx.x & 31) == 0) {{
            if (rain != 0.0) atomicAdd(&clock[{C_RAIN_RATE}], rain);
            if (source != 0.0) atomicAdd(&clock[{C_SOURCE_RATE}], source);
        }}
    }}
}}
'''
    return KernelBuilder(text, domain=n).compose("grid", bundles["grid"]).freeze()


def _codes_kernel(bundles, config):
    n = int(config["nx"]) * int(config["ny"])
    outlet = BOUNDARY_CODES[config["outlet_type"]]
    text = f'''
extern "C" __global__ void fusw_codes(unsigned char* boundary_type) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    unsigned char code = {BOUNDARY_CODES["wall"]};
    if ($ctx.grid.is_active(i)$ && $ctx.grid.can_out(i)$) {{
        int faces = 0;
        for (int k = 0; k < 4; ++k) faces += $ctx.grid.neighbour(i, k)$ < 0 ? 1 : 0;
        code = faces > 0 ? {outlet} : {BOUNDARY_CODES["sink"]};
    }}
    boundary_type[i] = code;
}}
'''
    return KernelBuilder(text, domain=n).compose("grid", bundles["grid"]).freeze()


# ----------------------------------------------------------------------------
# sequence factories
# ----------------------------------------------------------------------------

def step_factory(be, bundles, config):
    """Routine of one time step (see the module docstring)."""
    cupy_only(be)
    helpers = _helpers(bundles, config)
    rb = RoutineBuilder()
    if _lines(config) > 0:
        rb.step("lines", _line_kernel(bundles, config, helpers))
    if int(config["order"]) == 1:
        rb.step("flux", _flux_kernel(bundles, config, helpers, STATE, K_CMAX0, False))
        rb.step("stage", _stage_kernel(bundles, config, helpers, "euler", False))
    else:
        flux_h = _flux_kernel(bundles, config, helpers, STATE, K_CMAX0, False)
        flux_hs = _flux_kernel(bundles, config, helpers, STAGE_STATE, K_CMAX1, False)
        rb.step("flux", flux_h)
        rb.step("first", _stage_kernel(bundles, config, helpers, "first", False))
        rb.step("flux_stage", flux_hs)
        rb.step("check", _check_kernel(bundles, config, False))
        if config["time_step"] == "cfl" and int(config["max_restarts"]) > 0:
            redo_flux = _flux_kernel(bundles, config, helpers, STATE, None, True)
            redo_first = _stage_kernel(bundles, config, helpers, "first", True)
            redo_flux_stage = _flux_kernel(bundles, config, helpers, STAGE_STATE, K_CMAX1, True)
            redo_check = _check_kernel(bundles, config, True)
            for r in range(int(config["max_restarts"])):
                rb.step(f"redo{r}_flux", redo_flux)
                rb.step(f"redo{r}_first", redo_first)
                rb.step(f"redo{r}_flux_stage", redo_flux_stage)
                rb.step(f"redo{r}_check", redo_check)
        rb.step("second", _stage_kernel(bundles, config, helpers, "second", False))
    rb.step("finalize", _finalize_kernel(config))
    if int(config["record_capacity"]) > 0:
        rb.step("record", _record_kernel(config))
    return rb.freeze()


def prepare_factory(be, bundles, config):
    """Evaluate the time factors at the current time and clear the step flags
    and the rain / source rates, then cut h, u, v at HE_CA / VE_CA, dry the
    closed cells and sum the rain and source rates of the active cells."""
    cupy_only(be)
    return (RoutineBuilder().step("refresh", _refresh_kernel(config))
            .step("cells", _prepare_kernel(bundles, config)).freeze())


def codes_factory(be, bundles, config):
    """Boundary codes from the grid: outlet cells take ``outlet_type`` (sink
    when they have no boundary face), every other cell wall."""
    cupy_only(be)
    return _codes_kernel(bundles, config)


def shear_factory(location, method):
    """Factory of the shear stress kernel on ``location`` (cell | face) by
    ``method`` (``_shear.py``)."""
    def factory(be, bundles, config):
        cupy_only(be)
        kernel = cell_kernel if location == "cell" else face_kernel
        return kernel(bundles, config, _helpers(bundles, config), _real(config), method)
    return factory


def diagnostic_factory(kernel):
    """Factory of one diagnostic kernel builder of ``_diagnostics.py``."""
    def factory(be, bundles, config):
        cupy_only(be)
        return kernel(bundles, config, _helpers(bundles, config), _real(config))
    return factory


def bind_plan(frozen, _be):
    from pyfastflow.flow._program_cupy import _grid_leaf_plan

    values = {name: name for name in (
        STATE + STAGE_STATE + FACES
        + ("z", "boundary_type", "infiltrated", "infiltrated_stage", "clock",
           "boundary_faces", "control", "budget", "budget_step", "rain_table", "inflow_table",
           "source_table", "probe_index", "records", "line_cells", "line_offsets",
           "line_q", "line_slope", "line_last", "inflow_q", "inflow_depth",
           "dt_local")
        + CELL_OUT + FACE_OUT + FROUDE_CELL + FROUDE_FACE + ACC_IN + ACC_CELL + ACC_FACE)}
    values.update(PARAM_BIND)
    return _grid_leaf_plan(frozen, values)
