"""FullSWOF finite-volume shallow-water solver on a D4 quadtree (CuPy only).

The scheme of ``pyfastflow.flood.fusw`` with its Riemann fluxes, limiters and
friction laws: order 1, or MUSCL with Heun; hydrostatic reconstruction; a CFL
time step with the wave-speed growth and restarts. State h, u, v per leaf.

A side of a leaf (0 north, 1 west, 2 east, 3 south) holds the slots ``2 side``
and ``2 side + 1``: one neighbour of the same size or coarser, or two finer
ones. Each link is one face of length ``min(dx_i, dx_j)``:

  - ``slopes``: limited differences per axis. With d the distance between
    centres and two finer neighbours averaged, the one-sided differences are
    (h_i - h_j) dx_i / d, so a uniform tree gives the raster formula;
  - ``rhs``: fluxes, pressure and bed terms per leaf. Both leaves of a link
    evaluate its flux at the one call site, west or north side as left, so they
    get the same bits. Each subface carries its own hydrostatic cut;
  - ``dt``: min(dt_max, cfl / (growth max(c / dx_i))), over every leaf's links;
  - ``euler`` or ``first`` + ``second``: the update, with semi-implicit friction.

A side without a neighbour is a domain edge: on an outlet leaf the ``outlet_type``
ghost state of :mod:`pyfastflow.flood.fusw._boundary` (a sink absorbs the leaf's
water and its faces act as walls), a wall otherwise. Rain is uniform.

``outlet_type="neumann"`` copies the leaf depth above a ghost bed extrapolated
from the terrain: where the bed rises towards the edge this raises the ghost
water surface and feeds water into the domain. Prefer ``normal_depth`` or
``sink`` on a real DEM.

Author: B.G (11/2026)
"""

import cupy as cp
import numpy as np

from pyfastflow.core import KernelBuilder, RoutineBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flood.fusw._boundary import make_ghost
from pyfastflow.flood.fusw._common import BOUNDARY_CODES, OUTLET_TYPES, PRECISIONS, literal, warp_reduce
from pyfastflow.flood.fusw._friction import FRICTIONS, make_friction
from pyfastflow.flood.fusw._limiters import LIMITERS, make_limiter
from pyfastflow.flood.fusw._riemann import FLUXES, make_flux
from pyfastflow.flow._program_cupy import _grid_leaf_plan
from pyfastflow.quadtree import DynamicQuadtree, transfer

SHAPE = (Dim("ny"), Dim("nx"))
STATE = ("h", "u", "v")
STAGE_STATE = ("hs", "us", "vs")
SLOPES = ("sx_h", "sx_z", "sx_n", "sx_t", "sy_h", "sy_z", "sy_n", "sy_t")
RATES = ("rh", "rx", "ry")
CRITERIA = ("h", "speed", "u", "v")

C_TIME, C_DT, C_GROWTH = range(3)
K_RATE0, K_RATE1, K_REDO, K_RESTARTS, K_STEP = range(5)
S_OUT0, S_OUT1, S_THRESHOLD, S_SINK = range(4)
B_OUT, B_THRESHOLD = range(2)

BIND = {"GRAVITY": "gravity", "HE_CA": "he_ca", "VE_CA": "ve_ca", "EPSILON": "epsilon",
        "FRICTION": "friction_coefficient", "RAIN": "rain", "CFL": "cfl", "DT_MAX": "dt_max",
        "STAGE": "stage", "INFLOW": "inflow", "OUTLET_SLOPE": "outlet_slope",
        "OUTLET_MANNING": "outlet_manning", "RATING_A": "rating_a", "RATING_B": "rating_b",
        "RATING_ETA0": "rating_eta0", "NEWTON_ITERATIONS": "newton_iterations"}


def _real(config):
    return PRECISIONS[config["precision"]]


def _dtype(config):
    return config["precision"]


def _extent(config):
    return int(config["nx"]) * int(config["ny"])


def _order(config):
    return int(config["order"])


def _plan(frozen, _be):
    names = STATE + STAGE_STATE + SLOPES + RATES + ("z", "clock", "control", "step_budget", "budget")
    return _grid_leaf_plan(frozen, {**{name: name for name in names}, **BIND})


def _gate(gated):
    return f"if (control[{K_REDO}] == 0u) return;" if gated else ""


def _neighbours_text(config):
    """Mean state and distance of the neighbours of a side, and the state beyond an edge."""
    real = _real(config)
    c = literal(real)
    code = BOUNDARY_CODES[config["outlet_type"]]
    return f'''
__device__ int tf_mean(const {real}* h, const {real}* u, const {real}* v, const {real}* z,
                       int i, int sd, {real}* hs, {real}* un, {real}* ut, {real}* zs, {real}* d) {{
    bool xa = (sd == 1 || sd == 2);
    {real} sh = {c(0)}, sn = {c(0)}, st = {c(0)}, sz = {c(0)}, sd2 = {c(0)};
    int count = 0;
    for (int t = 0; t < 2; ++t) {{
        int j = $ctx.grid.neighbour_raw(i, 2 * sd + t)$;
        if (j < 0) continue;
        sh += h[j];
        sn += xa ? u[j] : v[j];
        st += xa ? v[j] : u[j];
        sz += z[j];
        sd2 += ({real})$ctx.grid.dx(j)$;
        ++count;
    }}
    if (count == 0) return 0;
    {real} w = {c(1)} / ({real})count;
    *hs = sh * w; *un = sn * w; *ut = st * w; *zs = sz * w;
    *d = {c(0.5)} * (({real})$ctx.grid.dx(i)$ + sd2 * w);
    return count;
}}

__device__ void tf_ghost(const {real}* h, const {real}* u, const {real}* v, const {real}* z,
                         int i, int sd, {real}* hs, {real}* un, {real}* ut, {real}* zs) {{
    bool xa = (sd == 1 || sd == 2);
    {real} zc = z[i], dx = ({real})$ctx.grid.dx(i)$, zopp = zc;
    {real} ho, no, to, zo, d;
    if (tf_mean(h, u, v, z, i, 3 - sd, &ho, &no, &to, &zo, &d) > 0) zopp = zc - (zc - zo) * dx / d;
    int edge = $ctx.grid.can_out(i)$ ? {code} : 0;
    $ctx.ghost(edge, i, sd >= 2 ? 1 : -1, h[i],
               xa ? u[i] : v[i], xa ? v[i] : u[i], zc, zopp, dx, {c(1)}, 1, hs, un, ut, zs)$;
}}
'''


def _ghost(config):
    return make_ghost(_real(config), "discharge")


def _slopes_kernel(bundles, config, src, gated):
    real, c = _real(config), literal(_real(config))
    h, u, v = src
    out = ", ".join(f"{real}* {name}" for name in SLOPES)

    def axis(left, right, sh, sz, sn, st):
        return f'''
        {{
            {real} h0, n0, t0, z0, d0, h2, n2, t2, z2, d2;
            if (tf_mean({h}, {u}, {v}, z, i, {left}, &h0, &n0, &t0, &z0, &d0) == 0) {{
                tf_ghost({h}, {u}, {v}, z, i, {left}, &h0, &n0, &t0, &z0);
                d0 = dx;
            }}
            if (tf_mean({h}, {u}, {v}, z, i, {right}, &h2, &n2, &t2, &z2, &d2) == 0) {{
                tf_ghost({h}, {u}, {v}, z, i, {right}, &h2, &n2, &t2, &z2);
                d2 = dx;
            }}
            {real} a = dx / d0, b = dx / d2;
            {real} n1 = {"u1" if left == 1 else "v1"}, t1 = {"v1" if left == 1 else "u1"};
            {sh}[i] = $ctx.limiter((h1 - h0) * a, (h2 - h1) * b)$;
            {sz}[i] = $ctx.limiter((h1 - h0 + z1 - z0) * a, (h2 - h1 + z2 - z1) * b)$;
            {sn}[i] = $ctx.limiter((n1 - n0) * a, (n2 - n1) * b)$;
            {st}[i] = $ctx.limiter((t1 - t0) * a, (t2 - t1) * b)$;
        }}'''

    text = _neighbours_text(config) + f'''
extern "C" __global__ void tf_slopes(const {real}* {h}, const {real}* {u}, const {real}* {v},
        const {real}* z, {out}, const unsigned int* control) {{
    {_gate(gated)}
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.grid.n_leaves()$) return;
    {real} dx = ({real})$ctx.grid.dx(i)$;
    {real} h1 = {h}[i], u1 = {u}[i], v1 = {v}[i], z1 = z[i];
    {axis(1, 2, "sx_h", "sx_z", "sx_n", "sx_t")}
    {axis(0, 3, "sy_h", "sy_z", "sy_n", "sy_t")}
}}
'''
    return (KernelBuilder(text, domain=_extent(config)).compose("grid", bundles["grid"])
            .compose("limiter", make_limiter(config["limiter"], real))
            .compose("ghost", _ghost(config)).freeze())


def _rhs_kernel(bundles, config, src, rate_slot, out_slot, gated):
    real, c = _real(config), literal(_real(config))
    h, u, v = src
    second = _order(config) == 2
    slopes_args = "".join(f", const {real}* {name}" for name in SLOPES) if second else ""
    slopes_pass = "".join(f", {name}" for name in SLOPES) if second else ""
    if second:
        recon = f'''
    bool right = sd >= 2, xa = (sd == 1 || sd == 2);
    {real} h1 = h[i], n1 = xa ? u[i] : v[i], t1 = xa ? v[i] : u[i], z1 = z[i];
    {real} dh = xa ? sx_h[i] : sy_h[i], dhz = xa ? sx_z[i] : sy_z[i];
    {real} dn = xa ? sx_n[i] : sy_n[i], dt = xa ? sx_t[i] : sy_t[i];
    {real} hr = h1 + dh * {c(0.5)}, hl = h1 - dh * {c(0.5)};
    *hs = right ? hr : hl;
    *zs = right ? z1 + {c(0.5)} * (dhz - dh) : z1 + {c(0.5)} * (dh - dhz);
    if (h1 > {c(0)}) {{
        *un = right ? n1 + hl * dn * {c(0.5)} / h1 : n1 - hr * dn * {c(0.5)} / h1;
        *ut = right ? t1 + hl * dt * {c(0.5)} / h1 : t1 - hr * dt * {c(0.5)} / h1;
    }} else {{
        *un = right ? n1 + dn * {c(0.5)} : n1 - dn * {c(0.5)};
        *ut = right ? t1 + dt * {c(0.5)} : t1 - dt * {c(0.5)};
    }}'''
        bed_x = f"gd * {c(2)} * {h}[i] * (sx_z[i] - sx_h[i])"
        bed_y = f"gd * {c(2)} * {h}[i] * (sy_z[i] - sy_h[i])"
    else:
        recon = f'''
    bool xa = (sd == 1 || sd == 2);
    *hs = h[i]; *un = xa ? u[i] : v[i]; *ut = xa ? v[i] : u[i]; *zs = z[i];'''
        bed_x = bed_y = c(0)
    reduce = (f'''
    rate = fusw_warp_max(rate);
    if ((threadIdx.x & 31) == 0 && rate > {c(0)})
        atomicMax(&control[{rate_slot}], __float_as_uint((float)rate));'''
              if rate_slot is not None else "")
    text = warp_reduce(real) + _neighbours_text(config) + f'''
__device__ void tf_recon(const {real}* h, const {real}* u, const {real}* v, const {real}* z{slopes_args},
                         int i, int sd, {real}* hs, {real}* un, {real}* ut, {real}* zs) {{{recon}
}}

extern "C" __global__ void tf_rhs(const {real}* {h}, const {real}* {u}, const {real}* {v}, const {real}* z{slopes_args},
        {real}* rh, {real}* rx, {real}* ry, unsigned int* control, double* step_budget) {{
    {_gate(gated)}
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    {real} rate = {c(0)};
    if (i < $ctx.grid.n_leaves()$) {{
        {real} dx = ({real})$ctx.grid.dx(i)$;
        {real} gd = {c(0.5)} * ({real})$ctx.GRAVITY.get(0)$;
        {real} mass = {c(0)}, mx = {c(0)}, my = {c(0)};
        double out = 0.0;
        for (int sd = 0; sd < 4; ++sd) {{
            bool right = sd >= 2, xa = (sd == 1 || sd == 2);
            {real} s = right ? {c(1)} : {c(-1)};
            {real} ho, no, to, zo;
            tf_recon({h}, {u}, {v}, z{slopes_pass}, i, sd, &ho, &no, &to, &zo);
            int filled = 0;
            for (int t = 0; t < 2; ++t) {{
                int j = $ctx.grid.neighbour_raw(i, 2 * sd + t)$;
                if (j < 0 && (t == 0 || filled > 0)) continue;
                {real} hb, nb, tb, zb, width = dx;
                if (j >= 0) {{
                    ++filled;
                    tf_recon({h}, {u}, {v}, z{slopes_pass}, j, 3 - sd, &hb, &nb, &tb, &zb);
                    width = fmin(dx, ({real})$ctx.grid.dx(j)$);
                }} else {{
                    tf_ghost({h}, {u}, {v}, z, i, sd, &hb, &nb, &tb, &zb);
                }}
                {real} hL = right ? ho : hb, nL = right ? no : nb, tL = right ? to : tb, zL = right ? zo : zb;
                {real} hR = right ? hb : ho, nR = right ? nb : no, tR = right ? tb : to, zR = right ? zb : zo;
                {real} dz = zR - zL;
                {real} hl = fmax(hL - fmax(dz, {c(0)}), {c(0)});
                {real} hr = fmax(hR - fmax(-dz, {c(0)}), {c(0)});
                {real} f1, fn, ft, cw;
                $ctx.flux(hl, nL, tL, hr, nR, tR, &f1, &fn, &ft, &cw)$;
                {real} hy = right ? hl : hr;
                {real} w = width / dx;
                {real} normal = w * s * (fn + gd * (ho - hy) * (ho + hy));
                {real} tangent = w * s * ft;
                mass += w * s * f1;
                if (xa) {{ mx += normal; my += tangent; }} else {{ my += normal; mx += tangent; }}
                rate = fmax(rate, cw / dx);
                if (j < 0) out += (double)width * (double)s * (double)f1;
            }}
        }}
        rh[i] = -mass / dx;
        rx[i] = -(mx + {bed_x}) / dx;
        ry[i] = -(my + {bed_y}) / dx;
        if (out != 0.0) atomicAdd(&step_budget[{out_slot}], out);
    }}{reduce}
}}
'''
    kb = (KernelBuilder(text, domain=_extent(config)).compose("grid", bundles["grid"])
          .compose("flux", make_flux(config["flux"], real)).compose("ghost", _ghost(config)))
    return kb.freeze()


def _dt_kernel(bundles, config):
    real, c = _real(config), literal(_real(config))
    return KernelBuilder(f'''
extern "C" __global__ void tf_dt(double* clock, const unsigned int* control) {{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    {real} rate = ({real})__uint_as_float(control[{K_RATE0}]);
    {real} cfl = ({real})$ctx.CFL.get(0)$;
    {real} top = ({real})$ctx.DT_MAX.get(0)$;
    if (top <= {c(0)}) top = cfl * ({real})$ctx.grid.DX0.get(0)$;
    {real} growth = fmax(({real})clock[{C_GROWTH}], {c(1)});
    clock[{C_DT}] = (double)(rate > {c(0)} ? fmin(top, cfl / (rate * growth)) : top);
}}
''', domain=1).compose("grid", bundles["grid"]).freeze()


def _apply_kernel(bundles, config, kind, gated):
    real, c = _real(config), literal(_real(config))
    src, dst = {"euler": (STATE, STATE), "first": (STATE, STAGE_STATE),
                "second": (STAGE_STATE, STATE)}[kind]
    sh, su, sv = src
    dh, du, dv = dst
    weight = "1.0" if kind == "euler" else "0.5"
    state = (f"{real}* h, {real}* u, {real}* v" if kind == "euler" else
             f"const {real}* {sh}, const {real}* {su}, const {real}* {sv}, {real}* {dh}, {real}* {du}, {real}* {dv}")
    average = "" if kind != "second" else f'''
    {real} ho = {dh}[i], uo = {du}[i], vo = {dv}[i];
    {real} mean = {c(0.5)} * (ho + hn);
    if (mean >= he) {{
        {real} qa = {c(0.5)} * (ho * uo + hn * un);
        {real} qb = {c(0.5)} * (ho * vo + hn * vn);
        un = qa / mean; vn = qb / mean; hn = mean;
        if (fabs(un) <= ve) un = {c(0)};
        if (fabs(vn) <= ve) vn = {c(0)};
    }} else {{
        cut += (double)mean * (double)dx * (double)dx;
        hn = {c(0)}; un = {c(0)}; vn = {c(0)};
    }}'''
    sink = "" if kind == "first" or config["outlet_type"] != "sink" else f'''
    if ($ctx.grid.can_out(i)$) {{
        atomicAdd(&step_budget[{S_SINK}], (double)hn * (double)dx * (double)dx);
        hn = {c(0)}; un = {c(0)}; vn = {c(0)};
    }}'''
    text = f'''
extern "C" __global__ void tf_apply({state}, const {real}* rh, const {real}* rx, const {real}* ry,
        const double* clock, const unsigned int* control, double* step_budget) {{
    {_gate(gated)}
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.grid.n_leaves()$) return;
    {real} dt = ({real})clock[{C_DT}];
    {real} dx = ({real})$ctx.grid.dx(i)$;
    {real} he = ({real})$ctx.HE_CA.get(0)$, ve = ({real})$ctx.VE_CA.get(0)$;
    {real} hi = {sh}[i], ui = {su}[i], vi = {sv}[i];
    {real} hn = hi + dt * (rh[i] + ({real})$ctx.RAIN.get(0)$);
    {real} qx = hi * ui + dt * rx[i];
    {real} qy = hi * vi + dt * ry[i];
    {real} den = $ctx.friction(hn, sqrt(ui * ui + vi * vi), dt, i)$;
    qx /= den;
    qy /= den;
    {real} un = {c(0)}, vn = {c(0)};
    double cut = 0.0;
    if (hn > he) {{ un = qx / hn; vn = qy / hn; }}
    else {{
        cut += {weight} * (double)hn * (double)dx * (double)dx;
        hn = {c(0)}; un = {c(0)}; vn = {c(0)};
    }}
    if (fabs(un) <= ve) un = {c(0)};
    if (fabs(vn) <= ve) vn = {c(0)};{average}{sink}
    if (cut != 0.0) atomicAdd(&step_budget[{S_THRESHOLD}], cut);
    {dh}[i] = hn; {du}[i] = un; {dv}[i] = vn;
}}
'''
    return (KernelBuilder(text, domain=_extent(config)).compose("grid", bundles["grid"])
            .compose("friction", make_friction(config["friction"], real)).freeze())


def _check_kernel(config, gated):
    real, c = _real(config), literal(_real(config))
    return KernelBuilder(f'''
extern "C" __global__ void tf_check(double* clock, unsigned int* control, double* step_budget) {{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    {_gate(gated)}
    {real} c0 = ({real})__uint_as_float(control[{K_RATE0}]);
    {real} c1 = ({real})__uint_as_float(control[{K_RATE1}]);
    {real} dt1 = ({real})clock[{C_DT}];
    {real} cfl = ({real})$ctx.CFL.get(0)$;
    {real} eps = ({real})$ctx.EPSILON.get(0)$;
    {real} dt2 = dt1;
    if (fabs(c1 * dt1) >= eps) dt2 = fmin(dt1, cfl / c1);
    if (dt2 < dt1 && control[{K_RESTARTS}] < {int(config["max_restarts"])}u) {{
        clock[{C_DT}] = (double)dt2;
        control[{K_REDO}] = 1u;
        control[{K_RESTARTS}] += 1u;
        control[{K_RATE1}] = 0u;
        for (int b = 0; b < 4; ++b) step_budget[b] = 0.0;
    }} else {{
        control[{K_REDO}] = 0u;
        clock[{C_GROWTH}] = c0 > {c(0)} && c1 > c0 ? (double)(c1 / c0) : 1.0;
    }}
}}
''', domain=1).freeze()


def _finalize_kernel(config):
    w0, w1 = ("1.0", "0.0") if _order(config) == 1 else ("0.5", "0.5")
    return KernelBuilder(f'''
extern "C" __global__ void tf_finalize(double* clock, unsigned int* control, double* step_budget, double* budget) {{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    double dt = clock[{C_DT}];
    budget[{B_OUT}] += dt * ({w0} * step_budget[{S_OUT0}] + {w1} * step_budget[{S_OUT1}]) + step_budget[{S_SINK}];
    budget[{B_THRESHOLD}] += step_budget[{S_THRESHOLD}];
    for (int b = 0; b < 4; ++b) step_budget[b] = 0.0;
    clock[{C_TIME}] += dt;
    control[{K_STEP}] += 1u;
    control[{K_RATE0}] = 0u;
    control[{K_RATE1}] = 0u;
    control[{K_RESTARTS}] = 0u;
    control[{K_REDO}] = 0u;
}}
''', domain=1).freeze()


def _step_factory(be, bundles, config):
    rb = RoutineBuilder()
    if _order(config) == 1:
        rb.step("rhs", _rhs_kernel(bundles, config, STATE, K_RATE0, S_OUT0, False))
        rb.step("dt", _dt_kernel(bundles, config))
        rb.step("euler", _apply_kernel(bundles, config, "euler", False))
    else:
        rb.step("slopes", _slopes_kernel(bundles, config, STATE, False))
        rb.step("rhs", _rhs_kernel(bundles, config, STATE, K_RATE0, S_OUT0, False))
        rb.step("dt", _dt_kernel(bundles, config))
        rb.step("first", _apply_kernel(bundles, config, "first", False))
        rb.step("slopes_stage", _slopes_kernel(bundles, config, STAGE_STATE, False))
        rb.step("rhs_stage", _rhs_kernel(bundles, config, STAGE_STATE, K_RATE1, S_OUT1, False))
        rb.step("check", _check_kernel(config, False))
        for r in range(int(config["max_restarts"])):
            rb.step(f"redo{r}_slopes", _slopes_kernel(bundles, config, STATE, True))
            rb.step(f"redo{r}_rhs", _rhs_kernel(bundles, config, STATE, None, S_OUT0, True))
            rb.step(f"redo{r}_first", _apply_kernel(bundles, config, "first", True))
            rb.step(f"redo{r}_slopes_stage", _slopes_kernel(bundles, config, STAGE_STATE, True))
            rb.step(f"redo{r}_rhs_stage", _rhs_kernel(bundles, config, STAGE_STATE, K_RATE1, S_OUT1, True))
            rb.step(f"redo{r}_check", _check_kernel(config, True))
        rb.step("second", _apply_kernel(bundles, config, "second", False))
    rb.step("finalize", _finalize_kernel(config))
    return rb.freeze()


def _build_program():
    b = ProgramBuilder("TreeFuswProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("tree")
    b.config("precision", choices=tuple(PRECISIONS), default="f32")
    b.config("order", choices=(1, 2), default=2)
    b.config("limiter", choices=LIMITERS, default="minmod")
    b.config("flux", choices=FLUXES, default="hll")
    b.config("friction", choices=FRICTIONS, default="manning")
    b.config("max_restarts", default=2)
    b.config("outlet_type", choices=OUTLET_TYPES, default="normal_depth")
    b.param("friction_coefficient", "auto", "f32", value=0.033, shape=SHAPE)
    for name, value in (("rain", 0.0), ("gravity", 9.81), ("cfl", 0.5), ("dt_max", 0.0),
                        ("he_ca", 1.0e-12), ("ve_ca", 1.0e-12), ("epsilon", 1.0e-13),
                        ("stage", 0.0), ("inflow", 0.0), ("outlet_slope", 0.0), ("outlet_manning", 0.033),
                        ("rating_a", 0.0), ("rating_b", 1.5), ("rating_eta0", 0.0)):
        b.param(name, "auto", "f32", value=value)
    b.param("newton_iterations", "auto", "i32", value=100)
    second = lambda config: _order(config) == 2
    b.data("z", _dtype, SHAPE, role="input", shape_source=True)
    for name in STATE:
        b.data(name, _dtype, SHAPE, role="state")
    for name in RATES:
        b.data(name, _dtype, SHAPE, role="internal")
    for name in STAGE_STATE + SLOPES:
        b.data(name, _dtype, lambda c: (c["ny"], c["nx"]) if second(c) else (0, 0), role="internal")
    b.data("clock", "f64", (3,), role="internal")
    b.data("control", "u32", (5,), role="internal")
    b.data("step_budget", "f64", (4,), role="internal")
    b.data("budget", "f64", (2,), role="output")
    b.bundle("grid", lambda be, *, tree, **_: tree.group,
             lambda be, pool, *, tree, **_: tree.parameters, dims=("nx", "ny"), config=("tree",))
    b.add("_step", _step_factory, bind=_plan)
    return b.freeze()


TreeFuswProgram = _build_program()


class QuadtreeFusw:
    """FullSWOF solver on a tree refined from its own solution.

    Parameters
    ----------
    be : Backend
        ``cupy`` backend.
    dem : ndarray, shape (ny, nx)
        Finest-resolution bed elevation.
    dx : float
        Finest cell side.
    rain : float
        Rain rate, in m/s.
    levels : int or ndarray
        Starting uniform level, or a ``(ny, nx)`` level map.
    max_level : int, optional
        Coarsest level of the tree; defaults to ``levels``.
    criterion : {"h", "speed", "u", "v"}
        Quantity refined on, compared in absolute value.
    threshold : float
        A leaf is flagged for refinement at or above this value.
    coarsen_below : float, optional
        A leaf is flagged for coarsening below this value.
    capacity : int, optional
        Largest leaf count; defaults to half the finest raster.
    manning, cfl, dt_max, order, limiter, flux, friction, precision, max_restarts
        Solver settings, see :mod:`pyfastflow.flood.fusw`.
    outlet_mask : ndarray, optional
        Outlet cells for the tree; defaults to the domain edges.
    outlet_type : str
        Law of the outlet faces, see :mod:`pyfastflow.flood.fusw._boundary`.
    outlet_slope, outlet_manning : float
        ``normal_depth``: energy slope (<= 0: the bed slope) and Manning n.
    """

    def __init__(self, be, dem, dx, rain, *, levels=5, max_level=None, criterion="h", threshold=0.05,
                 coarsen_below=None, capacity=None, manning=0.033, cfl=0.4, dt_max=0.0, order=2,
                 limiter="minmod", flux="hll", friction="manning", precision="f32", max_restarts=2,
                 outlet_mask=None, outlet_type="normal_depth", outlet_slope=0.01, outlet_manning=0.033):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
        dem = np.asarray(dem, dtype=np.float64 if precision == "f64" else np.float32)
        self.be, self.criterion, self.threshold = be, criterion, float(threshold)
        self.coarsen_below, self.rain, self.dx = coarsen_below, float(rain), float(dx)
        self.pool = be.PoolCls()
        level_map = levels if isinstance(levels, np.ndarray) else np.full(dem.shape, levels, np.int32)
        top = int(level_map.max()) if max_level is None else int(max_level)
        capacity = dem.size // 2 if capacity is None else int(capacity)
        options = {} if outlet_mask is None else {"outlet_mask": outlet_mask}
        self.tree = DynamicQuadtree(be, self.pool, level_map, dx, top, topology="D4", capacity=capacity, **options)
        self.sw = TreeFuswProgram(
            be, tree=self.tree, order=order, limiter=limiter, flux=flux, friction=friction,
            precision=precision, max_restarts=max_restarts, rain=self.rain, gravity=9.81, cfl=cfl,
            dt_max=dt_max, friction_coefficient=manning, outlet_type=outlet_type,
            outlet_slope=outlet_slope, outlet_manning=outlet_manning, **self.tree.dims)
        self.real = cp.float64 if precision == "f64" else cp.float32
        self.bed = transfer.FromRaster(dem)
        tree = self.tree
        z = cp.zeros(tree.capacity, self.real)
        z[: tree.n] = self.bed.sample(tree, tree.level, tree.x, tree.y, self.real)
        self.sw.z.array[...] = z.reshape(self.sw.z.array.shape)
        for name in STATE + STAGE_STATE + SLOPES + RATES + ("clock", "control", "step_budget", "budget"):
            self.sw._handle(name).array.fill(0)
        self.he_ca = 1.0e-12
        self.stop_reason = None

    def _field(self, name):
        return self.sw._handle(name).array.reshape(-1)

    @property
    def h(self):
        """Water depth per leaf, as a device array of ``capacity`` entries."""
        return self._field("h")

    @property
    def u(self):
        """Velocity along x per leaf."""
        return self._field("u")

    @property
    def v(self):
        """Velocity along y, towards growing rows, per leaf."""
        return self._field("v")

    @property
    def z(self):
        """Bed elevation per leaf."""
        return self._field("z")

    @property
    def time(self):
        """Simulated time, in s."""
        return float(self._field("clock")[C_TIME])

    @property
    def dt(self):
        """Last time step, in s."""
        return float(self._field("clock")[C_DT])

    @property
    def steps(self):
        """Time steps run."""
        return int(self._field("control")[K_STEP])

    def to_raster(self, field):
        """Return a per-leaf field as a host array on the finest raster."""
        return self.tree.to_raster(field)

    def run(self, n=1):
        """Advance ``n`` time steps."""
        self.sw._step(int(n))

    def area(self):
        """Leaf areas, ``n`` entries."""
        return (self.dx * self.dx) * 4.0 ** self.tree.level.astype(cp.float64)

    def volume(self):
        """Water volume on the domain, in m3."""
        return float((self.h[: self.tree.n].astype(cp.float64) * self.area()).sum())

    def outflow(self):
        """Volume gone through the outlet faces, in m3."""
        return float(self._field("budget")[B_OUT])

    def cut(self):
        """Volume removed by the dry cut, in m3 (negative: a negative depth was filled)."""
        return float(self._field("budget")[B_THRESHOLD])

    def rain_volume(self):
        """Volume rained so far, in m3."""
        return self.rain * self.tree.nx * self.tree.ny * self.dx * self.dx * self.time

    def indicator(self):
        """Return the absolute value of the refinement quantity per current leaf."""
        n = self.tree.n
        if self.criterion == "h":
            return cp.abs(self.h[:n])
        if self.criterion == "u":
            return cp.abs(self.u[:n])
        if self.criterion == "v":
            return cp.abs(self.v[:n])
        return cp.sqrt(self.u[:n] ** 2 + self.v[:n] ** 2)

    def flags(self):
        """Return the per-leaf update flags for the current state."""
        value = self.indicator()
        out = cp.zeros(self.tree.n, cp.int32)
        if self.coarsen_below is not None:
            out[value < self.coarsen_below] = -1
        out[(value >= self.threshold) & (self.tree.level > 0)] = 1
        return out

    def refine(self, flags=None):
        """Update the tree from ``flags`` (default: the criterion); return the leaf count.

        When the update does not fit the capacity the tree stays as it is and
        ``stop_reason`` tells why.
        """
        flags = self.flags() if flags is None else flags
        tree, h = self.tree, self.h
        try:
            tree.adapt(flags, [(self.z, self.bed), (self.u, transfer.Velocity(h)),
                               (self.v, transfer.Velocity(h)), (h, transfer.Depth(self.z))])
        except ValueError as error:
            if "exceed" not in str(error):
                raise
            self.stop_reason = str(error)
            return tree.n
        dry = h[: tree.n] <= self.he_ca
        self.u[: tree.n][dry] = 0
        self.v[: tree.n][dry] = 0
        return tree.n

    def mass_error(self):
        """Return ``volume - rain + outflow + cut``, in m3; zero for a closed balance."""
        return self.volume() - self.rain_volume() + self.outflow() + self.cut()

    def close(self):
        """Release the solver and the tree."""
        self.sw.close()
        self.tree.close()
