"""Local-inertia flood solver on a D4 quadtree (CuPy only).

The scheme of :mod:`pyfastflow.flood.inertial`: the semi-implicit Bates update
of the link discharges, limited by the Froude number and by the water the donor
leaf holds, an optional regulator of the provisional discharges, then
continuity.

State: depth per leaf, and ``Q`` per directed link in ``ADJ`` order, the volume
per time leaving the row's leaf through that whole face. A link's two entries
are opposite. A face has width ``min(dx_i, dx_j)`` and its head gradient runs
over ``(dx_i + dx_j) / 2``; both leaves of a link evaluate it at one call site,
west or north leaf first, so they get the same bits. The donor limit
``transfer_fraction h dx_donor / dt`` per unit width lets a side of the donor
give the same share of its water with one face or two. Refining moves ``Q`` with
``transfer.FaceDischarge``, which keeps the divergence of kept leaves.

Regulators weigh a link against the mean unit discharge on the outer sides of
its two leaves, or against itself where such a side has no link.

Domain edges are walls, unless the leaf is an outlet: ``outlet_depth`` holds its
depth, ``boundary_flux`` draws a unit discharge through its edge faces and
``outlet_discharge`` removes m3/s from it. With none of them set an outlet keeps
its water. ``time_step="cfl"`` takes dt = min(dt_max, cfl / max(sqrt(g h) / dx))
over each leaf and its neighbours, ``"fixed"`` takes ``dt``. Rain is uniform.

A regulator can take more water from a leaf than it holds; the depth floor then
adds the difference, and :meth:`QuadtreeInertial.cut` returns that volume.
``q_upwind`` and ``q_centered`` do this on a refined tree.

Author: B.G (11/2026)
"""

import cupy as cp
import numpy as np

from pyfastflow.core import HelperBuilder, KernelBuilder, RoutineBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flood.fusw._common import PRECISIONS, literal, warp_reduce
from pyfastflow.flow._program_cupy import _grid_leaf_plan
from pyfastflow.quadtree import DynamicQuadtree, transfer

SHAPE = (Dim("ny"), Dim("nx"))
REGULATORS = ("bates", "q_upwind", "q_centered", "s_upwind", "s_centered")
CRITERIA = ("h", "speed", "u", "v")
C_TIME, C_DT = range(2)
K_RATE, K_STEP = range(2)
B_OUT, B_CUT = range(2)


def _real(config):
    return PRECISIONS[config["precision"]]


def _extent(config):
    return int(config["nx"]) * int(config["ny"])


def _plan(frozen, _be):
    names = ("z", "h", "Q", "Q_reg", "u", "v", "clock", "control", "budget", "dt",
             "gravity", "manning", "froude_limit", "transfer_fraction", "regulator_theta",
             "min_flow_depth", "min_depth", "boundary_flux", "outlet_discharge", "rainfall",
             "cfl", "dt_max")
    return _grid_leaf_plan(frozen, {name: name for name in names})


def _kernel(text, config, bundles):
    return KernelBuilder(text, domain=_extent(config)).compose("grid", bundles["grid"]).freeze()


def _rate_kernel(bundles, config):
    real, c = _real(config), literal(_real(config))
    return _kernel(warp_reduce(real) + f'''
extern "C" __global__ void ti_rate(const {real}* h, unsigned int* control) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    {real} rate = {c(0)};
    if (i < $ctx.grid.n_leaves()$) {{
        {real} g = ({real})$ctx.gravity.get(0)$;
        {real} dxi = ({real})$ctx.grid.dx(i)$;
        {real} top = h[i];
        {real} d = dxi;
        unsigned int m = $ctx.grid.SLOTS.get(i)$;
        int p = $ctx.grid.PTR.get(i)$;
        for (int k = 0; k < 8; ++k) {{
            if (!((m >> k) & 1u)) continue;
            int j = $ctx.grid.ADJ.get(p)$;
            ++p;
            top = fmax(top, h[j]);
            d = fmin(d, ({real})$ctx.grid.dx(j)$);
        }}
        rate = sqrt(g * fmax(top, {c(0)})) / d;
    }}
    rate = fusw_warp_max(rate);
    if ((threadIdx.x & 31) == 0 && rate > {c(0)})
        atomicMax(&control[{K_RATE}], __float_as_uint((float)rate));
}}
''', config, bundles)


def _dt_kernel(bundles, config):
    real, c = _real(config), literal(_real(config))
    if config["time_step"] == "fixed":
        body = f"clock[{C_DT}] = (double)$ctx.dt.get(0)$;"
    else:
        body = f'''{real} rate = ({real})__uint_as_float(control[{K_RATE}]);
    {real} cfl = ({real})$ctx.cfl.get(0)$;
    {real} top = ({real})$ctx.dt_max.get(0)$;
    if (top <= {c(0)}) top = cfl * ({real})$ctx.grid.DX0.get(0)$;
    clock[{C_DT}] = (double)(rate > {c(0)} ? fmin(top, cfl / rate) : top);'''
    return KernelBuilder(f'''
extern "C" __global__ void ti_dt(double* clock, const unsigned int* control) {{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    {body}
}}
''', domain=1).compose("grid", bundles["grid"]).freeze()


def _flux_kernel(bundles, config):
    real, c = _real(config), literal(_real(config))
    filtered = config["regulator"] != "bates"
    out = "Q_reg" if filtered else "Q"
    args = f"const {real}* Q, {real}* Q_reg" if filtered else f"{real}* Q"
    return _kernel(f'''
__device__ __forceinline__ {real} ti_update({real} q_old, {real} h_a, {real} h_b, {real} z_a, {real} z_b,
        {real} dt, {real} gravity, {real} manning, {real} froude_limit, {real} transfer_fraction,
        {real} min_hflow, {real} length, {real} dx_a, {real} dx_b) {{
    {real} hflow = {c(0.5)} * (h_a + h_b);
    if (!(hflow > min_hflow)) return {c(0)};
    {real} slope = ((z_b + h_b) - (z_a + h_a)) / length;
    {real} h10_3 = hflow * hflow * hflow * cbrt(hflow);
    {real} friction = {c(1)} + gravity * hflow * dt * manning * manning * fabs(q_old) / fmax(h10_3, {c(1e-12)});
    {real} q = (q_old - gravity * hflow * dt * slope) / friction;
    {real} q_froude = hflow * sqrt(gravity * hflow) * froude_limit;
    q = fmin(fmax(q, -q_froude), q_froude);
    {real} donor_h = q >= {c(0)} ? h_a : h_b;
    {real} donor_dx = q >= {c(0)} ? dx_a : dx_b;
    {real} q_transfer = transfer_fraction * donor_h * donor_dx / fmax(dt, {c(1e-12)});
    return fmin(fmax(q, -q_transfer), q_transfer);
}}

extern "C" __global__ void ti_flux(const {real}* z, const {real}* h, {args}, const double* clock) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.grid.n_leaves()$) return;
    {real} dt = ({real})clock[{C_DT}];
    {real} gravity = ({real})$ctx.gravity.get(0)$;
    {real} manning = ({real})$ctx.manning.get(0)$;
    {real} froude_limit = ({real})$ctx.froude_limit.get(0)$;
    {real} transfer_fraction = ({real})$ctx.transfer_fraction.get(0)$;
    {real} min_hflow = ({real})$ctx.min_flow_depth.get(0)$;
    {real} dxi = ({real})$ctx.grid.dx(i)$;
    unsigned int m = $ctx.grid.SLOTS.get(i)$;
    int p = $ctx.grid.PTR.get(i)$;
    for (int k = 0; k < 8; ++k) {{
        if (!((m >> k) & 1u)) continue;
        int j = $ctx.grid.ADJ.get(p)$;
        bool right = k >= 4;
        {real} s = right ? {c(1)} : {c(-1)};
        int a = right ? i : j, b = right ? j : i;
        {real} dxj = ({real})$ctx.grid.dx(j)$;
        {real} width = fmin(dxi, dxj);
        {real} length = {c(0.5)} * (dxi + dxj);
        {real} q = ti_update(s * Q[p] / width, h[a], h[b], z[a], z[b], dt, gravity, manning,
                             froude_limit, transfer_fraction, min_hflow, length,
                             right ? dxi : dxj, right ? dxj : dxi);
        {out}[p] = s * width * q;
        ++p;
    }}
}}
''', config, bundles)


def _regulate_kernel(bundles, config):
    real, c = _real(config), literal(_real(config))
    upwind = config["regulator"] in ("q_upwind", "s_upwind")
    adaptive = config["regulator"] in ("s_upwind", "s_centered")
    if adaptive:
        theta = f'''{real} theta = {c(1)};
        {real} hflow = {c(0.5)} * (h[a] + h[b]);
        if (hflow > {c(0)}) {{
            theta = {c(1)} - (dt / length) * fmin(fabs(q) / hflow, sqrt(gravity * hflow));
            theta = fmin(fmax(theta, {c(0)}), {c(1)});
        }}'''
    else:
        theta = f"{real} theta = fmin(fmax(({real})$ctx.regulator_theta.get(0)$, {c(0)}), {c(1)});"
    neighbour = "q >= " + c(0) + " ? previous : next" if upwind else f"{c(0.5)} * (previous + next)"
    side = HelperBuilder(f'''
__device__ int ti_side(const {real}* Q_reg, int i, int sd, {real}* q) {{
    unsigned int m = $ctx.grid.SLOTS.get(i)$;
    int p = $ctx.grid.PTR.get(i)$ + __popc(m & ((1u << (2 * sd)) - 1u));
    {real} dxi = ({real})$ctx.grid.dx(i)$, total = {c(0)};
    int count = 0;
    for (int t = 0; t < 2; ++t) {{
        if (!((m >> (2 * sd + t)) & 1u)) continue;
        int j = $ctx.grid.ADJ.get(p)$;
        total += Q_reg[p] / fmin(dxi, ({real})$ctx.grid.dx(j)$);
        ++p;
        ++count;
    }}
    if (count > 0) *q = total / ({real})count;
    return count;
}}
''').compose("grid", bundles["grid"]).freeze()
    text = f'''
extern "C" __global__ void ti_regulate({real}* Q, const {real}* Q_reg, const {real}* h, const double* clock) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.grid.n_leaves()$) return;
    {real} dt = ({real})clock[{C_DT}];
    {real} gravity = ({real})$ctx.gravity.get(0)$;
    {real} dxi = ({real})$ctx.grid.dx(i)$;
    unsigned int m = $ctx.grid.SLOTS.get(i)$;
    int p = $ctx.grid.PTR.get(i)$;
    for (int k = 0; k < 8; ++k) {{
        if (!((m >> k) & 1u)) continue;
        int j = $ctx.grid.ADJ.get(p)$;
        int sd = k >> 1;
        bool right = sd >= 2, xa = (sd == 1 || sd == 2);
        {real} s = right ? {c(1)} : {c(-1)};
        int a = right ? i : j, b = right ? j : i;
        {real} dxj = ({real})$ctx.grid.dx(j)$;
        {real} width = fmin(dxi, dxj), length = {c(0.5)} * (dxi + dxj);
        {real} q = s * Q_reg[p] / width;
        {real} outer;
        {real} previous = q, next = q;
        if ($ctx.side(Q_reg, a, xa ? 1 : 0, &outer)$ > 0) previous = -outer;
        if ($ctx.side(Q_reg, b, xa ? 2 : 3, &outer)$ > 0) next = outer;
        {theta}
        Q[p] = s * width * (theta * q + ({c(1)} - theta) * ({neighbour}));
        ++p;
    }}
}}
'''
    return (KernelBuilder(text, domain=_extent(config)).compose("grid", bundles["grid"])
            .compose("side", side).freeze())


def _depth_kernel(bundles, config):
    real, c = _real(config), literal(_real(config))
    depth = config["outlet_depth"]
    hold = "" if depth is None else f'''
    if ($ctx.grid.can_out(i)$) {{
        out += (double)(h_new - {c(depth)}) * (double)area;
        h_new = {c(depth)};
    }}'''
    return _kernel(f'''
extern "C" __global__ void ti_depth({real}* h, const {real}* Q, const double* clock, double* budget) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.grid.n_leaves()$) return;
    {real} dt = ({real})clock[{C_DT}];
    {real} dx = ({real})$ctx.grid.dx(i)$;
    {real} area = dx * dx;
    {real} leaving = {c(0)};
    unsigned int m = $ctx.grid.SLOTS.get(i)$;
    int p = $ctx.grid.PTR.get(i)$;
    int count = __popc(m);
    for (int q = 0; q < count; ++q) leaving += Q[p + q];
    double out = 0.0;
    if ($ctx.grid.can_out(i)$) {{
        {real} drawn = ({real})$ctx.outlet_discharge.get(0)$;
        {real} unit = ({real})$ctx.boundary_flux.get(0)$;
        for (int sd = 0; sd < 4; ++sd)
            if (((m >> (2 * sd)) & 3u) == 0u) drawn += unit * dx;
        leaving += drawn;
        out += (double)drawn * (double)dt;
    }}
    {real} h_new = h[i] + dt * (-leaving / area + ({real})$ctx.rainfall.get(0)$);
    {real} lowest = ({real})$ctx.min_depth.get(0)$;
    if (h_new < lowest) {{
        atomicAdd(&budget[{B_CUT}], (double)(h_new - lowest) * (double)area);
        h_new = lowest;
    }}{hold}
    if (out != 0.0) atomicAdd(&budget[{B_OUT}], out);
    h[i] = h_new;
}}
''', config, bundles)


def _finalize_kernel():
    return KernelBuilder(f'''
extern "C" __global__ void ti_finalize(double* clock, unsigned int* control) {{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    clock[{C_TIME}] += clock[{C_DT}];
    control[{K_STEP}] += 1u;
    control[{K_RATE}] = 0u;
}}
''', domain=1).freeze()


def _velocity_kernel(bundles, config):
    real, c = _real(config), literal(_real(config))
    return _kernel(f'''
extern "C" __global__ void ti_velocity(const {real}* h, const {real}* Q, {real}* u, {real}* v) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.grid.n_leaves()$) return;
    {real} side[4] = {{{c(0)}, {c(0)}, {c(0)}, {c(0)}}};
    unsigned int m = $ctx.grid.SLOTS.get(i)$;
    int p = $ctx.grid.PTR.get(i)$;
    for (int k = 0; k < 8; ++k) {{
        if (!((m >> k) & 1u)) continue;
        side[k >> 1] += Q[p];
        ++p;
    }}
    {real} dx = ({real})$ctx.grid.dx(i)$;
    {real} wet = h[i] > {c(1e-6)} ? {c(0.5)} / (h[i] * dx) : {c(0)};
    u[i] = (side[2] - side[1]) * wet;
    v[i] = (side[3] - side[0]) * wet;
}}
''', config, bundles)


def _step_factory(be, bundles, config):
    rb = RoutineBuilder()
    if config["time_step"] == "cfl":
        rb.step("rate", _rate_kernel(bundles, config))
    rb.step("dt", _dt_kernel(bundles, config))
    rb.step("flux", _flux_kernel(bundles, config))
    if config["regulator"] != "bates":
        rb.step("regulate", _regulate_kernel(bundles, config))
    rb.step("depth", _depth_kernel(bundles, config))
    rb.step("finalize", _finalize_kernel())
    return rb.freeze()


def _velocity_factory(be, bundles, config):
    return RoutineBuilder().step("velocity", _velocity_kernel(bundles, config)).freeze()


def _build_program():
    b = ProgramBuilder("TreeInertialProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("tree")
    b.config("precision", choices=tuple(PRECISIONS), default="f32")
    b.config("regulator", choices=REGULATORS, default="bates")
    b.config("time_step", choices=("cfl", "fixed"), default="cfl")
    b.config("outlet_depth", default=None)
    for name, value in (("dt", 1.0e-3), ("rainfall", 0.0), ("manning", 0.033), ("gravity", 9.81),
                        ("froude_limit", 1.0), ("transfer_fraction", 0.2), ("regulator_theta", 0.9),
                        ("min_flow_depth", 1.0e-4), ("min_depth", 0.0), ("boundary_flux", 0.0),
                        ("outlet_discharge", 0.0), ("cfl", 0.5), ("dt_max", 0.0)):
        b.param(name, "scalar", "f32", value=value)
    dtype = lambda config: config["precision"]
    links = lambda config: (config["tree"].adj_capacity,)
    b.data("z", dtype, SHAPE, role="input", shape_source=True)
    for name in ("h", "u", "v"):
        b.data(name, dtype, SHAPE, role="state")
    b.data("Q", dtype, links, role="state")
    b.data("Q_reg", dtype, lambda c: links(c) if c["regulator"] != "bates" else (0,), role="internal")
    b.data("clock", "f64", (2,), role="internal")
    b.data("control", "u32", (2,), role="internal")
    b.data("budget", "f64", (2,), role="output")
    b.bundle("grid", lambda be, *, tree, **_: tree.group,
             lambda be, pool, *, tree, **_: tree.parameters, dims=("nx", "ny"), config=("tree",))
    b.add("_step", _step_factory, bind=_plan)
    b.add("_velocity", _velocity_factory, bind=_plan)
    return b.freeze()


TreeInertialProgram = _build_program()


class QuadtreeInertial:
    """Local-inertia solver on a tree refined from its own solution.

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
        Quantity refined on, compared in absolute value. ``u`` and ``v`` are
        the leaf's mean face discharges over its depth.
    threshold : float
        A leaf is flagged for refinement at or above this value.
    coarsen_below : float, optional
        A leaf is flagged for coarsening below this value.
    capacity : int, optional
        Largest leaf count; defaults to half the finest raster.
    manning, cfl, dt, dt_max, time_step, regulator, regulator_theta, froude_limit,
    transfer_fraction, min_flow_depth, outlet_depth, boundary_flux, outlet_discharge, precision
        Solver settings, see the module docstring.
    outlet_mask : ndarray, optional
        Outlet cells for the tree; defaults to the domain edges.
    """

    def __init__(self, be, dem, dx, rain, *, levels=5, max_level=None, criterion="h", threshold=0.05,
                 coarsen_below=None, capacity=None, manning=0.033, cfl=0.7, dt=1.0e-3, dt_max=0.0,
                 time_step="cfl", regulator="s_upwind", regulator_theta=0.9, froude_limit=1.0,
                 transfer_fraction=0.2, min_flow_depth=1.0e-4, outlet_depth=0.0, boundary_flux=0.0,
                 outlet_discharge=0.0, precision="f32", outlet_mask=None):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
        dem = np.asarray(dem, dtype=np.float64 if precision == "f64" else np.float32)
        self.be, self.criterion, self.threshold = be, criterion, float(threshold)
        self.coarsen_below, self.rain, self.dx = coarsen_below, float(rain), float(dx)
        self.real = cp.float64 if precision == "f64" else cp.float32
        self.pool = be.PoolCls()
        level_map = levels if isinstance(levels, np.ndarray) else np.full(dem.shape, levels, np.int32)
        top = int(level_map.max()) if max_level is None else int(max_level)
        capacity = dem.size // 2 if capacity is None else int(capacity)
        options = {} if outlet_mask is None else {"outlet_mask": outlet_mask}
        self.tree = DynamicQuadtree(be, self.pool, level_map, dx, top, topology="D4", capacity=capacity, **options)
        self.sw = TreeInertialProgram(
            be, tree=self.tree, precision=precision, regulator=regulator, time_step=time_step,
            outlet_depth=outlet_depth, **self.tree.dims)
        for name, value in (("rainfall", self.rain), ("manning", manning), ("cfl", cfl), ("dt", dt),
                            ("dt_max", dt_max), ("regulator_theta", regulator_theta),
                            ("froude_limit", froude_limit), ("transfer_fraction", transfer_fraction),
                            ("min_flow_depth", min_flow_depth), ("boundary_flux", boundary_flux),
                            ("outlet_discharge", outlet_discharge)):
            getattr(self.sw, name).set(value)
        for name in ("h", "u", "v", "Q", "Q_reg", "clock", "control", "budget"):
            self.sw._handle(name).array.fill(0)
        self.bed = transfer.FromRaster(dem)
        tree = self.tree
        z = cp.zeros(tree.capacity, self.real)
        z[: tree.n] = self.bed.sample(tree, tree.level, tree.x, tree.y, self.real)
        self.z[...] = z
        self.stop_reason = None

    def _field(self, name):
        return self.sw._handle(name).array.reshape(-1)

    @property
    def h(self):
        """Water depth per leaf, as a device array of ``capacity`` entries."""
        return self._field("h")

    @property
    def z(self):
        """Bed elevation per leaf."""
        return self._field("z")

    @property
    def Q(self):
        """Volume per time out of the row's leaf per link, ``adj_capacity`` entries."""
        return self._field("Q")

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

    def velocity(self):
        """Return device arrays ``(u, v)`` per leaf, from the face discharges."""
        self.sw._velocity()
        return self._field("u"), self._field("v")

    def area(self):
        """Leaf areas, ``n`` entries."""
        return (self.dx * self.dx) * 4.0 ** self.tree.level.astype(cp.float64)

    def volume(self):
        """Water volume on the domain, in m3."""
        return float((self.h[: self.tree.n].astype(cp.float64) * self.area()).sum())

    def outflow(self):
        """Volume gone through the outlets, in m3."""
        return float(self._field("budget")[B_OUT])

    def cut(self):
        """Volume removed by the depth floor, in m3 (negative: added to fill a deficit)."""
        return float(self._field("budget")[B_CUT])

    def rain_volume(self):
        """Volume rained so far, in m3."""
        return self.rain * self.tree.nx * self.tree.ny * self.dx * self.dx * self.time

    def mass_error(self):
        """Return ``volume - rain + outflow + cut``, in m3; zero for a closed balance."""
        return self.volume() - self.rain_volume() + self.outflow() + self.cut()

    def indicator(self):
        """Return the absolute value of the refinement quantity per current leaf."""
        n = self.tree.n
        if self.criterion == "h":
            return cp.abs(self.h[:n])
        u, v = self.velocity()
        if self.criterion == "u":
            return cp.abs(u[:n])
        if self.criterion == "v":
            return cp.abs(v[:n])
        return cp.sqrt(u[:n] ** 2 + v[:n] ** 2)

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
        try:
            self.tree.adapt(flags, [(self.z, self.bed), (self.h, transfer.Depth(self.z)),
                                    (self.Q, transfer.FaceDischarge())])
        except ValueError as error:
            if "exceed" not in str(error):
                raise
            self.stop_reason = str(error)
        return self.tree.n

    def close(self):
        """Release the solver and the tree."""
        self.sw.close()
        self.tree.close()
