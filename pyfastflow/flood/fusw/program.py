"""FuswProgram: finite-volume shallow-water solver after FullSWOF 2D
(Delestre et al. 2017), CuPy only.

State h, u, v (depth, x and y velocities; y grows with the row index) on a
flat row-major grid of D4 cells; the bed z is fixed. One step (see
``_kernels.py``): face fluxes from the reconstructed states (``order`` 1:
cell values, 2: MUSCL with ``limiter``) with the hydrostatic reconstruction
and the Riemann ``flux``, then the update of h (rain, sources,
``infiltration``) and of the discharges (bed slope, ``friction``). Order 2
is the Heun scheme of FullSWOF; when the intermediate state needs a smaller
time step, the first stage is recomputed from the step-start state with it
(at most ``max_restarts`` times per step). ``time_step="cfl"`` takes
dt = min(dt_max, cfl dx / c_max) (order 2: c_max times the growth of the
wave speed over the first stage of the last step, at least 1), ``"fixed"``
takes ``dt`` and flags a CFL violation (``run`` raises).

Configs:
  - grid: ``ny``, ``nx``, ``dx``, ``boundary`` (normal | periodic_EW |
    periodic_NS), ``outlet`` (edge | mask), ``nodata`` (closed cells);
  - ``precision`` f32 | f64 (state, fluxes; parameters stay f32);
  - ``order`` 1 | 2, ``reconstruction`` muscl, ``limiter`` minmod |
    vanalbada | vanleer, ``flux`` rusanov | hll | hll2 | hllc | hllc2,
    ``friction`` none | manning | darcy_weisbach | laminar,
    ``infiltration`` none | green_ampt, ``time_step`` cfl | fixed,
    ``max_restarts``;
  - boundaries: ``outlet_type`` (code of the grid outlet cells at
    ``initialize``), ``inflow_unit`` discharge (m3/s per cell) | unit (m2/s
    per face), ``inflow_lines`` (number of inflow lines, 0: none),
    ``inflow_line_faces`` (capacity of their faces, all lines together; 0:
    2 (nx + ny));
  - forcing in time: ``rain_series``, ``inflow_series``, ``source_series``
    (rows of the (time, factor) tables; 0: factor 1), ``series_interp``
    step (a factor holds from its time on) | linear; outside a table its end
    values hold;
  - monitoring: ``n_probes``, ``record_capacity`` (rows; 0: no recording),
    ``surface_slope`` gradient | d8 (water-surface slope of the ``surface``
    shear stress, ``_shear.py``).

Boundaries: every face of an active cell whose D4 neighbour is outside a
non-periodic edge or closed is a boundary face, acting by the u8 code of the
cell in ``boundary_type`` (``BOUNDARY_CODES``: wall, neumann, stage,
critical, normal_depth, rating, discharge, sink, inflow; ``_boundary.py``)
when bit k of its u8 ``boundary_faces`` is set for that face (k: 0 north, 1 west,
2 east, 3 south), as a wall otherwise. ``initialize()`` gives the grid
outlet cells ``outlet_type`` (sink when they have no boundary face), every
other cell wall, and selects all faces; both fields can then be edited per
cell (e.g. a corner cell of an inflow edge keeps only its edge face).

Inflow lines (code inflow): ``set_inflow_line(k, rows, cols, sides,
discharge, slope)`` makes line k of boundary faces (cell, side N | W | E |
S) carrying the total discharge ``discharge`` (m3/s, times the inflow time
factor). Each step whose line discharge changed, its faces share it as the
uniform flow (``fusw_uniform_q`` of the friction law, energy slope
``slope``) under one water level, found by bisection so that the line
carries exactly the discharge; each face then imposes its share as its mass
flux (``fusw_inflow_face``, ``_boundary.py``). Lines are set after
``initialize``, which clears them.

Parameters (f32; per-cell ones accept a scalar or an (ny, nx) array):
``rain`` (m/s), ``source`` (m3/s per cell), ``inflow`` (``inflow_unit``),
``stage`` (imposed water surface, m), ``outlet_slope``, ``outlet_manning``,
``rating_a`` / ``rating_b`` / ``rating_eta0``, ``friction_coefficient``,
Green-Ampt ``ks``, ``kc``, ``dtheta``, ``psi``, ``zcrust``, ``imax``;
scalars ``gravity``, ``cfl``, ``dt``, ``dt_max`` (<= 0: cfl dx), ``he_ca``,
``ve_ca``, ``ie_ca``, ``epsilon``, ``newton_iterations``,
``record_interval``, ``rho`` (water density of the shear stress). Rain,
inflow and sources are multiplied by their time factor.

Use: set ``z``, then ``initialize()`` (zeros h, u, v, the infiltrated depth,
time and budget, sets the boundary codes and face masks), optionally set
h, u, v, ``boundary_type`` and ``boundary_faces``, then ``run(n)`` / ``run_until(t)``. Monitoring:
``field(name)``, ``face_flux(axis)``, ``budget()``, ``time``, ``dt``,
``steps``, ``set_probes``, ``records()``; bed shear stress on request,
per cell ``shear_stress(method)`` and per face ``face_shear_stress(method)``
(method friction | surface, ``_shear.py``), Froude number ``froude()`` /
``face_froude()`` and acceleration ratio ``acceleration_ratio(steps)`` /
``face_acceleration_ratio(steps)`` (direct; runs ``steps`` steps) or
``acceleration_ratio_balance()`` / ``face_acceleration_ratio_balance()``
(from the momentum balance) (``_diagnostics.py``), in temporary buffers.

Author: B.G (10/2026)
"""

import numpy as np

from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.grid._mask import GridMaskAccessor

from ._boundary import INFLOW_UNITS
from ._common import (
    B_FACE, BOUNDARY_CODES, C_DT, C_T_END, C_TIME, FACE_TERMS, K_CFL_ERROR,
    K_REC_BASE, K_REC_OVERFLOW, K_STEP, K_TOTAL_RESTARTS, N_BUDGET, N_CLOCK,
    N_CONTROL, N_FACE_TERMS, OUTLET_TYPES, PRECISIONS, SERIES, cupy_only,
)
from ._friction import FRICTIONS
from ._infiltration import INFILTRATIONS
from . import _diagnostics as diag
from ._kernels import (
    bind_plan, codes_factory, diagnostic_factory, prepare_factory, record_width,
    shear_factory, step_factory,
)
from ._limiters import LIMITERS
from ._riemann import FLUXES
from ._shear import CELL_OUT, FACE_OUT, SHEAR_METHODS, SURFACE_SLOPES

SHAPE = (Dim("ny"), Dim("nx"))
_VOLUMES = ("rain", "source", "infiltration", "sink", "threshold")


def _real(config):
    return config["precision"]


def _if(flag, shape):
    return lambda config: shape(config) if flag(config) else (0, 0)


def _cells(config):
    return (config["ny"], config["nx"])


def _line_capacity(config):
    capacity = int(config["inflow_line_faces"])
    return capacity if capacity > 0 else 2 * (int(config["nx"]) + int(config["ny"]))


_SIDES = {"N": 0, "W": 1, "E": 2, "S": 3}


def _ratio(sx, sy, fx, fy):
    """|s - f| / |s| of two vector fields, NaN where |s| = 0."""
    norm = np.hypot(sx, sy)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(norm > 0.0, np.hypot(sx - fx, sy - fy) / norm, np.nan)


def fusw_builder(name="FuswProgram"):
    """ProgramBuilder holding the FuswProgram recipe (see the module
    docstring); ``finish_fusw_program`` attaches the host methods to the
    frozen class."""
    b = ProgramBuilder(name)
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("dx", default=1.0)
    b.config("boundary", choices=("normal", "periodic_EW", "periodic_NS"), default="normal")
    b.config("outlet", choices=("edge", "mask"), default="edge")
    b.config("nodata", choices=(False, True), default=False)
    b.config("precision", choices=tuple(PRECISIONS), default="f32")
    b.config("order", choices=(1, 2), default=2)
    b.config("reconstruction", choices=("muscl",), default="muscl")
    b.config("limiter", choices=LIMITERS, default="minmod")
    b.config("flux", choices=FLUXES, default="hll")
    b.config("friction", choices=FRICTIONS, default="manning")
    b.config("infiltration", choices=INFILTRATIONS, default="none")
    b.config("time_step", choices=("cfl", "fixed"), default="cfl")
    b.config("max_restarts", default=2)
    b.config("outlet_type", choices=OUTLET_TYPES, default="neumann")
    b.config("inflow_unit", choices=INFLOW_UNITS, default="discharge")
    b.config("series_interp", choices=("step", "linear"), default="step")
    for name in SERIES:
        b.config(f"{name}_series", default=0)
    b.config("surface_slope", choices=SURFACE_SLOPES, default="gradient")
    b.config("inflow_lines", default=0)
    b.config("inflow_line_faces", default=0)
    b.config("n_probes", default=0)
    b.config("record_capacity", default=0)

    for name, value in (
            ("rain", 0.0), ("source", 0.0), ("inflow", 0.0), ("stage", 0.0),
            ("outlet_slope", 0.0), ("outlet_manning", 0.033),
            ("rating_a", 0.0), ("rating_b", 1.5), ("rating_eta0", 0.0),
            ("friction_coefficient", 0.033),
            ("ks", 0.0), ("kc", 0.0), ("dtheta", 0.0), ("psi", 0.0),
            ("zcrust", 0.0), ("imax", 0.0)):
        b.param(name, "auto", "f32", value=value, shape=SHAPE)
    for name, value in (
            ("gravity", 9.81), ("cfl", 0.5), ("dt", 1.0e-3), ("dt_max", 0.0),
            ("he_ca", 1.0e-12), ("ve_ca", 1.0e-12), ("ie_ca", 1.0e-8),
            ("epsilon", 1.0e-13), ("rho", 1000.0)):
        b.param(name, "auto", "f32", value=value)
    b.param("newton_iterations", "auto", "i32", value=100)
    b.param("record_interval", "auto", "i32", value=100)

    order2 = lambda config: int(config["order"]) == 2
    infiltrates = lambda config: config["infiltration"] != "none"
    b.data("z", _real, SHAPE, role="input", shape_source=True)
    for name in ("h", "u", "v"):
        b.data(name, _real, SHAPE, role="state")
    for name in ("hs", "us", "vs"):
        b.data(name, _real, _if(order2, _cells), role="internal")
    for name in ("fx1", "fx2", "fx3"):
        b.data(name, _real, (Dim("ny"), Dim("nx") + 1), role="internal")
    for name in ("fy1", "fy2", "fy3"):
        b.data(name, _real, (Dim("ny") + 1, Dim("nx")), role="internal")
    b.data("zx", _real, _if(order2, lambda c: (c["ny"], c["nx"] + 1)), role="internal")
    b.data("zy", _real, _if(order2, lambda c: (c["ny"] + 1, c["nx"])), role="internal")
    b.data("infiltrated", _real, _if(infiltrates, _cells), role="output")
    b.data("infiltrated_stage", _real,
           _if(lambda c: infiltrates(c) and order2(c), _cells), role="internal")
    b.data("boundary_type", "u8", SHAPE, role="input")
    b.data("boundary_faces", "u8", SHAPE, role="input")
    b.data("clock", "f64", (N_CLOCK,), role="internal")
    b.data("control", "u32", (N_CONTROL,), role="internal")
    b.data("budget", "f64", (N_BUDGET,), role="internal")
    b.data("budget_step", "f64", (N_BUDGET,), role="internal")
    for name in SERIES:
        b.data(f"{name}_table", "f64",
               lambda config, name=name: (int(config[f"{name}_series"]), 2), role="internal")
    for name in CELL_OUT:
        b.data(name, _real, SHAPE, lifetime="temp")
    for name in FACE_OUT[:2]:
        b.data(name, _real, (Dim("ny"), Dim("nx") + 1), lifetime="temp")
    for name in FACE_OUT[2:]:
        b.data(name, _real, (Dim("ny") + 1, Dim("nx")), lifetime="temp")
    for name in diag.FROUDE_CELL + diag.ACC_CELL + diag.ACC_IN[:2]:
        b.data(name, _real, SHAPE, lifetime="temp")
    b.data("acc_dt", _real, (1,), lifetime="temp")
    for name in (diag.FROUDE_FACE[0], diag.ACC_FACE[0]):
        b.data(name, _real, (Dim("ny"), Dim("nx") + 1), lifetime="temp")
    for name in (diag.FROUDE_FACE[1], diag.ACC_FACE[1]):
        b.data(name, _real, (Dim("ny") + 1, Dim("nx")), lifetime="temp")
    lines = lambda config: int(config["inflow_lines"])
    b.data("line_cells", "i32", lambda config: (_line_capacity(config) if lines(config) else 0,),
           role="internal")
    b.data("line_offsets", "i32", lambda config: (lines(config) + 1 if lines(config) else 0,),
           role="internal")
    for name in ("line_q", "line_slope", "line_last"):
        b.data(name, "f64", lambda config: (lines(config),), role="internal")
    for name in ("inflow_q", "inflow_depth"):
        b.data(name, _real, _if(lambda config: lines(config) > 0, _cells), role="internal")
    b.data("probe_index", "i32", lambda config: (int(config["n_probes"]),), role="internal")
    b.data("records", "f64",
           lambda config: (int(config["record_capacity"]), record_width(config)),
           role="internal")

    def grid_structure(be, *, boundary, outlet, nodata, **_):
        cupy_only(be)
        return make_grid_group(be, topology="D4", boundary=boundary,
                               outlet=outlet, nodata=nodata)

    def grid_parameters(be, pool, *, nx, ny, dx, outlet, nodata, **_):
        if float(dx) <= 0.0:
            raise ValueError("FuswProgram requires dx > 0")
        return make_grid_parameters(be, pool, nx, ny, dx, topology="D4",
                                    outlet=outlet, nodata=nodata)

    b.bundle("grid", grid_structure, grid_parameters, dims=("nx", "ny"),
             config=("dx", "boundary", "outlet", "nodata"))

    b.add("_step", step_factory, bind=bind_plan)
    b.add("_prepare", prepare_factory, bind=bind_plan)
    b.add("_codes", codes_factory, bind=bind_plan)
    for location in ("cell", "face"):
        for method in SHEAR_METHODS:
            b.add(f"_tau_{location}_{method}", shear_factory(location, method), bind=bind_plan)
    for name, kernel in (("_froude_cell", diag.froude_cell_kernel),
                         ("_froude_face", diag.froude_face_kernel),
                         ("_acc_cell", diag.acceleration_cell_kernel),
                         ("_acc_face", diag.acceleration_face_kernel)):
        b.add(name, diagnostic_factory(kernel), bind=bind_plan)
    return b


def finish_fusw_program(program):
    """Attach the FuswProgram host methods to the frozen ``program``."""

    def initialize(self):
        """Zero h, u, v, the infiltrated depth, time, budget and recording;
        boundary codes from the grid; time factors at t = 0."""
        for name in ("h", "u", "v", "hs", "us", "vs", "infiltrated",
                     "infiltrated_stage", "clock", "control", "budget",
                     "budget_step", "records", "fx1", "fx2", "fx3",
                     "fy1", "fy2", "fy3", "zx", "zy", "line_cells",
                     "line_offsets", "line_q", "line_slope", "inflow_q", "inflow_depth"):
            self._handle(name).array.fill(0)
        self._handle("line_last").array.fill(-1)
        self._handle("boundary_faces").array.fill(15)
        self._codes()
        self._prepare()

    def run(self, n=1):
        """Advance ``n`` steps; raise on a CFL violation (fixed time step)."""
        self._prepare()
        self._step(int(n))
        control = self._handle("control").to_numpy()
        if control[K_CFL_ERROR]:
            raise RuntimeError(
                f"FuswProgram: CFL condition violated (cfl > {self.cfl.read()}) "
                f"with the fixed time step dt = {self.dt.read()}")

    def run_until(self, t, chunk=100):
        """Advance to time ``t`` (s), ``chunk`` steps between time checks."""
        clock = self._handle("clock")
        array = clock.to_numpy()
        array[C_T_END] = float(t)
        clock.from_numpy(array)
        while self.time < float(t):
            before = self.time
            self.run(chunk)
            if self.time <= before:
                break
        array = clock.to_numpy()
        array[C_T_END] = 0.0
        clock.from_numpy(array)

    def field(self, name):
        """Cell field as numpy: h, u, v, eta (z + h), qx (h u), qy (h v),
        q (|h u|, m2/s), Q (q dx, m3/s) or infiltrated."""
        if name in ("h", "u", "v", "infiltrated", "z"):
            return self._get_array(name)
        h, u, v = (self._get_array(k) for k in ("h", "u", "v"))
        if name == "eta":
            return self._get_array("z") + h
        if name == "qx":
            return h * u
        if name == "qy":
            return h * v
        q = h * np.sqrt(u * u + v * v)
        if name == "q":
            return q
        if name == "Q":
            return q * self.dx
        raise ValueError(f"unknown field {name!r}")

    def face_flux(self, axis="x", volumetric=False):
        """Mass flux of the last step's last flux evaluation (m2/s, or m3/s
        with ``volumetric``): x faces (ny, nx + 1) or y faces (ny + 1, nx),
        positive towards +x / +row."""
        out = self._get_array("fx1" if axis == "x" else "fy1")
        return out * self.dx if volumetric else out

    def budget(self):
        """Cumulative budget (see ``_common.py``): volumes rain, source,
        infiltration, sink, threshold and, per boundary code, the volume,
        momentum terms out; plus the stored volume and the balance error
        rain + source - infiltration - sink - threshold - out - stored."""
        raw = self._handle("budget").to_numpy()
        out = {name: float(raw[k]) for k, name in enumerate(_VOLUMES)}
        faces = {}
        for code_name, code in BOUNDARY_CODES.items():
            base = B_FACE + N_FACE_TERMS * code
            faces[code_name] = {term: float(raw[base + t]) for t, term in enumerate(FACE_TERMS)}
        out["boundary"] = faces
        stored = float(self._handle("h").array.sum(dtype="float64")) * self.dx * self.dx
        if self.infiltration != "none":
            stored_inf = float(self._handle("infiltrated").array.sum(dtype="float64")) * self.dx * self.dx
        else:
            stored_inf = 0.0
        out["stored"] = stored
        out["infiltrated_stored"] = stored_inf
        out["outflow"] = sum(f["volume"] for f in faces.values())
        out["error"] = (out["rain"] + out["source"] - out["infiltration"] - out["sink"]
                        - out["threshold"] - out["outflow"] - stored)
        return out

    def set_probes(self, rows, cols):
        """Cells recorded by ``records()`` (``n_probes`` of them)."""
        rows = np.asarray(rows, dtype=np.int64).ravel()
        cols = np.asarray(cols, dtype=np.int64).ravel()
        if rows.size != int(self.n_probes) or cols.size != rows.size:
            raise ValueError(f"set_probes: expected {self.n_probes} rows and cols")
        self._handle("probe_index").from_numpy((rows * self.nx + cols).astype(np.int32))

    def set_series(self, name, times, factors):
        """(time, factor) table of ``rain``, ``inflow`` or ``source``; takes
        effect at the next ``run``."""
        if name not in SERIES:
            raise ValueError(f"set_series: name must be one of {SERIES}")
        times = np.asarray(times, dtype=np.float64).ravel()
        factors = np.asarray(factors, dtype=np.float64).ravel()
        if times.size != int(getattr(self, f"{name}_series")) or factors.size != times.size:
            raise ValueError(f"set_series: {name}_series={getattr(self, f'{name}_series')} rows expected")
        self._handle(f"{name}_table").from_numpy(np.stack([times, factors], axis=1).ravel())

    def records(self):
        """Rows recorded since the last call, then cleared: time, dt, probe
        h / u / v (rows x n_probes) and the cumulative budget vector
        (rows x N_BUDGET); ``overflow`` is True when rows were dropped past
        ``record_capacity``."""
        if int(self.record_capacity) == 0:
            raise ValueError("records() requires record_capacity > 0")
        control = self._handle("control").to_numpy()
        every = int(self.record_interval.read())
        done = int(control[K_STEP]) // every if every > 0 else 0
        rows = max(0, min(done - int(control[K_REC_BASE]), int(self.record_capacity)))
        data = self._handle("records").to_numpy().reshape(int(self.record_capacity), -1)[:rows]
        p = int(self.n_probes)
        out = {
            "time": data[:, 0].copy(), "dt": data[:, 1].copy(),
            "h": data[:, 2:2 + p].copy(), "u": data[:, 2 + p:2 + 2 * p].copy(),
            "v": data[:, 2 + 2 * p:2 + 3 * p].copy(),
            "budget": data[:, 2 + 3 * p:].copy(),
            "overflow": bool(control[K_REC_OVERFLOW]),
        }
        control[K_REC_BASE] = done
        control[K_REC_OVERFLOW] = 0
        self._handle("control").from_numpy(control)
        return out

    def set_inflow_line(self, k, rows, cols, sides, discharge, slope):
        """Make inflow line ``k`` of the boundary faces (rows, cols, sides;
        sides N | W | E | S, one or one per cell) with the total
        ``discharge`` (m3/s) and the energy ``slope`` of its uniform-flow
        distribution; the cells take code inflow and only these faces."""
        n_lines = int(self.inflow_lines)
        if not 0 <= int(k) < n_lines:
            raise ValueError(f"set_inflow_line: line must be in [0, {n_lines})")
        rows = np.asarray(rows, dtype=np.int64).ravel()
        cols = np.asarray(cols, dtype=np.int64).ravel()
        sides = np.broadcast_to(np.asarray(sides), rows.shape)
        if cols.shape != rows.shape or rows.size == 0:
            raise ValueError("set_inflow_line: rows and cols must be non-empty and of one size")
        if not (np.all((0 <= rows) & (rows < self.ny)) and np.all((0 <= cols) & (cols < self.nx))):
            raise ValueError("set_inflow_line: cell outside the grid")
        bad = [side for side in np.unique(sides) if side not in _SIDES]
        if bad:
            raise ValueError(f"set_inflow_line: sides must be in {tuple(_SIDES)}, got {bad}")
        if self.friction != "none" and not float(slope) > 0.0:
            raise ValueError("set_inflow_line: slope must be > 0")
        if not self.nodata:
            edge = {"N": rows == 0, "S": rows == self.ny - 1, "W": cols == 0, "E": cols == self.nx - 1}
            periodic = {"periodic_EW": "WE", "periodic_NS": "NS"}.get(self.boundary, "")
            for side, at_edge in edge.items():
                pick = sides == side
                if np.any(pick & (~at_edge | (side in periodic))):
                    raise ValueError(f"set_inflow_line: a {side} face is not a boundary face")
        index = rows * self.nx + cols
        bits = np.array([1 << _SIDES[side] for side in sides], dtype=np.uint8)

        codes = self._handle("boundary_type").to_numpy()
        faces = self._handle("boundary_faces").to_numpy()
        codes[index] = BOUNDARY_CODES["inflow"]
        mask = np.zeros(self.nx * self.ny, dtype=np.uint8)
        np.bitwise_or.at(mask, index, bits)
        faces[index] = mask[index]
        self._handle("boundary_type").from_numpy(codes)
        self._handle("boundary_faces").from_numpy(faces)

        offsets = self._handle("line_offsets").to_numpy()
        cells = self._handle("line_cells").to_numpy()
        groups = [cells[offsets[j]:offsets[j + 1]] for j in range(n_lines)]
        groups[int(k)] = index.astype(np.int32)
        packed = np.concatenate(groups)
        if packed.size > cells.size:
            raise ValueError(f"set_inflow_line: {packed.size} faces exceed inflow_line_faces={cells.size}")
        cells[:packed.size] = packed
        offsets = np.concatenate([[0], np.cumsum([g.size for g in groups])]).astype(np.int32)
        self._handle("line_cells").from_numpy(cells)
        self._handle("line_offsets").from_numpy(offsets)
        self._handle("inflow_q").array.fill(0)
        self._handle("inflow_depth").array.fill(0)
        self._handle("line_last").array.fill(-1)
        self.set_line_discharge(k, discharge, slope)

    def set_line_discharge(self, k, discharge, slope=None):
        """Total discharge (m3/s) of inflow line ``k``, and its energy slope
        when given."""
        q = self._handle("line_q").to_numpy()
        q[int(k)] = float(discharge)
        self._handle("line_q").from_numpy(q)
        if slope is not None:
            s = self._handle("line_slope").to_numpy()
            s[int(k)] = float(slope)
            self._handle("line_slope").from_numpy(s)
            last = self._handle("line_last").to_numpy()
            last[int(k)] = -1.0
            self._handle("line_last").from_numpy(last)

    def _run_reading(self, name, outputs, inputs=None):
        """Run sequence ``name`` once with fresh temporary buffers (those
        named in ``inputs`` filled from numpy first) and return the named
        temporaries as numpy before they go back to the pool."""
        self._ensure_compiled(name)
        state = self._states[name]
        acquired = {}
        try:
            for temp, addrs in state.temp_plan:
                spec = self._recipe.data[temp]
                handle = self._pool.get_data(self._be.dtypes[self._dtype(temp)], self._device_shape(spec))
                acquired[temp] = handle
                for addr in addrs:
                    state.compiled.swap(addr, handle)
            for temp, array in (inputs or {}).items():
                acquired[temp].from_numpy(np.ascontiguousarray(array).reshape(-1))
            state.compiled()
            return tuple(acquired[k].to_numpy().reshape(self._host_shape(k)) for k in outputs)
        finally:
            for (temp, addrs), placeholder in zip(state.temp_plan, state.placeholders):
                for addr in addrs:
                    state.compiled.swap(addr, placeholder)
            for handle in acquired.values():
                self._pool.release_data(handle)

    def shear_stress(self, method="friction"):
        """Bed shear stress per cell (Pa) of the current state, (tau_x,
        tau_y) as (ny, nx) arrays; ``method`` friction (friction law of the
        solver) | surface (rho g h water-surface slope); see ``_shear.py``."""
        if method not in SHEAR_METHODS:
            raise ValueError(f"shear_stress: method must be one of {SHEAR_METHODS}")
        return self._run_reading(f"_tau_cell_{method}", CELL_OUT)

    def face_shear_stress(self, method="friction"):
        """Bed shear stress per face (Pa) of the current state: ((tau_x,
        tau_y) on the x faces (ny, nx + 1), (tau_x, tau_y) on the y faces
        (ny + 1, nx)); ``method`` as ``shear_stress``."""
        if method not in SHEAR_METHODS:
            raise ValueError(f"face_shear_stress: method must be one of {SHEAR_METHODS}")
        fxx, fxy, fyx, fyy = self._run_reading(f"_tau_face_{method}", FACE_OUT)
        return (fxx, fxy), (fyx, fyy)

    def froude(self):
        """Froude number |u| / sqrt(g h) per cell, (ny, nx), 0 when dry."""
        return self._run_reading("_froude_cell", diag.FROUDE_CELL)[0]

    def face_froude(self):
        """Froude number per face: (x faces (ny, nx + 1), y faces (ny + 1,
        nx)), from the averaged states on both sides (``_diagnostics.py``)."""
        return self._run_reading("_froude_face", diag.FROUDE_FACE)

    def _acceleration_inputs(self, steps):
        u0, v0, t0 = self.field("u"), self.field("v"), self.time
        self.run(int(steps))
        dt = self.time - t0
        if not dt > 0.0:
            raise RuntimeError("acceleration ratio: the run did not advance time")
        return {"acc_u0": u0, "acc_v0": v0, "acc_dt": np.array([dt], dtype=u0.dtype)}

    def acceleration_ratio(self, steps=1):
        """Acceleration ratio R = |du/dt + (u . grad) u| / (g |S_w|) per
        cell, (ny, nx), NaN when dry or S_w = 0. du/dt is the velocity change
        over ``steps`` time steps, which this call runs (the simulation
        advances); the rest is evaluated on the state after them."""
        inputs = self._acceleration_inputs(steps)
        return self._run_reading("_acc_cell", diag.ACC_CELL, inputs)[0]

    def face_acceleration_ratio(self, steps=1):
        """Acceleration ratio per face (x faces, y faces), as
        ``acceleration_ratio`` (runs ``steps`` time steps)."""
        inputs = self._acceleration_inputs(steps)
        return self._run_reading("_acc_face", diag.ACC_FACE, inputs)

    def acceleration_ratio_balance(self):
        """Acceleration ratio from the momentum balance du/dt + (u . grad) u
        = g S_w - g S_f: R = |tau_surface - tau_friction| / |tau_surface|
        per cell (``shear_stress``), NaN where tau_surface = 0. Relies on the
        friction law; does not advance the simulation."""
        sx, sy = self.shear_stress("surface")
        fx, fy = self.shear_stress("friction")
        return _ratio(sx, sy, fx, fy)

    def face_acceleration_ratio_balance(self):
        """``acceleration_ratio_balance`` per face (x faces, y faces), from
        ``face_shear_stress``."""
        (sxx, sxy), (syx, syy) = self.face_shear_stress("surface")
        (fxx, fxy), (fyx, fyy) = self.face_shear_stress("friction")
        return _ratio(sxx, sxy, fxx, fxy), _ratio(syx, syy, fyx, fyy)

    def _clock(self, index):
        return float(self._handle("clock").to_numpy()[index])

    program.initialize = initialize
    program.run = run
    program.run_until = run_until
    program.field = field
    program.face_flux = face_flux
    program.budget = budget
    program.set_probes = set_probes
    program.set_series = set_series
    program.records = records
    program._run_reading = _run_reading
    program.set_inflow_line = set_inflow_line
    program.set_line_discharge = set_line_discharge
    program.froude = froude
    program.face_froude = face_froude
    program._acceleration_inputs = _acceleration_inputs
    program.acceleration_ratio = acceleration_ratio
    program.face_acceleration_ratio = face_acceleration_ratio
    program.acceleration_ratio_balance = acceleration_ratio_balance
    program.face_acceleration_ratio_balance = face_acceleration_ratio_balance
    program.shear_stress = shear_stress
    program.face_shear_stress = face_shear_stress
    program._clock = _clock
    program.time = property(lambda self: self._clock(C_TIME))
    program.last_dt = property(lambda self: self._clock(C_DT))
    program.steps = property(lambda self: int(self._handle("control").to_numpy()[K_STEP]))
    program.restarts = property(
        lambda self: int(self._handle("control").to_numpy()[K_TOTAL_RESTARTS]))
    program.outlet_mask = property(
        lambda self: GridMaskAccessor(self, "OUTLET_MASK", "outlet='mask'"))
    program.nodata_mask = property(
        lambda self: GridMaskAccessor(self, "NODATA_MASK", "nodata=True"))
    return program


def build_fusw_program() -> type:
    """Build the FuswProgram class (see the module docstring)."""
    return finish_fusw_program(fusw_builder().freeze())


FuswProgram = build_fusw_program()

__all__ = ["FuswProgram", "build_fusw_program", "fusw_builder",
           "finish_fusw_program", "BOUNDARY_CODES"]
