"""GraphFlood relaxation to steady state on a quadtree refined by its own results.

Advance the solver until it is steady, flag leaves from ``h`` or the velocity,
then refine them. The bed is re-aggregated from the source DEM and depth moves
with a flat water surface and its volume kept, so each solve starts from the
last.

Steadiness is read from :class:`WaterLeft`: the share of the water that still
has to move. An optional fast phase first refines ``fast_cycles`` times, every
``fast_iterations`` iterations at ``fast_relaxation``, without waiting for it.

Author: B.G (11/2026)
"""

import cupy as cp
import numpy as np

from ..quadtree import DynamicQuadtree, transfer
from ._convergence import ConvergenceRule
from .relax import GraphFloodRelax

CRITERIA = ("h", "speed", "u", "v")


class WaterLeft:
    """Share of the water still to move before ``h`` settles.

    A depth relaxed by ``r`` per iteration closes a fraction ``1 - (1 - r)**N``
    of its remaining distance in ``N`` iterations, so the distance left is the
    change seen over those iterations divided by that fraction:

        left = sum(area * |h - h_before|) / sum(area * h) / (1 - (1 - r)**N)

    A relaxation large enough for ``h`` to flicker leaves ``left`` on a floor,
    which the plateau rule catches.

    Parameters
    ----------
    relaxation : float
        Relaxation per iteration.
    iterations : int
        Iterations between two calls of :meth:`update`.
    tol : float
        Steady below this share; 0.02 is 2 % of the water.
    window, eps : int, float
        Steady as well without a new minimum, by a relative ``eps``, over the
        last ``window`` calls.
    """

    def __init__(self, relaxation, iterations, tol=0.02, window=10, eps=0.01):
        self.closed = 1.0 - (1.0 - float(relaxation)) ** int(iterations)
        self.rule = ConvergenceRule(tol=tol, metric="left", window=window, eps=eps)
        self.previous = None

    def update(self, h, area):
        """Return ``(left, reason)``; ``reason`` is ``None`` until steady."""
        h = h.astype(cp.float64)
        if self.previous is None:
            self.previous = h
            return float("inf"), None
        volume = float((h * area).sum())
        moved = float((cp.abs(h - self.previous) * area).sum())
        self.previous = h
        left = moved / max(volume, 1.0e-30) / self.closed
        return left, self.rule.update({"left": left})


class QuadtreeFlood:
    """Steady-state flood solver alternating GraphFloodRelax and refinement.

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
    levels : int
        Starting uniform level; leaves have side ``dx * 2**levels``.
    criterion : {"h", "speed", "u", "v"}
        Quantity refined on, compared in absolute value. ``speed`` is
        ``Qo / (h * width)``; ``u`` and ``v`` are its x and y components along
        the MFD weights.
    threshold : float
        A leaf is flagged for refinement at or above this value.
    coarsen_below : float, optional
        A leaf is flagged for coarsening below this value.
    capacity : int, optional
        Largest leaf count; defaults to half the finest raster. Program cost
        follows this number, not the current leaf count.
    manning, relaxation, solver, local_minima
        GraphFloodRelax ``friction_coefficient``, ``relaxation``,
        ``analytical_solver`` and ``mfd_local_minima``.
    iterations_per_check, tol, window, eps
        Steady state, see :class:`WaterLeft`.
    fast_cycles, fast_relaxation, fast_iterations
        Fast phase: refinements done without a steady-state check, the
        relaxation used for them and the iterations between them.
    """

    def __init__(self, be, dem, dx, rain, *, levels=5, criterion="h", threshold=0.05,
                 coarsen_below=None, capacity=None, manning=0.033, relaxation=0.1,
                 solver="bottom_up", local_minima="carve_cordonnier",
                 iterations_per_check=20, tol=0.02, window=10, eps=0.01,
                 fast_cycles=0, fast_relaxation=0.2, fast_iterations=5):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
        dem = np.asarray(dem, dtype=np.float32)
        self.be, self.criterion, self.threshold = be, criterion, float(threshold)
        self.coarsen_below = coarsen_below
        self.relaxation, self.iterations_per_check = float(relaxation), int(iterations_per_check)
        self.tol, self.window, self.eps = tol, window, eps
        self.fast_cycles, self.fast_relaxation = int(fast_cycles), float(fast_relaxation)
        self.fast_iterations = int(fast_iterations)
        self.rain, self.dx = float(rain), float(dx)
        self.pool = be.PoolCls()
        capacity = dem.size // 2 if capacity is None else int(capacity)
        self.tree = DynamicQuadtree(
            be, self.pool, np.full(dem.shape, levels, np.int32), dx, levels,
            topology="D4", capacity=capacity,
        )
        self.gf = GraphFloodRelax(
            be, tree=self.tree, topology="D4", analytical_solver=solver,
            mfd_local_minima=local_minima, precipitation=self.rain,
            friction_coefficient=manning, **self.tree.dims,
        )
        self.bed = transfer.FromRaster(dem)
        self.gf.z.array[...] = self._padded_bed()
        self.gf.reset_h()
        self.stop_reason = None
        self.iterations = 0
        self._fresh = True
        self._warm = False

    def _padded_bed(self):
        tree = self.tree
        out = cp.zeros(tree.capacity, cp.float32)
        out[: tree.n] = self.bed.sample(tree, tree.level, tree.x, tree.y, cp.float32)
        return out

    @property
    def h(self):
        """Water depth per leaf, as a device array of ``capacity`` entries."""
        return self.gf.h.array.reshape(-1)

    @property
    def z(self):
        """Bed elevation per leaf, as a device array of ``capacity`` entries."""
        return self.gf.z.array.reshape(-1)

    def to_raster(self, field):
        """Return a per-leaf field as a host array on the finest raster."""
        return self.tree.to_raster(field)

    def speed(self):
        """Return ``Qo / (h * width)`` per leaf, zero where dry."""
        h = self.h
        width = self.gf._handle("flow_width").array.reshape(-1)
        q = self.gf.Qo.array.reshape(-1)
        wet = h > 1.0e-6
        return cp.where(wet, q / cp.where(wet, h * width, 1.0), 0.0)

    def velocity(self):
        """Return ``(u, v)`` per leaf: the speed along the MFD weights of the D4 faces."""
        speed = self.speed()
        w = self.gf._handle("weights").array.reshape(self.tree.capacity, -1)
        return (speed * (w[:, 4] + w[:, 5] - w[:, 2] - w[:, 3]),
                speed * (w[:, 6] + w[:, 7] - w[:, 0] - w[:, 1]))

    def indicator(self):
        """Return the absolute value of the refinement quantity per current leaf."""
        if self.criterion == "h":
            value = self.h
        elif self.criterion == "speed":
            value = self.speed()
        else:
            value = self.velocity()[0 if self.criterion == "u" else 1]
        return cp.abs(value[: self.tree.n])

    def flags(self):
        """Return the per-leaf update flags for the current state."""
        value = self.indicator()
        out = cp.zeros(self.tree.n, cp.int32)
        if self.coarsen_below is not None:
            out[value < self.coarsen_below] = -1
        out[(value >= self.threshold) & (self.tree.level > 0)] = 1
        return out

    def advance(self):
        """Run one check's iterations; return ``(left, reason)``, see :class:`WaterLeft`."""
        if not self._warm:
            self.gf.warmup()
            self._warm = True
        if self._fresh:
            self._fast = self.fast_cycles > 0
            relaxation = self.fast_relaxation if self._fast else self.relaxation
            self._count = self.fast_iterations if self._fast else self.iterations_per_check
            self.gf.relaxation.set(relaxation)
            self._check = WaterLeft(relaxation, self._count,
                                    tol=self.tol, window=self.window, eps=self.eps)
            self._area = (self.dx * self.dx) * 4.0 ** self.tree.level.astype(cp.float64)
            self._check.update(self.h[: self.tree.n], self._area)
            self.iterations = 0
            self._fresh = False
        self.gf.run(self._count)
        self.iterations += self._count
        left, reason = self._check.update(self.h[: self.tree.n], self._area)
        return left, "fast" if self._fast else reason

    def refine(self):
        """Apply the flags; return the number of leaves added, or ``None`` when none could be."""
        flags = self.flags()
        before = self.tree.n
        self._fresh = True
        self.fast_cycles = max(self.fast_cycles - 1, 0)
        try:
            after = self.tree.adapt(flags, [(self.gf.z, self.bed), (self.gf.h, transfer.Depth(self.gf.z))])
        except ValueError as error:
            if "exceed" not in str(error):
                raise
            self.stop_reason = str(error)
            self.fast_cycles = 0
            return None
        if after == before:
            self.stop_reason = "no leaf changed"
            self.fast_cycles = 0
            return None
        return after - before

    def close(self):
        """Release the solver and the tree."""
        self.gf.close()
        self.tree.close()
