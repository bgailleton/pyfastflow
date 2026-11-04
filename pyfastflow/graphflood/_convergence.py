"""Convergence measurement shared by the GraphFlood programs (CuPy only).

One check is one launch of ``measure_convergence`` over the grid (block
reduction, then a few global atomics), meant to run every N iterations. It
judges the current state on the live surface z + h, the same way for every
program, so the programs can be compared:

- gated cells: valid, non-outlet cells whose discharge (Qi, or the particle
  discharge field Qacc) is at least ``convergence_q_min``, set by
  ``reset_convergence`` to the ``convergence_percentile``-th percentile of
  the positive discharge;
- receiver: the steepest lower neighbour on z + h, else the lowest
  neighbour;
- dh: the newton depth residual, the exact local depth that passes the
  cell's discharge against its receiver's fixed head (newton helper of the
  particle walk, unrelaxed) minus h. It is in metres and vanishes on lakes,
  whose surface barely moves to pass any discharge. Its distribution goes
  into a 64-bin log10 histogram (1e-6 to 1e2 m, 8 bins per decade;
  quantiles interpolated log-linearly inside a bin) plus the exact maximum;
- Qo: Manning outflow of h at the steepest slope (floored at MIN_SLOPE), for
  the discharge residual sum|Q - Qo| / sum Q;
- drift: per cell, exponential averages (weight ``convergence_memory`` on the
  past) of the signed and absolute change of h between checks;
  r = |mean| / mean|.| is ~1 for a cell still moving one way and ~0 for one
  oscillating around its fixed point;
- flips: gated cells whose receiver changed since the last check.

The particle variant adds the routing staleness sum|Qacc - (rain + sends of
the higher neighbours)| / sum Qacc on gated cells, the outlet balance
(water sent into outlets plus rain on outlets) / total rain, and the
coverage: fraction of the active area whose visit count grew by at least
``convergence_coverage_min`` since the last check.

Host side (attached by ``attach_convergence``): ``reset_convergence()``,
``convergence()`` (one check, returns the metrics) and
``run_until_converged(check_every, ...)``, which stops on
``ConvergenceRule``: the chosen dh quantile below ``tol`` (metres), or no new
minimum over the last ``window`` checks (plateau: the solver's own floor).

Author: B.G (10/2026)
"""

import math
import time

from pyfastflow.core import ProgramError
from pyfastflow.flow._program_cupy import _grid_leaf_plan

from ._base import FLAT
from ._particle import _kernel, newton_depth_helper

BINS = 64
BIN_LOG_MIN = -6.0
BINS_PER_DECADE = 8
N_ACC = 16
BLOCK = 256

# Accumulator slots.
_CELLS, _DH_ABS, _DH, _Q, _RES_ABS, _RES, _DRIFT_Q, _FLIPS, _DRIFT_RQ, \
    _STALE, _OUTLET, _RAIN, _ACTIVE, _COVERED = range(14)


def _measure_text(n, particles):
    qtype = "double" if particles else "float"
    extra_args = (''',
        const float* pushed, const unsigned int* visits,
        unsigned int* visits_prev, const int* reach''' if particles else "")
    rain_balance = '''
        v[11] = (double)rain;
        if (out) v[10] += (double)rain;''' if particles else ""
    inflow_init = "double inflow = (double)rain;" if particles else ""
    neighbour_extra = '''
                if (other - head > DROP_MIN)
                    inflow += (double)pushed[j * nk + $ctx.grid.reverse(k)$];
                if ($ctx.grid.can_out(j)$)
                    v[10] += (double)pushed[i * nk + k];''' if particles else ""
    stale = "v[9] = fabs(q - inflow);" if particles else ""
    coverage = '''
            if (reach[i]) {
                v[12] = 1.0;
                if (visits[i] - visits_prev[i]
                        >= (unsigned int)$ctx.COVERAGE_MIN.get(0)$)
                    v[13] = 1.0;
            }
            visits_prev[i] = visits[i];''' if particles else ""
    return f'''
#define CONV_NACC {N_ACC}
#define CONV_BINS {BINS}
extern "C" __global__ void graphflood_convergence(
        const float* z, const float* h, const {qtype}* Q, float* h_prev,
        float* drift_mean, float* drift_abs, unsigned char* steepest,
        double* acc, unsigned int* hist{extra_args}) {{
    __shared__ double s_acc[CONV_NACC];
    __shared__ unsigned int s_hist[CONV_BINS + 1];
    if (threadIdx.x < CONV_NACC) s_acc[threadIdx.x] = 0.0;
    if (threadIdx.x <= CONV_BINS) s_hist[threadIdx.x] = 0u;
    __syncthreads();

    double v[CONV_NACC];
    for (int c = 0; c < CONV_NACC; ++c) v[c] = 0.0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {n} && !$ctx.grid.nodata(i)$) {{
        int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
        float dx = $ctx.grid.dx(i)$;
        float rain = $ctx.PRECIPITATION.get(i)$ * dx * dx;
        bool out = $ctx.grid.can_out(i)$;{rain_balance}
        if (!out) {{
            double q = (double)Q[i];
            float depth = fmaxf(h[i], 0.0f);
            double zi = (double)z[i];
            double head = zi + (double)depth;
            int rk = 255, low_k = 255;
            float best_slope = 0.0f, width = dx, length = dx, low_length = dx;
            double receiver_head = 1.0e300, low_head = 1.0e300;
            {inflow_init}
            for (int k = 0; k < nk; ++k) {{
                int j = $ctx.grid.neighbour(i, k)$;
                if (j < 0 || $ctx.grid.nodata(j)$) continue;
                double other = (double)z[j] + (double)h[j];
                float dist = $ctx.grid.dist(i, k)$;
                if (other < low_head) {{
                    low_head = other;
                    low_k = k;
                    low_length = dist;
                }}
                double drop = head - other;
                if (drop > 0.0 && (float)(drop / dist) > best_slope) {{
                    best_slope = (float)(drop / dist);
                    rk = k;
                    width = $ctx.grid.flow_width(i, k)$;
                    length = dist;
                    receiver_head = other;
                }}{neighbour_extra}
            }}
            if (rk == 255) {{
                rk = low_k;
                receiver_head = low_head;
                if (low_k != 255) width = $ctx.grid.flow_width(i, low_k)$;
                length = low_length;
            }}

            // Drift averages of every valid non-outlet cell.
            float memory = fminf(fmaxf($ctx.MEMORY.get(0)$, 0.0f), 1.0f);
            float r = 0.0f;
            bool first = $ctx.FIRST.get(0)$ != 0;
            if (first) {{
                drift_mean[i] = 0.0f;
                drift_abs[i] = 0.0f;
            }} else {{
                float step = h[i] - h_prev[i];
                float m = memory * drift_mean[i] + (1.0f - memory) * step;
                float a = memory * drift_abs[i]
                        + (1.0f - memory) * fabsf(step);
                drift_mean[i] = m;
                drift_abs[i] = a;
                r = a > 1.0e-9f ? fabsf(m) / a : 0.0f;
            }}

            if (q > 0.0 && q >= (double)$ctx.Q_MIN.get(0)$) {{
                float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
                float alpha = fmaxf(1.0f + $ctx.FRICTION_EXPONENT.get(i)$,
                                    1.0e-3f);
                float qout = width / manning * powf(depth, alpha)
                           * sqrtf(fmaxf(best_slope, MIN_SLOPE));
                float target = receiver_head < 1.0e299
                    ? $ctx.hupdate(depth, (float)q, width,
                                   (float)(zi - receiver_head), length,
                                   manning, alpha, 1.0f)$
                    : depth;
                float dh = target - depth;
                float adh = fabsf(dh);
                v[0] = 1.0;
                v[1] = (double)adh;
                v[2] = (double)dh;
                v[3] = q;
                v[4] = fabs(q - (double)qout);
                v[5] = q - (double)qout;
                int bin = adh > 0.0f
                    ? (int)floorf((log10f(adh) - ({BIN_LOG_MIN}f))
                                  * {BINS_PER_DECADE}.0f)
                    : 0;
                bin = min(max(bin, 0), CONV_BINS - 1);
                atomicAdd(&s_hist[bin], 1u);
                atomicMax(&s_hist[CONV_BINS], __float_as_uint(adh));
                if (!first) {{
                    v[6] = r > 0.5f ? q : 0.0;
                    v[7] = steepest[i] != (unsigned char)rk ? 1.0 : 0.0;
                    v[8] = q * (double)r;
                }}
                {stale}
            }}
            h_prev[i] = h[i];
            steepest[i] = (unsigned char)rk;{coverage}
        }}
    }}

    for (int c = 0; c < CONV_NACC; ++c) {{
        double x = v[c];
        for (int offset = 16; offset > 0; offset >>= 1)
            x += __shfl_down_sync(0xffffffffu, x, offset);
        if ((threadIdx.x & 31) == 0 && x != 0.0) atomicAdd(&s_acc[c], x);
    }}
    __syncthreads();
    if (threadIdx.x < CONV_NACC && s_acc[threadIdx.x] != 0.0)
        atomicAdd(&acc[threadIdx.x], s_acc[threadIdx.x]);
    if (threadIdx.x < CONV_BINS && s_hist[threadIdx.x] != 0u)
        atomicAdd(&hist[threadIdx.x], s_hist[threadIdx.x]);
    if (threadIdx.x == CONV_BINS)
        atomicMax(&hist[CONV_BINS], s_hist[CONV_BINS]);
}}
'''


def add_convergence(b, particles=False):
    """Add the convergence params, buffers and ``measure_convergence``.

    ``particles``: measure the discharge field Qacc and add staleness,
    outlet balance and coverage (needs Qacc, pushed, visits and reach).
    """
    b.param("convergence_percentile", "auto", "f32", value=90.0)
    b.param("convergence_memory", "auto", "f32", value=0.8)
    b.param("convergence_q_min", "scalar", "f32", value=0.0)
    b.param("convergence_first", "scalar", "i32", value=1)
    for data in ("conv_h_prev", "conv_drift_mean", "conv_drift_abs"):
        b.data(data, "f32", (FLAT,), role="internal")
    b.data("conv_steepest", "u8", (FLAT,), role="internal")
    b.data("conv_acc", "f64", (N_ACC,), role="internal")
    b.data("conv_hist", "u32", (BINS + 1,), role="internal")
    leaves = {
        "z": "z", "h": "h", "Q": "Qacc" if particles else "Qi",
        "h_prev": "conv_h_prev", "drift_mean": "conv_drift_mean",
        "drift_abs": "conv_drift_abs", "steepest": "conv_steepest",
        "acc": "conv_acc", "hist": "conv_hist",
        "PRECIPITATION": "precipitation", "MANNING": "friction_coefficient",
        "FRICTION_EXPONENT": "friction_exponent",
        "Q_MIN": "convergence_q_min", "MEMORY": "convergence_memory",
        "FIRST": "convergence_first",
    }
    if particles:
        b.param("convergence_coverage_min", "auto", "i32", value=1)
        b.data("conv_visits_prev", "u32", (FLAT,), role="internal")
        leaves.update({
            "pushed": "pushed", "visits": "visits",
            "visits_prev": "conv_visits_prev", "reach": "reach",
            "COVERAGE_MIN": "convergence_coverage_min",
        })

    def factory(be, bundles, config):
        n = config["nx"] * config["ny"]
        return _kernel(be, bundles, _measure_text(n, particles),
                       -(-n // BLOCK) * BLOCK, block=BLOCK,
                       helpers={"hupdate": newton_depth_helper()})

    b.add("measure_convergence", factory,
          bind=lambda f, be: _grid_leaf_plan(f, leaves))
    return b


def _quantile(hist, q):
    """Quantile of the histogram, log-linear inside the bin it falls in."""
    total = int(hist.sum())
    if total == 0:
        return 0.0
    target = q * total
    running = 0
    for b, count in enumerate(hist):
        count = int(count)
        if count and running + count >= target:
            fraction = (target - running) / count
            return 10.0 ** (BIN_LOG_MIN + (b + fraction) / BINS_PER_DECADE)
        running += count
    return 10.0 ** (BIN_LOG_MIN + len(hist) / BINS_PER_DECADE)


class ConvergenceRule:
    """Stop rule on one metric of ``convergence()`` (default ``dh_p99``).

    ``update(metrics)`` returns ``"tolerance"`` once the metric is below
    ``tol``, ``"plateau"`` once the best value of the last ``window`` checks
    is not lower than (1 - eps) times the best before them, else None.
    """

    def __init__(self, tol=1.0e-3, metric="dh_p99", window=10, eps=0.01):
        self.tol, self.metric = float(tol), metric
        self.window, self.eps = int(window), float(eps)
        self.values = []

    def update(self, metrics):
        value = float(metrics[self.metric])
        self.values.append(value)
        if value < self.tol:
            return "tolerance"
        if len(self.values) > self.window:
            before = min(self.values[:-self.window])
            if min(self.values[-self.window:]) > (1.0 - self.eps) * before:
                return "plateau"
        return None


def attach_convergence(program, discharge="Qi", particles=False):
    """Attach the host convergence methods to ``program`` (see module doc)."""

    def reset_convergence(self):
        """Set the gate from the current discharge and restart drift/flips."""
        import cupy as cp

        q = self._handle(discharge).array.ravel()
        positive = q[q > 0.0]
        if positive.size == 0:
            raise ProgramError("no discharge yet: run the solver first")
        self.convergence_q_min.set(float(cp.percentile(
            positive, float(self.convergence_percentile.read()))))
        self.convergence_first.set(1)
        if particles:
            self._handle("conv_visits_prev").array[...] = \
                self._handle("visits").array
        object.__setattr__(self, "_convergence_ready", True)

    def convergence(self):
        """One check; returns the metrics (see module doc).

        Keys: ``cells`` (gated), ``dh_mean``, ``dh_p50``, ``dh_p90``,
        ``dh_p99``, ``dh_max`` (m; quantiles log-interpolated in the
        histogram bins),
        ``fill_bias`` (sum dh / sum|dh|: +1 all cells must rise, -1 drain),
        ``residual`` and ``residual_signed`` (sum|Q - Qo|, sum(Q - Qo) over
        sum Q), ``drift`` (Q-weighted fraction of gated cells with r > 0.5),
        ``drift_mean`` (Q-weighted mean r), ``flips`` (fraction of gated
        cells whose receiver changed); particles add ``staleness``,
        ``outlet_balance`` and ``coverage``. Drift and flips are NaN on the
        first check after a reset.
        """
        if not getattr(self, "_convergence_ready", False):
            self.reset_convergence()
        first = int(self.convergence_first.read()) != 0
        self._handle("conv_acc").array.fill(0)
        self._handle("conv_hist").array.fill(0)
        self.measure_convergence()
        self.convergence_first.set(0)
        acc = self._handle("conv_acc").to_numpy()
        raw = self._handle("conv_hist").to_numpy()
        hist = raw[:BINS]
        cells, q = acc[_CELLS], acc[_Q]
        nan = float("nan")

        def ratio(a, b):
            return float(a / b) if b > 0 else nan

        metrics = {
            "cells": int(cells),
            "dh_mean": ratio(acc[_DH_ABS], cells),
            "dh_p50": _quantile(hist, 0.50),
            "dh_p90": _quantile(hist, 0.90),
            "dh_p99": _quantile(hist, 0.99),
            "dh_max": float(raw[BINS:BINS + 1].view("float32")[0]),
            "fill_bias": ratio(acc[_DH], acc[_DH_ABS]),
            "residual": ratio(acc[_RES_ABS], q),
            "residual_signed": ratio(acc[_RES], q),
            "drift": nan if first else ratio(acc[_DRIFT_Q], q),
            "drift_mean": nan if first else ratio(acc[_DRIFT_RQ], q),
            "flips": nan if first else ratio(acc[_FLIPS], cells),
        }
        if particles:
            metrics["staleness"] = ratio(acc[_STALE], q)
            metrics["outlet_balance"] = ratio(acc[_OUTLET], acc[_RAIN])
            metrics["coverage"] = (nan if first
                                   else ratio(acc[_COVERED], acc[_ACTIVE]))
        return metrics

    def run_until_converged(self, check_every, max_checks=100, tol=1.0e-3,
                            metric="dh_p99", window=10, eps=0.01,
                            callback=None):
        """``run(check_every)`` then ``convergence()`` until the rule stops.

        Returns ``{"reason", "history"}``; reason is ``"tolerance"``,
        ``"plateau"`` or ``"max_checks"``. Each history entry is the metrics
        plus ``check``, ``iterations`` (cumulated run units) and ``time``
        (s since the call, synchronised by the check); a dict returned by
        ``run`` (particle statistics) is merged with keys prefixed
        ``run_``. ``callback(metrics)`` is called after each check.
        """
        rule = ConvergenceRule(tol=tol, metric=metric, window=window, eps=eps)
        history = []
        start = time.perf_counter()
        reason = "max_checks"
        for check in range(1, int(max_checks) + 1):
            out = self.run(int(check_every))
            metrics = self.convergence()
            metrics["check"] = check
            metrics["iterations"] = check * int(check_every)
            metrics["time"] = time.perf_counter() - start
            if isinstance(out, dict):
                metrics.update({f"run_{k}": v for k, v in out.items()})
            history.append(metrics)
            if callback is not None:
                callback(metrics)
            if math.isnan(float(metrics[metric])):
                continue
            stop = rule.update(metrics)
            if stop is not None:
                reason = stop
                break
        return {"reason": reason, "history": history}

    program.reset_convergence = reset_convergence
    program.convergence = convergence
    program.run_until_converged = run_until_converged
    return program
