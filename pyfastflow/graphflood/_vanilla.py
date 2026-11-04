"""Kernels of GraphFloodVanilla: explicit depth update and local transient.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder, RoutineBuilder
from pyfastflow.flow._program_cupy import _cupy_only, _grid_leaf_plan

from ._friction import build_friction_velocity


def _update_factory(be, bundles, config):
    """h += (Qi - Qo) dt / dx^2 with Qo the friction outflow at the frozen
    steepest slope of the current MFD topology."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    friction = build_friction_velocity(config["friction_law"])
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_update_depth(
                float* h, const float* Qi, float* Qo,
                const float* steepest_slope, const float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.nodata(i)$) {{
                h[i] = 0.0f;
                Qo[i] = 0.0f;
                return;
            }}
            if ($ctx.grid.can_out(i)$) {{
                Qo[i] = Qi[i];
                h[i] = 0.0f;
                return;
            }}
            float qout = $ctx.friction(h[i], steepest_slope[i], i)$
                * h[i] * flow_width[i];
            Qo[i] = qout;
            float dx = $ctx.grid.dx(i)$;
            float next = h[i] + (Qi[i] - qout) / (dx * dx) * $ctx.DT.get(0)$;
            h[i] = next > 0.0f ? next : 0.0f;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).compose("friction", friction).freeze()


def _update_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "h": "h", "Qi": "Qi", "Qo": "Qo",
        "steepest_slope": "steepest_slope", "flow_width": "flow_width",
        "MANNING": "friction_coefficient", "EXPO": "friction_exponent",
        "DT": "dt",
    })


def _transient_factory(be, bundles, config):
    """Conservative local MFD transport with no depression conditioning.

    topology: MFD on the live hydraulic surface (weights proportional to
    drop); outflow: Manning Qo at the steepest slope; limit: Qo at most what
    the cell holds over dt plus its rain; update: Qi from the donors' Qo
    split by their weights, h += (Qi - Qo) dt / dx^2.
    """
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    nk = 8 if config["topology"] == "D8" else 4

    topology = KernelBuilder(
        f'''extern "C" __global__ void graphflood_transient_topology(
                const double* hydraulic_surface, unsigned char* dirs,
                float* weights, float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            unsigned char mask = 0u;
            double scores[8];
            double sum_score = 0.0;
            double best_slope = 0.0;
            float best_width = $ctx.grid.dx(i)$;
            for (int k = 0; k < 8; ++k) scores[k] = 0.0;

            if (!$ctx.grid.nodata(i)$ && !$ctx.grid.can_out(i)$) {{
                for (int k = 0; k < {nk}; ++k) {{
                    int r = $ctx.grid.neighbour(i, k)$;
                    if (r == -1) continue;
                    double width = (double)$ctx.grid.width(i, k)$;
                    double slope = (hydraulic_surface[i]
                                  - hydraulic_surface[r]) / width;
                    if (slope <= 0.0) continue;
                    double score = slope * width;
                    scores[k] = score;
                    sum_score += score;
                    mask |= (unsigned char)(1u << k);
                    if (slope > best_slope) {{
                        best_slope = slope;
                        best_width = (float)width;
                    }}
                }}
            }}
            for (int k = 0; k < {nk}; ++k)
                weights[i * {nk} + k] = sum_score > 0.0
                    ? (float)(scores[k] / sum_score) : 0.0f;
            dirs[i] = mask;
            steepest_slope[i] = (float)best_slope;
            flow_width[i] = best_width;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    outflow = KernelBuilder(
        f'''extern "C" __global__ void graphflood_transient_outflow(
                const float* h, const unsigned char* dirs,
                const float* steepest_slope, const float* flow_width,
                float* Qo) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            Qo[i] = 0.0f;
            if ($ctx.grid.nodata(i)$ || $ctx.grid.can_out(i)$
                    || dirs[i] == 0u) return;
            float depth = fmaxf(h[i], 0.0f);
            float slope = fmaxf(steepest_slope[i], 0.0f);
            float width = fmaxf(flow_width[i], 1.0e-9f);
            float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
            float alpha = fmaxf(1.0f + $ctx.EXPO.get(i)$, 1.0e-3f);
            Qo[i] = width / manning * powf(depth, alpha) * sqrtf(slope);
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    limit = KernelBuilder(
        f'''extern "C" __global__ void graphflood_transient_limit(
                const float* h, float* Qo) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float dx = $ctx.grid.dx(i)$;
            float area = dx * dx;
            float dt = fmaxf($ctx.DT.get(0)$, 1.0e-12f);
            float available = area * fmaxf(h[i], 0.0f) / dt
                            + fmaxf($ctx.PRECIPITATION.get(i)$, 0.0f) * area;
            Qo[i] = fminf(Qo[i], available);
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    update = KernelBuilder(
        f'''extern "C" __global__ void graphflood_transient_update(
                float* h, float* Qi, float* Qo,
                const unsigned char* dirs, const float* weights) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.nodata(i)$) {{
                h[i] = 0.0f;
                Qi[i] = 0.0f;
                Qo[i] = 0.0f;
                return;
            }}
            float dx = $ctx.grid.dx(i)$;
            float area = dx * dx;
            float qin = fmaxf($ctx.PRECIPITATION.get(i)$, 0.0f) * area;
            for (int k = 0; k < {nk}; ++k) {{
                int donor = $ctx.grid.neighbour(i, k)$;
                if (donor == -1) continue;
                int reverse = $ctx.grid.reverse(k)$;
                if (!(dirs[donor] & (1u << reverse))) continue;
                qin += Qo[donor] * weights[donor * {nk} + reverse];
            }}
            Qi[i] = qin;
            if ($ctx.grid.can_out(i)$) {{
                h[i] = 0.0f;
                Qo[i] = qin;
                return;
            }}
            float dt = fmaxf($ctx.DT.get(0)$, 1.0e-12f);
            h[i] = fmaxf(h[i] + (qin - Qo[i]) * dt / area, 0.0f);
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    return (RoutineBuilder().step("topology", topology)
            .step("outflow", outflow).step("limit", limit)
            .step("update", update).freeze())


def _transient_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "hydraulic_surface": "hydraulic_surface", "h": "h",
        "Qi": "Qi", "Qo": "Qo", "dirs": "directions",
        "weights": "weights", "steepest_slope": "steepest_slope",
        "flow_width": "flow_width", "MANNING": "friction_coefficient",
        "EXPO": "friction_exponent", "PRECIPITATION": "precipitation",
        "DT": "dt",
    })
