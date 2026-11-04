"""Optional implicit surface diffusion with sediment-first phase allocation."""

from pyfastflow.core import KernelBuilder, SequenceBuilder
from pyfastflow.flow._program_cupy import _cupy_only, _grid_leaf_plan, _noop_factory


def sediment_hillslope_factory(be, bundles, config):
    """Build implicit linear diffusion and update the sediment-layer phase.

    The implicit solve evolves the total surface. Its local erosion first
    consumes sediment thickness; positive surface change becomes deposited
    sediment. This is a conservative surface-diffusion closure, not a
    separate grain-size-resolved hillslope transport law.
    """
    _cupy_only(be)
    n = int(config["nx"]) * int(config["ny"])
    iterations = int(config["diffusion_iterations"])
    if iterations < 0:
        raise ValueError("diffusion_iterations must be non-negative")
    if iterations == 0:
        return _noop_factory(be, bundles, config)

    initialize = KernelBuilder(
        f'''extern "C" __global__ void golem_sed_diffusion_initialize(
                const float* z, float* z0, float* a) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) {{ z0[i] = z[i]; a[i] = z[i]; }}
        }}''', domain=n,
    ).freeze()

    def relax(name):
        return KernelBuilder(
            f'''extern "C" __global__ void golem_sed_diffusion_{name}(
                    const float* z0, const float* src, float* dst) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= {n}) return;
                if (!$ctx.grid.is_active(i)$ || $ctx.grid.can_out(i)$) {{
                    dst[i] = z0[i];
                    return;
                }}
                float di = fmaxf($ctx.DIFFUSIVITY.get(i)$, 0.0f);
                float dt = fmaxf($ctx.DT.get(i)$, 0.0f);
                float numerator = z0[i];
                float denominator = 1.0f;
                int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
                for (int k = 0; k < nk; ++k) {{
                    int j = $ctx.grid.neighbour(i, k)$;
                    if (j < 0) continue;
                    float dj = fmaxf($ctx.DIFFUSIVITY.get(j)$, 0.0f);
                    float length = $ctx.grid.dist(i, k)$;
                    float weight = dt * 0.5f * (di + dj) / (length * length);
                    numerator += weight * src[j];
                    denominator += weight;
                }}
                dst[i] = numerator / denominator;
            }}''', domain=n,
        ).compose("grid", bundles["grid"]).freeze()

    forward, backward = relax("forward"), relax("backward")
    finish_a = KernelBuilder(
        f'''extern "C" __global__ void golem_sed_diffusion_finish_a(
                const float* a, float* z) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) z[i] = a[i];
        }}''', domain=n,
    ).freeze()
    finish_b = KernelBuilder(
        f'''extern "C" __global__ void golem_sed_diffusion_finish_b(
                const float* b, float* z) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) z[i] = b[i];
        }}''', domain=n,
    ).freeze()
    allocate_phase = KernelBuilder(
        f'''extern "C" __global__ void golem_sed_diffusion_allocate_phase(
                const float* z0, const float* z, float* sediment_thickness) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n} || !$ctx.grid.is_active(i)$ || $ctx.grid.can_out(i)$) return;
            float change = z[i] - z0[i];
            float h = fmaxf(sediment_thickness[i], 0.0f);
            sediment_thickness[i] = change >= 0.0f ? h + change
                : fmaxf(0.0f, h + change);
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    sequence = SequenceBuilder()
    sequence.add("initialize", initialize)
    if iterations >= 1:
        sequence.add("forward", forward)
    if iterations >= 2:
        sequence.add("backward", backward)
    sequence.add("allocate_phase", allocate_phase)
    sequence.step("initialize")
    for _ in range(iterations // 2):
        sequence.step("forward").step("backward")
    if iterations % 2:
        sequence.add("finish_b", finish_b)
        sequence.step("forward").step("finish_b")
    else:
        sequence.add("finish_a", finish_a)
        sequence.step("finish_a")
    sequence.step("allocate_phase")
    return sequence.freeze()


def sediment_hillslope_plan(frozen, _be):
    """Bind the diffusion buffers and sediment thickness."""
    plan = _grid_leaf_plan(frozen, {
        "z": "z", "z0": "diffusion_z0", "a": "diffusion_a", "b": "diffusion_b",
        "src": "diffusion_a", "dst": "diffusion_b",
        "sediment_thickness": "sediment_thickness",
        "DIFFUSIVITY": "hillslope_diffusivity", "DT": "dt",
    })
    bound = frozen.build()
    try:
        addresses = set(bound.addresses())
    finally:
        bound.close()
    if ("backward", "src") in addresses:
        plan.update({
            "backward.src": "diffusion_b",
            "backward.dst": "diffusion_a",
        })
    return plan


__all__ = ["sediment_hillslope_factory", "sediment_hillslope_plan"]
