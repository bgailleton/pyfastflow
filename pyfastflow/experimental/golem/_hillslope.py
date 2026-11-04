"""Implicit, conservative bare-surface hillslope diffusion for GOLEM."""

from pyfastflow.core import KernelBuilder, SequenceBuilder
from pyfastflow.flow._program_cupy import _cupy_only, _grid_leaf_plan, _noop_factory


def hillslope_factory(be, bundles, config):
    """Build Jacobi iterations for ``z_t - dt div(D grad z_t) = z_0``.

    Face diffusivity is the arithmetic mean of the adjacent fields, making
    the discrete internal flux antisymmetric.  Therefore the operator
    conserves volume on a closed active domain; fixed outlet cells are the
    only permitted external source/sink.
    """
    _cupy_only(be)
    n = int(config["nx"]) * int(config["ny"])
    iterations = int(config["diffusion_iterations"])
    if iterations < 0:
        raise ValueError("diffusion_iterations must be non-negative")
    if iterations == 0:
        return _noop_factory(be, bundles, config)

    initialize = KernelBuilder(
        f'''extern "C" __global__ void golem_diffusion_initialize(
                const float* z, float* z0, float* guess) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            z0[i] = z[i];
            guess[i] = z[i];
        }}''', domain=n,
    ).freeze()

    def relax(name, src, dst):
        return KernelBuilder(
            f'''extern "C" __global__ void golem_diffusion_{name}(
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
                    if (j == -1) continue;
                    float dj = fmaxf($ctx.DIFFUSIVITY.get(j)$, 0.0f);
                    float length = $ctx.grid.dist(i, k)$;
                    float a = dt * 0.5f * (di + dj) / (length * length);
                    numerator += a * src[j];
                    denominator += a;
                }}
                dst[i] = numerator / denominator;
            }}''', domain=n,
        ).compose("grid", bundles["grid"]).freeze()

    forward = relax("forward", "a", "b")
    backward = relax("backward", "b", "a")
    finish_from_a = KernelBuilder(
        f'''extern "C" __global__ void golem_diffusion_finish_a(
                const float* a, float* z) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) z[i] = a[i];
        }}''', domain=n,
    ).freeze()
    finish_from_b = KernelBuilder(
        f'''extern "C" __global__ void golem_diffusion_finish_b(
                const float* b, float* z) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) z[i] = b[i];
        }}''', domain=n,
    ).freeze()

    sequence = SequenceBuilder()
    sequence.add("initialize", initialize)
    if iterations >= 1:
        sequence.add("forward", forward)
    if iterations >= 2:
        sequence.add("backward", backward)
    sequence.step("initialize")
    for _ in range(iterations // 2):
        sequence.step("forward").step("backward")
    if iterations % 2:
        sequence.add("finish_b", finish_from_b)
        sequence.step("forward").step("finish_b")
    else:
        sequence.add("finish_a", finish_from_a)
        sequence.step("finish_a")
    return sequence.freeze()


def hillslope_plan(frozen, _be):
    plan = _grid_leaf_plan(frozen, {
        "z": "z", "z0": "diffusion_z0", "guess": "diffusion_a",
        "src": "diffusion_a", "dst": "diffusion_b",
        "a": "diffusion_a", "b": "diffusion_b",
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
