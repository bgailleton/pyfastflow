"""
cupy (CUDA source) block templates behind make_hillshade_group /
make_hillshade_kernel.

Mirrors _closure_blocks.py: same gradient rewrite (grid.neighbour(i, k),
falling back to z[i] where there is no neighbour - see that module's own
docstring for why), same hillshade formula, written as CUDA text. No
`ctx.bk` here - cupy stays plain C (`sqrtf`/`atan2f`/`cosf`/`sinf`,
`fmaxf`/`fminf`), same as grid/_cupy_blocks.py and noise/_cupy_blocks.py.
Every `__device__`/`__global__` symbol is prefixed with this build's own tag
(a fresh new_uid()) so two make_hillshade_group()/make_hillshade_kernel()
calls in one process never collide inside a single compiled cupy module.

Author: B.G (08/2026)
"""

from ..core import KernelBuilder, freeze_helper as _helper, new_uid


def build_group(group, *, grid, k_top, k_left, k_right, k_bottom):
    """
    Compose `at(z, i)` (and its private `grad_x`/`grad_y`) onto `group` (a
    GroupBuilder) for the cupy backend. `grid` (a FrozenGroup) is composed
    independently under `grad_x` and `grad_y` - see __init__.py's own module
    docstring.

    Returns nothing - `at` is compose()d onto `group` itself, under its own
    public name, by this call.

    Author: B.G (08/2026)
    """
    t = f"pf{new_uid()}"

    grad_x = _helper(
        f"""
__device__ float {t}_gradient_x(const float* z, int i) {{
    int zl = $ctx.GRID.neighbour(i, {k_left})$;
    int zr = $ctx.GRID.neighbour(i, {k_right})$;
    float left_val = (zl != -1) ? z[zl] : z[i];
    float right_val = (zr != -1) ? z[zr] : z[i];
    return (right_val - left_val) / (2.0f * $ctx.GRID.DX.get(0)$);
}}
""",
        helpers={"GRID": grid},
    )
    grad_y = _helper(
        f"""
__device__ float {t}_gradient_y(const float* z, int i) {{
    int zt = $ctx.GRID.neighbour(i, {k_top})$;
    int zb = $ctx.GRID.neighbour(i, {k_bottom})$;
    float top_val = (zt != -1) ? z[zt] : z[i];
    float bottom_val = (zb != -1) ? z[zb] : z[i];
    return (bottom_val - top_val) / (2.0f * $ctx.GRID.DX.get(0)$);
}}
""",
        helpers={"GRID": grid},
    )
    at = _helper(
        f"""
__device__ float {t}_at(const float* z, int i) {{
    float dzdx = $ctx.grad_x(z, i)$ * $ctx.ZFACTOR.get(0)$;
    float dzdy = $ctx.grad_y(z, i)$ * $ctx.ZFACTOR.get(0)$;

    float slope_rad = atan2f(sqrtf(dzdx * dzdx + dzdy * dzdy), 1.0f);
    float azimuth_rad = $ctx.AZIMUTH.get(0)$ * 0.017453292519943295f;
    float zenith_rad = 1.5707963267948966f - $ctx.ALTITUDE.get(0)$ * 0.017453292519943295f;

    float aspect_rad = 0.0f;
    if (dzdx != 0.0f || dzdy != 0.0f) {{
        aspect_rad = 1.5707963267948966f - atan2f(dzdy, dzdx);
        if (aspect_rad < 0.0f) aspect_rad += 6.283185307179586f;
    }}

    float hillshade_value = cosf(zenith_rad) * cosf(slope_rad)
        + sinf(zenith_rad) * sinf(slope_rad) * cosf(azimuth_rad - aspect_rad);
    return fmaxf(0.0f, fminf(1.0f, hillshade_value));
}}
""",
        helpers={"grad_x": grad_x, "grad_y": grad_y},
    )
    group.compose("at", at)


def build_kernel(hillshade_group):
    """
    The standalone `hillshade` FrozenKernel for the cupy backend - see
    __init__.py's own `make_hillshade_kernel`. `n` (the node count) is a
    third DATA slot, unlike the closure backends: a `cp.RawModule` kernel
    has no auto-ranging equivalent to Taichi/Quadrants' `for i in
    range(n)`.

    Author: B.G (08/2026)
    """
    t = f"pf{new_uid()}"
    template = f"""
extern "C" __global__ void {t}_hillshade(const float* z, float* out, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = $ctx.hillshade.at(z, i)$;
}}
"""
    return (
        KernelBuilder(template, domain="z")
        .compose("hillshade", hillshade_group)
        .freeze()
    )
