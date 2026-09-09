"""
Taichi/Quadrants (closure) block templates behind make_hillshade_group /
make_hillshade_kernel, on the new builder/frozen/bound stack
(core/context/builder.py, frozen.py, bound.py).

`_gradient_x`/`_gradient_y` each independently compose the caller's grid
FrozenGroup (under name `GRID`) and call `ctx.GRID.neighbour(i, k)` /
`ctx.GRID.DX.get(0)` - see __init__.py's own module docstring for why this
is the first nested-FrozenGroup-in-FrozenGroup case in this rewrite.
`k_left`/`k_right`/`k_top`/`k_bottom` (which axis of D4/D8's own delta table
means which direction - picked once in __init__.py from `topology`) are
per-call integers, not template-global constants, so `_make_gradient_x_tmpl`/
`_make_gradient_y_tmpl` build and return a nested def that closes over them
as ordinary python closure variables - compile_closure.py's own rebuild
(`_compile_dropping_ctx`) carries a template's closure cells forward into
its rebuilt globals alongside `__globals__`, so this needs no special
handling here.

The gradient here is deliberately not the legacy fixed-stencil,
clamped-index version (pyfastflow/visu/hillshading.py's gradient_x_flat /
gradient_y_flat): those are wrong under a periodic grid and wrong at a
nodata cell, since clamping an index at a nodata boundary silently reads the
wrong neighbour instead of recognising there is none. Going through
`grid.neighbour(i, k)` instead gets both cases right for free: a -1 return
means "no neighbour" (off a bounded edge, off a nodata source, or into a
nodata target), and a periodic axis wraps automatically. A missing sample is
substituted with z[i] itself, i.e. that side of the central difference
degenerates to a one-sided difference against the cell's own elevation
rather than an out-of-range read.

`ctx.bk` (core/context/bk.py) supplies `sqrt`/`atan2`/`cos`/`sin` for `_at`'s
own hillshade formula - `max`/`min` stay plain python builtins, as grid
already established.

Author: B.G (08/2026)
"""

import importlib
import math

from ..core import KernelBuilder, freeze_helper as _helper

_DEG2RAD = math.pi / 180.0
_HALF_PI = math.pi / 2.0
_TWO_PI = 2.0 * math.pi


def _make_gradient_x_tmpl(k_left: int, k_right: int):
    def _gradient_x_tmpl(ctx, z, i):
        zl = ctx.GRID.neighbour(i, k_left)
        zr = ctx.GRID.neighbour(i, k_right)
        left_val = z[i]
        if zl != -1:
            left_val = z[zl]
        right_val = z[i]
        if zr != -1:
            right_val = z[zr]
        return (right_val - left_val) / (2.0 * ctx.GRID.DX.get(0))
    return _gradient_x_tmpl


def _make_gradient_y_tmpl(k_top: int, k_bottom: int):
    def _gradient_y_tmpl(ctx, z, i):
        zt = ctx.GRID.neighbour(i, k_top)
        zb = ctx.GRID.neighbour(i, k_bottom)
        top_val = z[i]
        if zt != -1:
            top_val = z[zt]
        bottom_val = z[i]
        if zb != -1:
            bottom_val = z[zb]
        return (bottom_val - top_val) / (2.0 * ctx.GRID.DX.get(0))
    return _gradient_y_tmpl


def _at_tmpl(ctx, z, i):
    dzdx = ctx.grad_x(z, i) * ctx.ZFACTOR.get(0)
    dzdy = ctx.grad_y(z, i) * ctx.ZFACTOR.get(0)

    slope_rad = ctx.bk.atan2(ctx.bk.sqrt(dzdx * dzdx + dzdy * dzdy), 1.0)
    azimuth_rad = ctx.AZIMUTH.get(0) * _DEG2RAD
    zenith_rad = _HALF_PI - ctx.ALTITUDE.get(0) * _DEG2RAD

    aspect_rad = 0.0
    if dzdx != 0.0 or dzdy != 0.0:
        aspect_rad = _HALF_PI - ctx.bk.atan2(dzdy, dzdx)
        if aspect_rad < 0.0:
            aspect_rad += _TWO_PI

    hillshade_value = ctx.bk.cos(zenith_rad) * ctx.bk.cos(slope_rad) + ctx.bk.sin(zenith_rad) * ctx.bk.sin(
        slope_rad
    ) * ctx.bk.cos(azimuth_rad - aspect_rad)
    return max(0.0, min(1.0, hillshade_value))


def build_group(group, *, grid, k_top, k_left, k_right, k_bottom):
    """
    Compose `at(z, i)` (and its private `grad_x`/`grad_y`) onto `group` (a
    GroupBuilder) for a closure backend (Taichi or Quadrants). `grid`
    (a FrozenGroup) is composed independently under `grad_x` and `grad_y` -
    see the module docstring.

    Returns nothing - `at` is compose()d onto `group` itself, under its own
    public name, by this call.

    Author: B.G (08/2026)
    """
    grad_x = _helper(_make_gradient_x_tmpl(k_left, k_right), helpers={"GRID": grid})
    grad_y = _helper(_make_gradient_y_tmpl(k_top, k_bottom), helpers={"GRID": grid})
    at = _helper(
        _at_tmpl,
        params=["AZIMUTH", "ALTITUDE", "ZFACTOR"],
        helpers={"grad_x": grad_x, "grad_y": grad_y},
    )
    group.compose("at", at)


def _make_hillshade_kernel_tmpl(backend: str):
    bmod = importlib.import_module(backend)
    T = bmod.template()

    def _hillshade_kernel_tmpl(ctx, z: T, out: T):
        n = ctx.hillshade.NX.get(0) * ctx.hillshade.NY.get(0)
        for i in range(n):
            out[i] = ctx.hillshade.at(z, i)
    return _hillshade_kernel_tmpl


def build_kernel(hillshade_group, *, backend: str):
    """
    The standalone `hillshade` FrozenKernel for a closure backend - see
    __init__.py's own `make_hillshade_kernel`. `backend` ("taichi" or
    "quadrants") picks the real `ti.template()`/`qd.template()` marker the
    kernel's own `z`/`out` data arguments are annotated with, closed over by
    `_make_hillshade_kernel_tmpl` exactly like `k_left`/... above.

    Author: B.G (08/2026)
    """
    tmpl = _make_hillshade_kernel_tmpl(backend)
    return KernelBuilder(tmpl).compose("hillshade", hillshade_group).freeze()
