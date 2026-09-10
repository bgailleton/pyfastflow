"""Hillshade helpers and a standalone hillshade kernel.

Use ``make_hillshade_group`` to compose ``at(z, i)`` into another kernel, or
``make_hillshade_kernel`` for a complete raster pass.
"""

from ..core import (
    Backend, FrozenGroup, FrozenKernel, GroupBuilder, SlotKind, require_backend,
    share_leaf,
)

_TOPOLOGIES = {"D4": 4, "D8": 8}
_MODES = ("const", "scalar")


def _blocks_for(be: Backend):
    """Return the implementation module for one backend family."""
    if be.family == "closure":
        from . import _closure_blocks as blocks
    elif be.family == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"make_hillshade_group: unsupported backend family {be.family!r}")
    return blocks


def _k_indices(topology: str):
    if topology == "D4":
        return {"k_top": 0, "k_left": 1, "k_right": 2, "k_bottom": 3}
    if topology == "D8":
        return {"k_top": 1, "k_left": 3, "k_right": 4, "k_bottom": 6}
    raise ValueError(f"make_hillshade_group: topology must be one of {sorted(_TOPOLOGIES)}, got {topology!r}")


def make_hillshade_group(be: Backend, grid: FrozenGroup, *, topology: str = "D8") -> FrozenGroup:
    """Return composable hillshade helpers for a grid.

``topology`` must match the supplied grid structure.
"""
    if topology not in _TOPOLOGIES:
        raise ValueError(f"make_hillshade_group: topology must be one of {sorted(_TOPOLOGIES)}, got {topology!r}")
    be = require_backend(be)
    blocks = _blocks_for(be)
    k = _k_indices(topology)

    group = GroupBuilder()
    group.param("AZIMUTH")
    group.param("ALTITUDE")
    group.param("ZFACTOR")
    grid_param_names = grid.slots.names(SlotKind.PARAM)
    for name in grid_param_names:
        group.param(name)

    blocks.build_group(group, grid=grid, **k)

    share_leaf(group, "AZIMUTH")
    share_leaf(group, "ALTITUDE")
    share_leaf(group, "ZFACTOR")
    for name in grid_param_names:
        share_leaf(group, name)

    return group.freeze()


def make_hillshade_parameters(
    be: Backend,
    pool,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    azimuth_mode: str = "const",
    altitude_mode: str = "const",
    z_factor_mode: str = "const",
) -> dict:
    """
    Build the concrete, caller-owned Parameter objects one hillshade group's
    own value PARAM slots need bound: {"AZIMUTH": ..., "ALTITUDE": ...,
    "ZFACTOR": ...}. NX/NY/DX/N_NEIGHBOURS (and NODATA_MASK/OUTLET_MASK, if
    present) are not among these: bind the same Parameter objects from
    ``make_grid_parameters`` into
    `hillshade.NX`/... as well as `grid.NX`/....

    `azimuth`/`altitude` are light-source angles in degrees (315/45 is the
    classic NW-lit default); `z_factor` scales the gradient before it enters
    the slope/aspect computation.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    pool : Pool
        Device-buffer pool backing scalar-mode Parameters.
    azimuth, altitude : float, optional
        Light-source angles in degrees.
    z_factor : float, optional
        Gradient scale factor.
    azimuth_mode, altitude_mode, z_factor_mode : str, optional
        "const" (default) or "scalar" - same convention as
        make_grid_parameters/make_noise_parameters.

    Returns
    -------
    dict
        {"AZIMUTH": ..., "ALTITUDE": ..., "ZFACTOR": ...}.

    Raises
    ------
    ValueError
        If any `*_mode` is not "const" or "scalar".

    """
    for label, mode in (
        ("azimuth_mode", azimuth_mode),
        ("altitude_mode", altitude_mode),
        ("z_factor_mode", z_factor_mode),
    ):
        if mode not in _MODES:
            raise ValueError(f"make_hillshade_parameters: {label} must be 'const' or 'scalar', got {mode!r}")

    ParamCls = require_backend(be).ParameterCls

    azimuth_p = ParamCls("HS_AZIMUTH", dtype="f32", mode=azimuth_mode, value=float(azimuth), pool=pool)
    altitude_p = ParamCls("HS_ALTITUDE", dtype="f32", mode=altitude_mode, value=float(altitude), pool=pool)
    z_factor_p = ParamCls("HS_ZFACTOR", dtype="f32", mode=z_factor_mode, value=float(z_factor), pool=pool)

    return {"AZIMUTH": azimuth_p, "ALTITUDE": altitude_p, "ZFACTOR": z_factor_p}


def make_hillshade_kernel(be: Backend, hillshade_group: FrozenGroup) -> FrozenKernel:
    """
    The standalone `hillshade` pass: a FrozenKernel composing the *whole*
    `hillshade_group` under the name `hillshade` and writing `out[i] =
    hillshade.at(z, i)` for every node. ``z`` and ``out`` are DATA slots
    (cupy additionally takes ``n``,
    the node count, as a DATA slot too - a `cp.RawModule` kernel has no
    auto-ranging equivalent to Taichi/Quadrants' `for i in range(n)`).

    A caller `.build()`s the result, binds `z`/`out` (and, on cupy, `n`) plus
    every PARAM address `hillshade_group` itself carries (`hillshade.NX`,
    `hillshade.AZIMUTH`, ...), then `.compile()`s.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    hillshade_group : FrozenGroup
        The group built by make_hillshade_group(), composed whole under
        `hillshade`.

    Returns
    -------
    FrozenKernel

    """
    be = require_backend(be)
    blocks = _blocks_for(be)
    if be.family == "cupy":
        return blocks.build_kernel(hillshade_group)
    return blocks.build_kernel(hillshade_group, backend=be.name)
