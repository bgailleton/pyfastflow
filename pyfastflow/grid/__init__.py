"""Grid topology and boundary helpers.

``make_grid_group`` returns reusable grid structure;
``make_grid_parameters`` returns the matching dimensions, spacing, and masks.
Compose the group into a feature, then bind those parameters to it.
"""

import numpy as np

from ..core import Backend, FrozenGroup, GroupBuilder, require_backend, share_leaf

_TOPOLOGIES = {"D4": 4, "D8": 8}
_BOUNDARIES = frozenset({"normal", "periodic_EW", "periodic_NS"})
_OUTLETS = frozenset({"edge", "mask"})


def _blocks_for(be: Backend):
    """Return the implementation module for one backend family."""
    if be.family == "closure":
        from . import _closure_blocks as blocks
    elif be.family == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"make_grid_group: unsupported backend family {be.family!r}")
    return blocks


def _check_config(topology: str, boundary: str, outlet: str) -> None:
    if topology not in _TOPOLOGIES:
        raise ValueError(f"make_grid_group: topology must be one of {sorted(_TOPOLOGIES)}, got {topology!r}")
    if boundary not in _BOUNDARIES:
        raise ValueError(f"make_grid_group: boundary must be one of {sorted(_BOUNDARIES)}, got {boundary!r}")
    if outlet not in _OUTLETS:
        raise ValueError(f"make_grid_group: outlet must be one of {sorted(_OUTLETS)}, got {outlet!r}")


def make_grid_group(
    be: Backend,
    *,
    topology: str = "D8",
    boundary: str = "normal",
    nodata: bool = False,
    outlet: str = "edge",
) -> FrozenGroup:
    """Return reusable grid topology and boundary helpers.

    Compose this group into a feature, then bind the matching parameters from
    :func:`make_grid_parameters` before compilation.

    Parameters
    ----------
    be : Backend
        Target backend.
    topology : {"D4", "D8"}, default="D8"
        Four- or eight-neighbour grid connectivity.
    boundary : {"normal", "periodic_EW", "periodic_NS"}, default="normal"
        Domain-edge behaviour.
    nodata : bool, default=False
        Include a per-cell no-data mask.
    outlet : {"edge", "mask"}, default="edge"
        Permit drainage through grid edges or a supplied outlet mask.

    Returns
    -------
    FrozenGroup
        Helpers with ``NX``, ``NY``, ``DX``, and ``N_NEIGHBOURS`` parameter
        slots, plus mask slots requested by the options.
    """
    be = require_backend(be)
    _check_config(topology, boundary, outlet)
    blocks = _blocks_for(be)

    group = GroupBuilder()
    group.param("NX")
    group.param("NY")
    group.param("DX")
    group.param("N_NEIGHBOURS")
    if nodata:
        group.param("NODATA_MASK")
    if outlet == "mask":
        group.param("OUTLET_MASK")

    blocks.build_group(group, topology=topology, boundary=boundary, nodata=nodata, outlet=outlet)

    share_leaf(group, "NX")
    share_leaf(group, "NY")
    share_leaf(group, "DX")
    if nodata:
        share_leaf(group, "NODATA_MASK")
    if outlet == "mask":
        share_leaf(group, "OUTLET_MASK")

    return group.freeze()


def make_grid_parameters(
    be: Backend,
    pool,
    nx: int,
    ny: int,
    dx: float,
    *,
    topology: str = "D8",
    nodata: bool = False,
    outlet: str = "edge",
    nx_mode: str = "const",
    ny_mode: str = "const",
    dx_mode: str = "const",
) -> dict:
    """Return the parameters required by a grid helper group.

    Use the same ``topology``, ``nodata``, and ``outlet`` options as the grid
    group. ``NX``, ``NY``, and ``DX`` default to constants; scalar storage
    lets a model update dimensions or spacing at runtime, while ``DX`` may
    also be a spatial field.

    Parameters
    ----------
    be : Backend
        Target backend.
    pool : Pool
        Pool that owns the allocated parameters.
    nx, ny : int
        Raster dimensions.
    dx : float
        Cell spacing, or the initial value of a spatial spacing field.
    topology : {"D4", "D8"}, default="D8"
        Connectivity; must match :func:`make_grid_group`.
    nodata : bool, default=False
        Create an all-valid ``NODATA_MASK`` field.
    outlet : {"edge", "mask"}, default="edge"
        Create an all-closed ``OUTLET_MASK`` field when set to ``"mask"``.
    nx_mode, ny_mode : {"const", "scalar"}, default="const"
        Storage mode for grid dimensions.
    dx_mode : {"const", "scalar", "field"}, default="const"
        Storage mode for cell spacing.

    Returns
    -------
    dict[str, Parameter]
        Parameters keyed by the slots exposed by the matching grid group.
    """
    be = require_backend(be)
    _check_config(topology, "normal", outlet)
    if nx_mode not in ("const", "scalar"):
        raise ValueError(f"make_grid_parameters: nx_mode must be 'const' or 'scalar', got {nx_mode!r}")
    if ny_mode not in ("const", "scalar"):
        raise ValueError(f"make_grid_parameters: ny_mode must be 'const' or 'scalar', got {ny_mode!r}")
    if dx_mode not in ("const", "scalar", "field"):
        raise ValueError(f"make_grid_parameters: dx_mode must be 'const', 'scalar' or 'field', got {dx_mode!r}")

    ParamCls = be.ParameterCls
    n_flat = int(nx) * int(ny)

    nx_p = ParamCls("GRID_NX", dtype="i32", mode=nx_mode, value=int(nx), pool=pool)
    ny_p = ParamCls("GRID_NY", dtype="i32", mode=ny_mode, value=int(ny), pool=pool)

    if dx_mode == "field":
        dx_p = ParamCls(
            "GRID_DX",
            dtype="f32",
            mode="field",
            value=np.full(n_flat, dx, dtype=np.float32),
            pool=pool,
            shape=(n_flat,),
        )
    else:
        dx_p = ParamCls("GRID_DX", dtype="f32", mode=dx_mode, value=float(dx), pool=pool)

    n_neighbours_p = ParamCls(
        "GRID_NNEIGHBOURS", dtype="i32", mode="const", value=_TOPOLOGIES[topology], pool=pool
    )

    params = {"NX": nx_p, "NY": ny_p, "DX": dx_p, "N_NEIGHBOURS": n_neighbours_p}

    if nodata:
        params["NODATA_MASK"] = ParamCls(
            "GRID_NODATA_MASK",
            dtype="u8",
            mode="field",
            value=np.zeros(n_flat, dtype=np.uint8),
            pool=pool,
            shape=(n_flat,),
        )

    if outlet == "mask":
        params["OUTLET_MASK"] = ParamCls(
            "GRID_OUTLET_MASK",
            dtype="u8",
            mode="field",
            value=np.zeros(n_flat, dtype=np.uint8),
            pool=pool,
            shape=(n_flat,),
        )

    return params
