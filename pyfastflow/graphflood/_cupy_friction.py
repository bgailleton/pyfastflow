"""CUDA GraphFlood friction templates."""

from ..core import FrozenHelper, HelperBuilder, new_uid

_MIN_SLOPE = 1.0e-5
_MIN_MANNING = 1.0e-9


def _qo_manning(grid, t: str) -> FrozenHelper:
    return (
        HelperBuilder(
            f"""
__device__ float {t}_qo_manning(float h, float slope) {{
    float hh = h > 0.0f ? h : 0.0f;
    float ss = slope > {_MIN_SLOPE}f ? slope : {_MIN_SLOPE}f;
    float coeff = $ctx.MANNING.get(0)$;
    coeff = coeff > {_MIN_MANNING}f ? coeff : {_MIN_MANNING}f;
    float u = powf(hh, $ctx.EXPO.get(0)$) / coeff * sqrtf(ss);
    return hh * u * $ctx.grid.DX.get(0)$;
}}
"""
        ).compose("grid", grid).freeze()
    )


_LAWS = {"manning": _qo_manning}


def _velocity_manning(t: str) -> FrozenHelper:
    return HelperBuilder(
        f"""
__device__ float {t}_velocity_manning(float h, float slope) {{
    float hh = h > 0.0f ? h : 0.0f;
    float ss = slope > {_MIN_SLOPE}f ? slope : {_MIN_SLOPE}f;
    float coeff = $ctx.MANNING.get(0)$;
    coeff = coeff > {_MIN_MANNING}f ? coeff : {_MIN_MANNING}f;
    return powf(hh, $ctx.EXPO.get(0)$) / coeff * sqrtf(ss);
}}
"""
    ).freeze()


_VELOCITY_LAWS = {"manning": _velocity_manning}


def build_friction_velocity(law: str) -> FrozenHelper:
    """Return a hydraulic velocity helper, leaving flow width to its caller."""
    if law not in _VELOCITY_LAWS:
        raise ValueError(
            f"build_friction_velocity: law must be one of "
            f"{sorted(_VELOCITY_LAWS)}, got {law!r}"
        )
    return _VELOCITY_LAWS[law](f"gfv{new_uid()}")


def build_friction_qo(law: str, grid) -> FrozenHelper:
    """
    `qo(h, slope)` FrozenHelper for the cupy backend - see
    _closure_friction.py's build_friction_qo (identical contract).

    Parameters
    ----------
    law : str
        "manning" (only value implemented).
    grid : FrozenGroup

    Returns
    -------
    FrozenHelper

    Raises
    ------
    ValueError
        If `law` is not a recognised friction law.

    """
    if law not in _LAWS:
        raise ValueError(f"build_friction_qo: law must be one of {sorted(_LAWS)}, got {law!r}")
    t = f"gf{new_uid()}"
    return _LAWS[law](grid, t)
