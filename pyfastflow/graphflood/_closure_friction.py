"""Python GraphFlood friction templates for Taichi and Quadrants."""

from ..core import FrozenHelper, HelperBuilder

_MIN_SLOPE = 1.0e-5
_MIN_MANNING = 1.0e-9


def _qo_manning_tmpl(ctx, h, slope):
    hh = h if h > 0.0 else 0.0
    ss = slope if slope > _MIN_SLOPE else _MIN_SLOPE
    coeff = ctx.MANNING.get(0)
    coeff = coeff if coeff > _MIN_MANNING else _MIN_MANNING
    u = (hh ** ctx.EXPO.get(0)) / coeff * ctx.bk.sqrt(ss)
    return hh * u * ctx.grid.DX.get(0)


_LAWS = {"manning": _qo_manning_tmpl}


def build_friction_qo(law: str, grid) -> FrozenHelper:
    """
    `qo(h, slope)` FrozenHelper computing volumetric outflow from local
    depth/slope via the friction law named `law`. Wires its own `MANNING`/
    `EXPO` PARAM slots (any mode - a caller binds Parameters there after
    `.build()`) and composes its own `grid` occurrence for `DX`.

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
    return HelperBuilder(_LAWS[law]).compose("grid", grid).freeze()
