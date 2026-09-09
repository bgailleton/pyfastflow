"""
Taichi/Quadrants (closure) friction-law block behind make_graphflood's
compute_qo step, on the builder/frozen/bound stack (../core/context/
builder.py, frozen.py, bound.py).

`build_friction_qo(law, grid)` returns one FrozenHelper computing `qo(h,
slope)`, composed onto compute_qo's own KernelBuilder under the name
"friction" so a caller reaches it directly as `ctx.friction(h, slope)` (a
composed HelperBuilder is itself the callable - "friction" is only the
compose() name, not a namespace with a "qo" member). `law`
picks which private template gets ingested - only "manning" exists so far,
selected the same way grid/flow pick a block variant at build time
(`_blocks_for`/`mode`) - `qo`'s call-site shape (h, slope) -> Q stays fixed
regardless of which law backs it, so a future law is a second template
picked by this same dispatch, not a change to any caller.

Author: B.G (08/2026)
"""

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

    Author: B.G (08/2026)
    """
    if law not in _LAWS:
        raise ValueError(f"build_friction_qo: law must be one of {sorted(_LAWS)}, got {law!r}")
    return HelperBuilder(_LAWS[law]).compose("grid", grid).freeze()
