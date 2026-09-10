"""Reserved device intrinsics for Taichi and Quadrants templates."""

from typing import Any

from .errors import PyFastFlowError

RESERVED_BK_NAME = "bk"
"""Reserved ``ctx`` member for backend intrinsics; never a slot or child."""


class BkError(PyFastFlowError):
    """
    Raised by an unknown `ctx.bk.*` attribute - naming it and listing what is
    actually available, rather than letting a typo fall through to a bare
    AttributeError deep inside backend trace machinery.

    """


_BK_METHOD_NAMES = (
    "sqrt", "atan2", "cos", "sin", "floor", "u32",
    "bit_cast", "select", "cast", "atomic_min", "atomic_max", "atomic_add", "i32", "i64", "f32",
    "grouped", "Vector",
)
"""Names available through ``ctx.bk``."""


class ClosureBkNode:
    """``ctx.bk`` for one closure backend, exposing its native intrinsics."""

    __slots__ = ("_backend", "_fns")

    def __init__(self, backend: Any):
        self._backend = backend
        self._fns = {n: getattr(backend, n) for n in _BK_METHOD_NAMES}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._fns[name]
        except KeyError:
            raise BkError(
                f"ctx.bk.{name} is not a recognised backend intrinsic - available: "
                f"{', '.join(_BK_METHOD_NAMES)}"
            ) from None

    def __repr__(self) -> str:
        return f"ClosureBkNode(backend={self._backend.__name__}, provides={_BK_METHOD_NAMES})"


def make_closure_bk(backend: Any) -> ClosureBkNode:
    """Create the intrinsic namespace for one Taichi or Quadrants compile."""
    return ClosureBkNode(backend)
