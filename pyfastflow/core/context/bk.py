"""
`ctx.bk`: the reserved backend-intrinsics namespace for Taichi/Quadrants
templates that need real transcendental functions or a typed literal cast.

Why this exists
----------------
Most closure templates get by on plain arithmetic and the handful of python
builtins Taichi/Quadrants trace natively (`abs`, `min`, `max`, `int`,
`float`). A few need more: transcendental math (`sqrt`, `atan2`, `cos`,
`sin`, `floor`) and typed dtype casts, neither of which has a python
built-in equivalent that traces correctly - bare `math.sqrt`/`math.floor`
inside a `ti.func` raises `TaichiTypeError: must be real number, not Taichi
Expression`, since Taichi's AST transformer only special-cases its own short
builtin list, never the `math` module.

`ctx.bk` exposes this surface as a reserved, always-present member of `ctx`
on the closure backends only: `ctx.bk.sqrt(x)`, `ctx.bk.atan2(y, x)`,
`ctx.bk.u32(0x846CA68B)`. One template text works against both Taichi and
Quadrants because `make_closure_bk(backend)` resolves the same attribute
surface against whichever module (`ti` or `qd`) `backend` is.

A free global spliced into a template's namespace (bypassing `ctx`
entirely) was considered and rejected: every reference a template makes
should be visible in its derived Contract, not smuggled in through
`__globals__` - see ctx.py/contract.py for the grammar this keeps intact.

Reserved, not a slot
---------------------
`ctx.bk.*` is a builtin the grammar recognises structurally, not a
capability a template's Contract requires satisfied: contract.py's AST walk
drops any chain rooted at `RESERVED_BK_NAME` before it reaches
`Contract.chains`, so `ctx.bk.sqrt(...)` never shows up in `inspect()` or
`unmet()`, and `ingest()` never demands a wired "bk" slot for it.
Symmetrically, `_Builder._wire()`/`compose()` (builder.py) raise if a caller
tries to declare a slot or compose a sub-structure named "bk" - the name is
reserved and can never mean anything else. `bk` is exposed on every level of
the ctx tree compile_closure.py builds, not just the root, since a helper
several levels deep may need it just as much as its caller.

cupy is unaffected: its templates are raw CUDA text, where the native C
spelling (`sqrtf`, `atan2f`, `floorf`, a plain `0x846CA68Bu` literal) is
already the natural way to write this - `ctx.bk` is never resolved against
a cupy compile.

Surface
-------
`sqrt`, `atan2`, `cos`, `sin`, `floor` - transcendental math.
`u32`, `i32`, `i64`, `f32` - dtype tokens, exposed as the backend's own
dtype objects (see "Typed literal casts" below).
`bit_cast`, `select`, `cast` - the IEEE-754 bit-flip trick (reinterpreting a
float's bits as u32 and back) plus a branchless ternary-style select.
`atomic_min`, `atomic_max`, `atomic_add` - the three atomic ops Taichi/
Quadrants expose, for genuinely concurrent writes into a DATA buffer.
`grouped`, `Vector` - `ti.grouped`/`ti.Vector` (and `qd.` equivalents)
passed through unchanged, for dimensionality-agnostic iteration and small
per-thread fixed-size local arrays.

`abs`/`min`/`max`/`int`/`float` stay plain python builtins rather than
`ctx.bk` members - both backends special-case them for casting the same way
they special-case `abs`/`min`, and they work uniformly on a bare literal or
an already-traced expression, which the dtype objects below do not.

Typed literal casts
--------------------
`u32`/`i32`/`i64`/`f32` are exposed as the backend's own dtype objects
themselves (`ti.u32`, never a `lambda x: ti.cast(x, ti.u32)` wrapper) -
`ctx.bk.u32(0x846CA68B)` is called exactly as `ti.u32(0x846CA68B)` would be.
This is load-bearing: Taichi's frontend recognises a call whose callee
resolves, by identity, to one of its own dtype objects, and exempts that
call's literal argument from default-int-type inference even reached
through an attribute chain - `ti.u32(2221713035)` compiles, but wrapping the
same cast in an ordinary python callable breaks it (`Integer literal
2221713035 exceeded the range of default_ip: i32`), because the literal is
type-inferred as a bare argument before the wrapper body ever runs. So
`ctx.bk.u32(...)` must resolve to the real `ti.u32` object one attribute hop
away, not to a function that happens to produce the same value. Only `u32`/
`i64` currently need the oversized-literal exemption in practice; `i32`/
`f32` are exposed the same way for symmetry, since `ctx.bk.cast`'s second
argument is always one of these four dtype tokens regardless.

Author: B.G (08/2026)
"""

from typing import Any

from .errors import PyFastFlowError

RESERVED_BK_NAME = "bk"
"""The reserved ctx member name for the backend-intrinsics namespace - see
the module docstring. Never wirable as a slot, never composable as a root."""


class BkError(PyFastFlowError):
    """
    Raised by an unknown `ctx.bk.*` attribute - naming it and listing what is
    actually available, rather than letting a typo fall through to a bare
    AttributeError deep inside backend trace machinery.

    Author: B.G (08/2026)
    """


_BK_METHOD_NAMES = (
    "sqrt", "atan2", "cos", "sin", "floor", "u32",
    "bit_cast", "select", "cast", "atomic_min", "atomic_max", "atomic_add", "i32", "i64", "f32",
    "grouped", "Vector",
)
"""Every name `ctx.bk` resolves - see the module docstring's "Surface" section."""


class ClosureBkNode:
    """
    `ctx.bk` itself, for one closure backend module (`ti` or `qd`) - see the
    module docstring. Every entry in `_BK_METHOD_NAMES` is resolved once, at
    construction, straight to the backend's own object (`ti.sqrt`, `ti.u32`,
    ...) - never wrapped, so a closure backend's own trace-time recognition
    of e.g. `ti.sqrt` as a builtin op, or `ti.u32` as a dtype-cast callee that
    exempts its own literal argument from default-int-type inference (see
    the module docstring), is identity-based and applies exactly as it would
    to a template calling `ti.sqrt`/`ti.u32` directly.

    Author: B.G (08/2026)
    """

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
    """
    `ctx.bk` for one closure compile - `backend` is the `taichi` or
    `quadrants` module (`BoundKernel.compile()`'s own `backend` argument,
    compile_closure.py). Stateless and cheap; built once per compile() and
    shared by every node in that compile's own ctx tree.

    Author: B.G (08/2026)
    """
    return ClosureBkNode(backend)
