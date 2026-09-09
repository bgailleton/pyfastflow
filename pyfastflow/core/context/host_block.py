"""
HostBlockBuilder / FrozenHostBlock / BoundHostBlock: a leaf builder, in the
same build -> freeze -> bind -> compile family as KernelBuilder/
HelperBuilder (builder.py/frozen.py/bound.py), for host-side python code that
needs to read/write Parameters between device launches - the layer
Sequence's (sequence.py) loop control (`loop`'s `max_times`/`until`) and
inter-block bookkeeping (a depression solver zeroing a counter Parameter
before each pass) run on.

A host block is a leaf, not a composite: it declares PARAM slots (from its
`ctx.X.value`/`.handle()`/`.set()/.read()` contract) and DATA slots (its signature after
`ctx`, like a kernel). A DATA argument resolves at compile to the bound
DataHandle, so the block's own source reads `h.to_numpy()` and the
device->host copy is visible there. compose() raises - a device
helper cannot run on the host, and a host block composes no sub-structure.

`ctx` resolves unwrapped
--------------------------
Where a kernel/helper's `ctx.z` resolves to a Parameter's `device_view()`
(compile_closure.py/compile_cupy.py), a host block's `ctx.z` resolves to the
bound Parameter itself - parameter.py's host-facing surface, `.value`/
`.handle()`/`.set(value)`/`.read()`, not `.get(node)`/`.set_node(node, value)`. A device
view genuinely cannot run outside kernel-trace context, so there is nothing
else `ctx.z` could mean here. `check_legal_host_accessors` enforces the
matching legal-chain set - `(name, "value"|"handle"|"set"|"read")`, two segments, one of
those three - the same role compile_shared.py's `check_legal_accessors` plays
for device code, just against a different legal set.

One class for every backend
-----------------------------
"Compiling" a host block is resolving names, not emitting device code: build
the ctx tree of raw Parameters and the DATA-argument list, and return
`lambda: template(ctx, *data)`. There is no Taichi/Quadrants/cupy variant of
that, so `BoundHostBlock.compile()` takes no backend argument (it accepts and
ignores `backend`, only to keep a call site that already carries a `backend`
variable from having to special-case this block kind).

Author: B.G (08/2026)
"""

import inspect
from typing import Any

from .bound import _Bound
from .builder import _Builder
from .compile_shared import CompileError, check_unmet
from .ctx import CTX_PARAM_NAME
from .frozen import Node
from .slot import BuildError, SlotKind

_LEGAL_HOST_ACCESSORS = ("value", "handle", "set", "read")


class HostBlockBuilder(_Builder):
    """
    Builds a host block: PARAM slots (from the template's own
    `ctx.X.value`/`.handle()`/`.set()/.read()` contract, the host-facing accessor set
    `_LEGAL_ACCESSORS`, plus any param() adds) and DATA slots (its signature
    after `ctx`, exactly like a kernel - review point 5). A DATA argument
    resolves at compile to the bound DataHandle, so the author writes
    `h.to_numpy()` and the device->host sync is visible in the block's source.
    compose() raises - a host block composes no sub-structure.

    Author: B.G (09/2026)
    """

    _LEGAL_ACCESSORS = ("get", "set", "read")
    _HAS_DATA = True

    def __init__(self, template: Any = None):
        super().__init__(template)

    def compose(self, name: str, frozen: Node) -> "HostBlockBuilder":
        """Always raises: a host block composes no sub-structure."""
        raise BuildError(
            "HostBlockBuilder.compose() is not allowed: a host block composes no sub-structure "
            "- it reads Parameters through ctx and takes DATA as signature arguments."
        )

    def freeze(self) -> "FrozenHostBlock":
        """
        Derive this block's PARAM slots (host-accessor contract) and DATA slots
        (signature after ctx) and return the FrozenHostBlock. See
        _Builder._build().

        Author: B.G (09/2026)
        """
        slots, composed, contract = self._build()
        return FrozenHostBlock(self._template, slots, composed, contract)


class FrozenHostBlock(Node):
    """
    The frozen result of a HostBlockBuilder's freeze(): a `Node` of kind
    "hostblock". `children` is always empty (compose() raises during build), so
    `.provides` reports only this block's own wired PARAM names, and the walk
    mints one bindable address per PARAM slot. build() is inherited from Node.

    Author: B.G (09/2026)
    """

    KIND = "hostblock"


def check_legal_host_accessors(bound: "_Bound") -> None:
    """
    Raise on the first PARAM chain in `bound`'s frozen contract that is not
    exactly `(name, "value"|"handle"|"set"|"read")` - the host-facing accessor set
    (parameter.py), as opposed to compile_shared.py's device-facing
    `(name, "get"|"set_node")`. A host block never composes anything, so -
    unlike compile_shared.check_legal_accessors - there is no composition
    tree to walk, just this block's own contract.

    Author: B.G (08/2026)
    """
    frozen = bound.frozen
    param_names = frozen.slots.names(SlotKind.PARAM)
    for chain in frozen.contract.chains:
        root = chain[0]
        if root not in param_names:
            continue
        if len(chain) != 2 or chain[1] not in _LEGAL_HOST_ACCESSORS:
            raise CompileError(
                f"{root!r}: illegal host PARAM accessor 'ctx.{'.'.join(chain)}' - legal "
                f"accessors on a host block are .value, .handle(), .set(...) and .read()"
            )


class _HostCtxNode:
    """
    What `ctx` resolves to inside a compiled host block's template body - a
    plain attribute bag holding this block's Parameters unwrapped. See the
    module docstring's "ctx resolves unwrapped" section.

    Author: B.G (08/2026)
    """


class BoundHostBlock(_Bound):
    """
    The bound result of build()-ing a FrozenHostBlock. See the module
    docstring.

    Author: B.G (08/2026)
    """

    def compile(self, backend: "str | None" = None, **kwargs) -> Any:
        """
        Resolve this block's ctx (each PARAM slot's bound Parameter, unwrapped)
        and its DATA arguments (each bound value - a DataHandle - in signature
        order), and return `lambda: template(ctx, *data)`. Checks unmet slots
        and legal host accessors first. `backend` is accepted and ignored (a
        host block is name resolution, not device emission).

        Author: B.G (09/2026)
        """
        check_unmet(self)
        check_legal_host_accessors(self)
        frozen = self._frozen
        template = frozen.template
        params = list(inspect.signature(template).parameters)
        if not params or params[0] != CTX_PARAM_NAME:
            label = getattr(template, "__name__", "?")
            raise CompileError(f"host block template {label!r}: first parameter must be {CTX_PARAM_NAME!r}, got {params}")
        ctx = _HostCtxNode()
        for name in frozen.slots.names(SlotKind.PARAM):
            setattr(ctx, name, self.value_at((name,)))
        data = [self.value_at((name,)) for name in params[1:]]  # DATA args, signature order
        return lambda: template(ctx, *data)


# Node.build() reads this to mint a BoundHostBlock (frozen.py); set here since
# BoundHostBlock is defined in this module alongside FrozenHostBlock.
FrozenHostBlock.bound_cls = BoundHostBlock
