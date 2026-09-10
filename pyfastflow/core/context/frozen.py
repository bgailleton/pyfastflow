"""Immutable computation recipes produced by builders."""

from types import MappingProxyType
from typing import Any, Mapping

from ..pool.base import new_uid
from .contract import Contract
from .slot import BuildError, SlotGroup, SlotKind

Address = tuple[str, ...]


class FrozenError(BuildError):
    """
    Raised on any attempt to mutate a frozen Node, or a builder that has
    already closed its build phase (builder.py's `_check_mutable`) - build a
    new one instead of poking a new value into a done one.

    """


def _norm_shared(shared: "Mapping | None") -> "dict[Address, Address]":
    """
    A `{relative path -> relative canonical path}` map with both sides as
    segment tuples, defensively copied. `None` yields an empty map. See the
    module docstring's `shared` field.

    """
    out: dict[Address, Address] = {}
    for path, canonical in (shared or {}).items():
        out[tuple(path)] = tuple(canonical)
    return out


class Node:
    """
    The one immutable graph node. Not instantiated directly - each `kind` is
    one of the six subclasses. See the module docstring for every field and
    for `build()`.

    """

    KIND: str = None  # set by each subclass
    bound_cls: type = None  # set by each subclass (frozen leaves: assigned in bound.py)

    def __init__(
        self,
        template: Any,
        slots: SlotGroup,
        children: "Mapping[str, Node]",
        contract: Contract,
        shared: "Mapping | None" = None,
        synthetic: "Mapping | None" = None,
        order: tuple = (),
    ):
        object.__setattr__(self, "_uid", new_uid())
        object.__setattr__(self, "kind", self.KIND)
        object.__setattr__(self, "template", template)
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "children", MappingProxyType(dict(children)))
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "shared", MappingProxyType(_norm_shared(shared)))
        object.__setattr__(self, "synthetic", MappingProxyType(dict(synthetic or {})))
        object.__setattr__(self, "order", tuple(order))

    @property
    def uid(self) -> int:
        """
        Process-wide identity assigned at construction, from the same counter
        as Parameter/DataHandle. Two references to one node share a uid;
        composing "the same" node into two builders never changes it.

        """
        return self._uid

    @property
    def provides(self) -> set[str]:
        """
        This node's own top-level PARAM/HELPER slot names plus its own child
        names - what a compose() one level further out checks a chain's next
        segment against. DATA excluded.

        """
        return self.slots.names(SlotKind.PARAM) | self.slots.names(SlotKind.HELPER) | set(self.children)

    def build(self) -> Any:
        """
        Enter the bind phase: walk this node's whole tree and return a fresh
        Bound* (`self.bound_cls`) with one independently-bindable slot per
        full dotted path. See the module docstring and bound.py.

        `bound` is imported locally so its module-load assignment of every
        frozen leaf's `bound_cls` has run before `self.bound_cls` is read
        (bound.py imports this module).

        """
        from . import bound

        return self.bound_cls(self, *bound.walk(self))

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenError(
            f"{type(self).__name__}(uid={self._uid}) is frozen and cannot be mutated - "
            f"build a new {type(self).__name__} instead"
        )

    def __delattr__(self, name: str) -> None:
        raise FrozenError(
            f"{type(self).__name__}(uid={self._uid}) is frozen and cannot be mutated - "
            f"build a new {type(self).__name__} instead"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(uid={self._uid}, provides={sorted(self.provides)})"


class FrozenKernel(Node):
    """
    A frozen kernel - a KernelBuilder's freeze() result: a device entry point
    with a template, PARAM/DATA slots and composed children. See the module
    docstring.

    ``domain`` and ``block`` describe the launch configuration for a CuPy
    kernel. Closure backends use the iteration space in their template.

    """

    KIND = "kernel"

    def __init__(
        self,
        template,
        slots,
        children,
        contract,
        shared=None,
        synthetic=None,
        domain=None,
        block=None,
    ):
        super().__init__(template, slots, children, contract, shared=shared, synthetic=synthetic)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "block", block)


class FrozenHelper(Node):
    """
    A frozen device helper - a HelperBuilder's freeze() result: PARAM/HELPER
    slots and composed children, no DATA slots of its own (a helper's data
    reaches it as a trusted call argument of its caller). See the module
    docstring.

    """

    KIND = "helper"


class FrozenGroup(Node):
    """
    A frozen group - a GroupBuilder's freeze() result: a non-callable,
    navigable composite of PARAM/HELPER slots and composed children, `template`
    always None. `ctx.grid.NX.get(0)` (a PARAM leaf reached through it) and
    `ctx.grid.neighbour(i, k)` (a composed HELPER child called through it) both
    resolve by ordinary chain recursion through `.slots`/`.children`, exactly
    as through a helper one level in - a group differs only in having no
    template to compile, so calling it bare (`ctx.grid(...)`) is illegal
    (compile_closure.py attaches its ctx node uncompiled; compile_cupy.py's
    chain resolver raises).

    `.contract` is always empty - a group has no body of its own to derive one
    from - which is exactly what compile_shared.check_legal_accessors' walk
    expects: it recurses into a group's composed children (where real
    contracts live) and finds no PARAM chain of the group's own to check.

    """

    KIND = "group"
