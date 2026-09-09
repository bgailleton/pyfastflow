"""
Node: the one immutable graph node every frozen build-phase result is.

A builder's `.freeze()` (builder.py) hands back a Node - the inert,
value-like recipe the bind phase (bound.py) builds against. There is one node
protocol and six kinds of it (`kind`): `"kernel"`, `"helper"`, `"group"`,
`"hostblock"` (the leaves, frozen.py/host_block.py) and `"routine"`,
`"sequence"` (the ordered composites, routine.py/sequence.py). All six are the
same class family, so `isinstance(x, Node)` holds for every one of them and a
single walker (`bound.walk`) mints the address table for all of them.

Fields, all fixed once at construction (`__setattr__` raises FrozenError
afterwards):

  kind      one of the six strings above (from each subclass's `KIND`).
  template  the ingested template - a python def, CUDA source text, or None
            (group/routine/sequence carry no template of their own).
  slots     a SlotGroup snapshot (slot.py): this node's own PARAM/DATA/HELPER
            leaves, frozen at the size they had when the builder closed.
  children  {name: Node}, insertion-ordered - the named sub-nodes composed
            under this one. For a routine, insertion order IS launch order;
            for a sequence, children is the block registry and `order` is the
            schedule. Stored behind a read-only view.
  contract  the Contract (contract.py) derived from `template`.
  shared    {relative path -> relative canonical path}, each a segment tuple
            relative to THIS node - this node's own build-phase sharing
            (_Builder.share()/share_identical(), builder.py). A path may name a
            PARAM leaf, a DATA leaf, or a child root (the whole subtree then
            redirects). The canonical may be an existing address of this node's
            tree or, when share(as_=...) re-roots, a synthetic top-level name
            declared in `synthetic`. See bound.py's walk. Read-only view.
  synthetic {name -> Node | Slot} - the synthetic top-level roots/leaves that
            share(as_=...) mints: a Node backs a shared child root, a Slot a
            shared leaf. The walk mints each at `name` and `shared` redirects
            the original paths to it. Empty unless as_ was used. Read-only view.
  order     the schedule tuple. `()` for a leaf; a routine's step names in
            launch order; a sequence's ordered step/loop entries.

`.provides` is what a compose() one level further out checks a chain's next
segment against: this node's own top-level PARAM/HELPER slot names plus its
own child names. DATA is excluded - a DATA slot is never reached through
`ctx.*` (slot.py), so it is not part of what `outer.this.member` can ask this
node to provide.

`.build()` enters the bind phase: `self.bound_cls(self, *walk(self))`. The
walk mints one independently-bindable slot per full dotted path; `bound_cls`
(a class attribute per kind) says which Bound* class receives it. No
isinstance dispatch anywhere - one node protocol, one build path.

A node is shared by identity, never copied: compose the same Node into two
builders and both hold that one object (checked by `is`/`uid` wherever
sameness matters). One grid helper, built once, backs eighty kernels with no
eighty copies of its recipe. `.build()` never mutates the node and allocates a
fresh bind-time table on every call, so those eighty kernels each get their
own independently-bindable address tree from one frozen recipe.

Author: B.G (09/2026)
"""

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

    Author: B.G (09/2026)
    """


def _norm_shared(shared: "Mapping | None") -> "dict[Address, Address]":
    """
    A `{relative path -> relative canonical path}` map with both sides as
    segment tuples, defensively copied. `None` yields an empty map. See the
    module docstring's `shared` field.

    Author: B.G (09/2026)
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

    Author: B.G (09/2026)
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

        Author: B.G (09/2026)
        """
        return self._uid

    @property
    def provides(self) -> set[str]:
        """
        This node's own top-level PARAM/HELPER slot names plus its own child
        names - what a compose() one level further out checks a chain's next
        segment against. DATA excluded. See the module docstring.

        Author: B.G (09/2026)
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

        Author: B.G (09/2026)
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

    `domain`/`block` are its launch config (Unit 4): `domain` names one of the
    kernel's DATA arguments (the launch extent is that buffer's length at launch
    time) or is an int (a fixed extent) or None (closure backends range over the
    template's own loop; cupy then falls back to the compile-time grid/block
    compat path). `block` is the cupy threads-per-block.

    Author: B.G (09/2026)
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
    A frozen device helper - a HelperBuilder's ingest() result: PARAM/HELPER
    slots and composed children, no DATA slots of its own (a helper's data
    reaches it as a trusted call argument of its caller). See the module
    docstring.

    Author: B.G (09/2026)
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

    Author: B.G (09/2026)
    """

    KIND = "group"
