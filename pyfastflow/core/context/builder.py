"""Builders for kernels, helpers, and reusable groups."""

from typing import Any

from ..pool.base import new_uid
from .bk import RESERVED_BK_NAME
from .contract import Contract, ContractError, extract_cupy_contract, extract_python_contract
from .frozen import FrozenError, FrozenGroup, FrozenHelper, FrozenKernel, Node
from .slot import BuildError, DataSlot, HelperSlot, ParamSlot, Slot, SlotGroup, SlotGroupError, SlotKind


class _ShareMixin:
    """
    The build-phase sharing surface - `share()`/`share_identical()` with
    identical semantics at every builder level (KernelBuilder/HelperBuilder/
    GroupBuilder via _Builder, plus RoutineBuilder/SequenceBuilder). A user of
    this mixin provides `_check_mutable()`, `_composed`, `_shared`,
    `_synthetic`, `_shared_seen`, and optionally `_slots`. See share() for the
    contract and compute_share (below) for the implementation.

    """

    def share(self, canonical: str, *paths: str, as_: "str | None" = None) -> "_Builder":
        """
        Declare that `canonical` and every `paths` mean the same thing - one
        build-phase-shared quantity, collapsed to a single bindable address.

        Each of `canonical`/`paths` is a dotted relative address of THIS
        builder's own tree (a top-level slot, or a path reaching into an
        already-composed child, e.g. `"neighbour_raw.row.NX"`). Each may name a
        PARAM leaf, a DATA leaf, or a child root - if a root, the whole subtree
        under it is shared. All must be the same kind; shared roots must be the
        identical frozen object (`is`), not merely alike.

        Without `as_`, `canonical` is the surviving, independently-bindable
        address and every `paths` redirects to it. With `as_`, a synthetic new
        top-level name is minted and BOTH `canonical` and every `paths`
        redirect there: after `share("slope.lap.grid", "diffuse.grid",
        as_="grid")` the bindable addresses are `grid.NX`, `grid.DX`, ... and
        both original paths collapse into them.

        This is explicit, by identity, local to one builder's authoring - never
        name-based matching across independently-authored composites.
        `share_identical()` is the one-call form for "collapse every occurrence
        of this bundle".

        Raises
        ------
        BuildError
            A path does not resolve, the paths are not all the same kind,
            shared roots are not the identical object, a path is already
            shared, or `as_` collides with an existing top-level name.

        """
        self._check_mutable()
        new_shared, synthetic, seen_add = compute_share(
            self._share_top_slots(), self._composed, self._top_level_names(),
            self._shared_seen, canonical, paths, as_,
        )
        self._shared.update(new_shared)
        self._synthetic.update(synthetic)
        self._shared_seen |= seen_add
        return self

    def share_identical(self, path: str, as_: "str | None" = None) -> "_Builder":
        """
        Collapse every composed child root anywhere in this builder's tree that
        `is` the identical frozen object as the child root at `path`, sharing
        them all (via `share(path, *others, as_=as_)`). The one-call form of
        "propagate this bundle": one grid object composed under several children
        becomes one bindable `grid.*` set.

        `path` must name a child root, not a leaf. A no-op (not an error) if
        nothing else in the tree is that same object.

        """
        self._check_mutable()
        kind, node, _slot = classify_path(self._share_top_slots(), self._composed, tuple(path.split(".")), what="share_identical")
        if kind != "root":
            raise BuildError(f"share_identical({path!r}): names a {kind} leaf, not a child root - only a root can be shared by identity")
        others = [p for p in find_identical_roots(self._composed, node) if p != path]
        if not others:
            return self
        return self.share(path, *others, as_=as_)

    def _share_top_slots(self) -> "SlotGroup | None":
        """
        The SlotGroup share() resolves a top-level leaf name against: the
        builder's explicitly-declared PARAM slots (param()), which
        is what a share canonical names. Derived PARAM slots are not known until
        freeze(), and a share canonical is never a derived slot (a group has no
        template; a kernel shares its own explicit params), so this suffices at
        share() time. Empty for a routine/sequence (no top-level params).

        """
        sg = SlotGroup()
        for name in getattr(self, "_explicit_params", {}):
            sg.add(ParamSlot(name))
        return sg

    def _top_level_names(self) -> set:
        """Every name already claimed at this level - composed children, synthetic roots, and explicit top-level params."""
        return set(self._composed) | set(self._synthetic) | set(getattr(self, "_explicit_params", {}))



class _Builder(_ShareMixin):
    """
    Shared build-phase machinery behind KernelBuilder/HelperBuilder. Not
    instantiated directly.

    """

    def __init__(self, template: Any = None):
        self._uid = new_uid()
        self._template = template
        # PARAM slots the template does not derive. GroupBuilder has no
        # template, so all of its params are explicit.
        self._explicit_params: set[str] = set()
        # dtype contracts for DATA args the signature already declares:
        # {name -> dtype}. data() populates it; validated at freeze
        # against the signature-derived DATA set.
        self._data_contracts: dict[str, Any] = {}
        self._composed: dict[str, Node] = {}
        # build-phase sharing in the Node form: {source path -> canonical path}
        # (segment tuples), plus {synthetic top-level name -> Node|Slot} for the
        # roots/leaves share(as_=...) re-roots, and the set of already-shared
        # source paths for dup detection. See share()/share_identical().
        self._shared: dict[tuple, tuple] = {}
        self._synthetic: dict[str, Any] = {}
        self._shared_seen: set[tuple] = set()
        self._frozen = False

    # legal PARAM accessor set for this builder's device/host surface - a
    # two-segment chain (name, accessor) is a PARAM read/write; anything else
    # at an uncomposed root is a missing composition. Overridden by
    # HostBlockBuilder (host-facing get/set/read).
    _LEGAL_ACCESSORS = ("get", "set_node")
    # whether this builder's signature after ctx contributes DATA slots
    # (Kernel/HostBlock yes; Helper/Group no - a helper's args are device-call
    # arguments, a group has no signature).
    _HAS_DATA = False

    def _check_mutable(self) -> None:
        if self._frozen:
            raise FrozenError(
                f"{type(self).__name__}(uid={self._uid}) has already closed its build phase "
                f"(freeze()) and is frozen - build a new {type(self).__name__} "
                f"instead of reusing this one"
            )

    def _check_name(self, name: str, what: str) -> None:
        if name == RESERVED_BK_NAME:
            raise SlotGroupError(
                f"'{RESERVED_BK_NAME}' is reserved - ctx.{RESERVED_BK_NAME} is the "
                f"backend-intrinsics namespace (bk.py) and can never be a {what}"
            )

    def param(self, name: str) -> "_Builder":
        """
        Declare an EXTRA PARAM slot the template does not reference - the case
        the template's own contract cannot derive: a canonical leaf a later
        share()/share_leaf collapses into (a GroupBuilder, which has no template,
        declares every one of its params this way), or a param bound but read
        only through a composed child.

        Strict: raises at freeze() if `name` turns out to be a derived slot
        (already implied by the template) or a composed child.

        """
        self._check_mutable()
        self._check_name(name, "PARAM slot")
        self._explicit_params.add(name)
        return self

    def data(self, name: str, *, dtype: Any = None) -> "_Builder":
        """
        Attach a dtype contract to a DATA slot the template's signature already
        declares (a kernel's parameters after `ctx`). Raises at freeze() if
        `name` is not a signature argument. Kernel and HostBlock only - a helper
        has no DATA slots of its own, a group no signature.

        """
        self._check_mutable()
        self._data_contracts[name] = dtype
        return self

    def compose(self, name: str, frozen: Node) -> "_Builder":
        """
        Attach an already-frozen sub-structure (a FrozenHelper or FrozenGroup -
        frozen.py) under `name`, giving a template reaching `ctx.{name}` access
        to whatever `frozen` provides at its own top level. `frozen` is stored
        by identity, not copied. Composing under a name already composed, or
        already declared an explicit PARAM slot, raises. A FrozenKernel raises:
        a kernel is a host entry point, not device-callable.

        """
        self._check_mutable()
        self._check_name(name, "composed root")
        if not isinstance(frozen, Node):
            raise TypeError(f"compose({name!r}, ...): expected a FrozenHelper/FrozenGroup, got {type(frozen).__name__}")
        if isinstance(frozen, FrozenKernel):
            raise SlotGroupError(
                f"compose({name!r}, ...): got a FrozenKernel, not a FrozenHelper - a kernel is a "
                f"host entry point, not a device-callable helper, and cannot be composed into "
                f"another builder (on a GPU backend a kernel cannot call another kernel). Build "
                f"the shared logic as a HelperBuilder instead."
            )
        if name in self._composed:
            raise SlotGroupError(f"'{name}' is already composed on this builder")
        if name in self._explicit_params:
            raise SlotGroupError(f"'{name}' is already declared a PARAM slot on this builder; compose() cannot reuse it")
        self._composed[name] = frozen
        return self

    def _derive_contract(self):
        """
        The (contract, data_names) this builder's template implies: a ctx.* AST
        walk (python) or `$...$` span scan (cupy). The signature is read for
        DATA names only when this builder contributes DATA (`_HAS_DATA` - a
        kernel/host block; not a helper, whose signature after ctx is
        device-call arguments, nor a template-less group).

        """
        template = self._template
        if template is None:
            return Contract(frozenset()), []
        if isinstance(template, str):
            contract = extract_cupy_contract(template)
            if self._HAS_DATA:
                from .compile_cupy import _check_cupy_data_signature
                return contract, _check_cupy_data_signature(template)
            return contract, []
        contract = extract_python_contract(template)
        if self._HAS_DATA:
            from .compile_shared import check_data_signature
            return contract, check_data_signature(template)
        return contract, []

    def _resolve_chain(self, chain: tuple, root_node: Node) -> None:
        """
        Resolve a contract chain that entered composed child `root_node`
        (chain[0]) to its end through the child's own children/slots. A chain
        that dead-ends - a segment that is neither a child nor a PARAM leaf of
        the deepest node reached, or trailing segments after a PARAM leaf that
        are not a single legal accessor - raises ContractError naming the full
        chain and the deepest node.

        """
        cur = root_node
        walked = chain[0]
        segs = chain[1:]
        i = 0
        while i < len(segs):
            seg = segs[i]
            if seg in cur.children:
                cur = cur.children[seg]
                walked = f"{walked}.{seg}"
                i += 1
                continue
            if seg in cur.slots.names(SlotKind.PARAM):
                rest = segs[i + 1 :]
                if rest == () or (len(rest) == 1 and rest[0] in self._LEGAL_ACCESSORS):
                    return
                raise ContractError(
                    f"chain 'ctx.{'.'.join(chain)}': after PARAM leaf {seg!r} of {walked!r} "
                    f"expected a single accessor {self._LEGAL_ACCESSORS}, got {'.'.join(rest)!r}"
                )
            raise ContractError(
                f"chain 'ctx.{'.'.join(chain)}' dead-ends: {seg!r} is neither a composed child "
                f"nor a PARAM slot of {walked!r} (it provides {sorted(cur.provides)})"
            )

    def _classify_roots(self, contract) -> tuple[set, set]:
        """
        Partition this template's contract roots into (derived PARAM names,
        missing names). A root that is composed is resolved through its child
        (raising on a dead-end / composed-as-Parameter chain) and contributes to
        neither set. Every other root is PARAM iff all its chains are two
        segments ending in a legal accessor, else missing. See the module
        docstring's disambiguation rules.

        """
        derived: set[str] = set()
        missing: set[str] = set()
        by_root: dict[str, list] = {}
        for chain in contract.chains:
            by_root.setdefault(chain[0], []).append(chain)
        for root, chains_r in by_root.items():
            if root == RESERVED_BK_NAME:
                continue
            if root in self._composed:
                for chain in chains_r:
                    self._resolve_chain(chain, self._composed[root])
                continue
            if all(len(c) == 2 and c[1] in self._LEGAL_ACCESSORS for c in chains_r):
                derived.add(root)
            else:
                missing.add(root)
        return derived, missing

    def missing(self) -> set:
        """
        The contract roots that are neither composed nor PARAM-able - a root
        used with a bare call or a further segment but never composed. Empty
        means freeze() will succeed (contract-wise).

        """
        contract, _data = self._derive_contract()
        _derived, missing = self._classify_roots(contract)
        return missing

    def _build(self) -> "tuple[SlotGroup, dict, Contract]":
        """
        The (slots, composed, contract) triple freeze() turns into a Frozen:
        PARAM slots derived from the template's contract merged with explicit
        param() declarations, DATA slots derived from the signature (Kernel/
        HostBlock) carrying any data() dtype contract. Raises ContractError on a
        missing root or a dead-end chain, SlotGroupError on an explicit-param
        conflict or a data() name absent from the signature.

        """
        self._check_mutable()
        contract, data_names = self._derive_contract()
        derived, missing = self._classify_roots(contract)
        if missing:
            label = getattr(self._template, "__name__", repr(self._template))
            raise ContractError(
                f"template {label!r}: ctx root(s) {sorted(missing)} are used with a call or a "
                f"further segment but never composed - compose() a frozen sub-structure under "
                f"each (a bare `ctx.X.get(...)` would instead derive a PARAM slot)"
            )
        param_names = set(derived)
        for name in self._explicit_params:
            if name in self._composed:
                raise SlotGroupError(f"'{name}' is both an explicit PARAM slot and a composed child")
            if name in derived:
                raise SlotGroupError(
                    f"param({name!r}): {name!r} is already a template-derived PARAM slot - "
                    f"drop the explicit param() (it is only for slots the template does not imply)"
                )
            param_names.add(name)

        data_set = set(data_names) if self._HAS_DATA else set()
        for name in self._data_contracts:
            if name not in data_set:
                raise SlotGroupError(
                    f"data({name!r}): {name!r} is not a DATA argument of the template signature "
                    f"{sorted(data_set)}"
                )

        sg = SlotGroup()
        for name in sorted(param_names):
            sg.add(ParamSlot(name))
        for name in data_names if self._HAS_DATA else []:
            sg.add(DataSlot(name, dtype=self._data_contracts.get(name)))
        self._frozen = True
        return sg, dict(self._composed), contract


class HelperBuilder(_Builder):
    """
    Builds a device helper (frozen.py's FrozenHelper): PARAM slots and composed
    children only. A helper's template signature after `ctx` is ordinary
    device-call arguments, not DATA slots, so it has no DATA of its own -
    data() raises. PARAM slots are derived from the template's
    contract; param() adds any the template does not imply.

    """

    _HAS_DATA = False

    def __init__(self, template: Any = None):
        super().__init__(template)

    def data(self, name: str, *, dtype: Any = None) -> "HelperBuilder":
        """Always raises: a helper takes data as a trusted call argument of its caller, never as a slot of its own."""
        raise BuildError(
            "HelperBuilder.data() is not allowed: a helper is device-only and takes "
            "data only as a trusted call argument of its caller. Declare the DATA argument on the "
            "enclosing KernelBuilder's own signature, and pass the value through."
        )

    def freeze(self) -> FrozenHelper:
        """Derive slots from the template's contract (frozen.py) and return the FrozenHelper. See _Builder._build()."""
        slots, composed, contract = self._build()
        return FrozenHelper(self._template, slots, composed, contract, shared=self._shared, synthetic=self._synthetic)


class KernelBuilder(_Builder):
    """
    Builds a kernel (frozen.py's FrozenKernel): a device entry point. PARAM
    slots are derived from the template's `ctx.X.get(...)` contract; DATA slots
    from the template's own signature after `ctx` (python) / `__global__`
    parameter list (cupy). param() adds an extra PARAM the template does not
    imply (a share canonical); data() attaches a dtype contract to a
    signature-declared DATA argument. `domain`/`block` are the launch config
    .

    """

    _HAS_DATA = True

    def __init__(self, template: Any = None, *, domain: "str | int | None" = None, block: "int | None" = None):
        super().__init__(template)
        self._domain = domain
        self._block = block

    def freeze(self) -> FrozenKernel:
        """Derive slots from the template's contract and signature (frozen.py) and return the FrozenKernel. See _Builder._build()."""
        slots, composed, contract = self._build()
        return FrozenKernel(
            self._template, slots, composed, contract,
            shared=self._shared, synthetic=self._synthetic,
            domain=self._domain, block=self._block,
        )


class GroupBuilder(_Builder):
    """
    Builds a non-callable, navigable composite (frozen.py's FrozenGroup): PARAM
    slots and composed children only, no template of its own. Because there is
    no template to derive from, EVERY one of a group's PARAM slots is declared
    explicitly via param() (the canonical leaves share_leaf collapses into);
    data() raises (a group is never a call argument's signature).

    """

    _HAS_DATA = False

    def __init__(self):
        super().__init__(None)

    def data(self, name: str, *, dtype: Any = None) -> "GroupBuilder":
        """Always raises: a group is a passive device-structure composite, never a call argument's signature."""
        raise BuildError(
            "GroupBuilder.data() is not allowed: a group is a passive, device-"
            "structure-only composite. Declare the DATA argument on whichever KernelBuilder "
            "composes this group."
        )

    def freeze(self) -> FrozenGroup:
        """Return the FrozenGroup: PARAM slots from param(), children from compose(), empty contract (no template)."""
        slots, composed, contract = self._build()
        return FrozenGroup(None, slots, composed, contract, shared=self._shared, synthetic=self._synthetic)


def freeze_helper(template, *, helpers=None, params=()):
    """Freeze a helper template with its composed children.

    ``params`` remains accepted only while feature call sites are consolidated;
    helper PARAM slots are derived from the template contract.

    """
    builder = HelperBuilder(template)
    for name, frozen in (helpers or {}).items():
        builder.compose(name, frozen)
    return builder.freeze()


def freeze_kernel(template, *, helpers=None, params=(), data=(), domain=None, block=None):
    """Freeze a kernel template with its composed children and launch domain.

    ``params`` and ``data`` remain accepted only while feature call sites are
    consolidated; kernel slots are derived from the template contract.

    """
    builder = KernelBuilder(template, domain=domain, block=block)
    for name, frozen in (helpers or {}).items():
        builder.compose(name, frozen)
    return builder.freeze()


def find_param_paths(frozen: Node, leaf_name: str, prefix: tuple = ()) -> list:
    """
    Every relative dotted path, as a `"a.b.NAME"` string, under `frozen`'s own
    children subtree whose PARAM slot is literally named `leaf_name` - the
    itemized list `share_leaf` hands to GroupBuilder.share(). Recurses through
    `.children` only (a HELPER slot with nothing composed raises earlier, at
    that structure's own freeze()/build(), never reached here). Generic over
    whether a composed node is itself a FrozenHelper or a nested FrozenGroup.

    Shared by grid/noise/visu's own factories - see grid/__init__.py's module
    docstring ("Build-phase sharing collapses the duplicate addresses") for
    why this exists.

    Parameters
    ----------
    frozen : Node
        Sub-structure to search.
    leaf_name : str
        PARAM slot name to find.
    prefix : tuple, optional
        Path segments prepended to every result; used internally for
        recursion.

    Returns
    -------
    list[str]
        Dotted relative paths to every occurrence of `leaf_name`.

    """
    paths = []
    if leaf_name in frozen.slots.names(SlotKind.PARAM):
        paths.append(".".join(prefix + (leaf_name,)))
    for name, child in frozen.children.items():
        paths.extend(find_param_paths(child, leaf_name, prefix + (name,)))
    return paths


def share_leaf(group: "GroupBuilder", canonical: str) -> None:
    """
    Declare every occurrence of a PARAM slot named `canonical` anywhere in
    `group`'s already-composed subtree as build-phase-shared with `group`'s
    own top-level `canonical` slot. A no-op if `canonical` occurs nowhere in
    the subtree (e.g. OUTLET_MASK when no block happens to reference it under
    the current config) - share() itself requires at least one path, so this
    only calls it when there is something to share.

    Parameters
    ----------
    group : GroupBuilder
        Builder whose own `canonical` PARAM slot every found occurrence
        collapses into.
    canonical : str
        PARAM slot name to search for and share.

    """
    paths = []
    for name, child in group._composed.items():
        paths.extend(find_param_paths(child, canonical, (name,)))
    if paths:
        group.share(canonical, *paths)


def classify_path(top_slots: "SlotGroup | None", top_composed: dict, segs: tuple, *, what: str):
    """
    Resolve a dotted relative path (as a segment tuple `segs`) into
    `(kind, node, slot)`, where kind is `"param"`, `"data"` or `"root"`. A leaf
    resolves against `top_slots` (a top-level slot) or a slot of a composed
    child reached through `top_composed`; a child root resolves to the composed
    Node found there (returned as `node`). Raises BuildError, tagged with
    `what` (the calling method), if the path does not resolve.

    Shared by every builder's share()/share_identical() (_Builder, RoutineBuilder,
    SequenceBuilder): `top_slots` is that builder's own top-level slots (None for
    a routine/sequence, which have none) and `top_composed` its composed children
    (a kernel/helper/group's composed subtree, or a routine/sequence's blocks).

    """
    root = segs[0]
    if len(segs) == 1:
        if top_slots is not None and root in top_slots:
            k = top_slots[root].kind
            if k is SlotKind.PARAM:
                return ("param", None, top_slots[root])
            if k is SlotKind.DATA:
                return ("data", None, top_slots[root])
            raise BuildError(f"{what} {'.'.join(segs)!r}: {root!r} names a HELPER slot, which is not shareable")
        if root in top_composed:
            return ("root", top_composed[root], None)
        raise BuildError(f"{what} {'.'.join(segs)!r}: {root!r} is not a top-level slot or composed child")
    if root not in top_composed:
        raise BuildError(f"{what} {'.'.join(segs)!r}: {root!r} is not a composed child")
    node = top_composed[root]
    walked = root
    for seg in segs[1:-1]:
        if seg not in node.children:
            raise BuildError(f"{what} {'.'.join(segs)!r}: {seg!r} is not composed under {walked!r}")
        node = node.children[seg]
        walked = f"{walked}.{seg}"
    leaf = segs[-1]
    if leaf in node.children:
        return ("root", node.children[leaf], None)
    if leaf in node.slots.names(SlotKind.PARAM):
        return ("param", None, node.slots[leaf])
    if leaf in node.slots.names(SlotKind.DATA):
        return ("data", None, node.slots[leaf])
    raise BuildError(f"{what} {'.'.join(segs)!r}: {leaf!r} is not a PARAM/DATA slot or child root under {walked!r}")


def find_identical_roots(top_composed: dict, target, prefix: tuple = ()) -> list:
    """
    Every dotted path, as a string, of a composed child root anywhere in
    `top_composed` (any depth) whose Node `is` `target`. What share_identical()
    hands to share(). See classify_path for the tree shape.

    """
    found = []
    for name, node in top_composed.items():
        p = prefix + (name,)
        if node is target:
            found.append(".".join(p))
        found.extend(find_identical_roots(dict(node.children), target, p))
    return found


def compute_share(top_slots, top_composed, existing_top_names, shared_seen, canonical, paths, as_):
    """
    The shared implementation of `.share(canonical, *paths, as_=)` for every
    builder level. Resolves and validates the specs (all same kind; shared
    roots identical; no path shared twice; `as_` not colliding), and returns
    `(new_shared, synthetic, seen_add)`:

      new_shared  {source path -> canonical path} to merge into the builder's
                  own `_shared` (the Node.shared form).
      synthetic   {as_ name -> Node|Slot} to merge into `_synthetic`, empty
                  unless `as_` was given.
      seen_add    the source paths to record as already-shared.

    See _Builder.share() for the caller-facing contract. Raises BuildError.

    """
    specs = [canonical, *paths]
    classified = []
    for spec in specs:
        segs = tuple(spec.split("."))
        kind, node, slot = classify_path(top_slots, top_composed, segs, what="share")
        classified.append((spec, segs, kind, node, slot))
    kinds = {c[2] for c in classified}
    if len(kinds) != 1:
        raise BuildError(f"share: every path must be the same kind, got { {c[0]: c[2] for c in classified} }")
    kind = next(iter(kinds))
    if kind == "root":
        ids = {id(c[3]) for c in classified}
        if len(ids) != 1:
            raise BuildError(f"share{specs}: shared roots must be the identical frozen object ('is'), got distinct objects")

    new_shared: dict = {}
    synthetic: dict = {}
    seen_add: set = set()

    if as_ is not None:
        if as_ in existing_top_names:
            raise BuildError(f"share(as_={as_!r}): collides with an existing top-level name")
        canonical_addr = (as_,)
        for spec, segs, _k, _node, _slot in classified:
            if segs in shared_seen:
                raise BuildError(f"share: {spec!r} is already shared")
            new_shared[segs] = canonical_addr
            seen_add.add(segs)
        c0 = classified[0]
        if kind == "root":
            synthetic[as_] = c0[3]
        elif kind == "param":
            synthetic[as_] = ParamSlot(as_)
        else:
            synthetic[as_] = DataSlot(as_, dtype=getattr(c0[4], "dtype", None))
    else:
        if len(classified) < 2:
            raise BuildError("share: needs at least one path besides canonical (or pass as_= to re-root a single path)")
        canonical_addr = classified[0][1]
        for spec, segs, _k, _node, _slot in classified[1:]:
            if segs in shared_seen:
                raise BuildError(f"share: {spec!r} is already shared")
            new_shared[segs] = canonical_addr
            seen_add.add(segs)

    return new_shared, synthetic, seen_add
