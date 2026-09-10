"""
The bind phase: `Node.build()` (frozen.py) mints one of these, then bind() its
slots freely, any number of times, in any order, before compile() emits the
real device callable. See parameter.py's module docstring for the overall
build -> bind -> compile scheme.

One walk, one address table
----------------------------
`walk(node) -> (table, redirect, roots)` is the single walker for every node
kind (kernel, helper, group, hostblock, routine, sequence). It recurses
`node.children`, mints one PARAM/DATA leaf per full dotted path under each
child's own name, and returns:

  table     {full address -> _LeafInfo} - every independently-bindable leaf.
  redirect  {collapsed address -> canonical address} - build-phase-shared
            leaves that were NOT minted independently (see below).
  roots     {full child address -> Node} - every addressable child root
            reached, metadata for structural validation, not a bindable leaf.

Addressing is by qualified dotted path rooted at the explicit name a slot or a
child was given - `flux.grad.z`, never a positional `step0.*`. Every address is
a segment tuple (`Address`) internally so a path/glob layer can match per
segment; `parse_address`/`format_address` convert to and from a dotted string.
Only PARAM/DATA leaves get a table entry - a prefix naming a child rather than
one of its leaves (`flux.grad.grid` vs `flux.grad.grid.NX`) is in `roots`, not
`table`, so binding it raises "unknown address" like any other typo.

Build-phase sharing
--------------------
A node may declare (`_Builder.share()`, builder.py; `node.shared`) that several
paths in its own composed subtree mean the same value as one canonical path -
grid's `neighbour_raw.row.NX` and `is_on_edge.row.NX` both mean the grid's own
`NX`, because a device template can only call what is composed directly onto
its own scope, so grid's public helpers each re-compose the same private
`row`/`col` blocks under their own local names. Left alone, walk would mint one
independent address per occurrence; sharing collapses them so a caller binds
one `NX`, not seventeen.

A shared entry's path may name a PARAM leaf, a DATA leaf, or a whole child root
(then every leaf under it redirects to the same leaf under the canonical
child). Collapsed leaves are recorded in `redirect` (collapsed -> canonical)
instead of `table`; the canonical is minted normally. `value_at()` resolves a
redirect in one hop; bind()/unmet()/addresses()/inspect() never consult it, so
a collapsed address is genuinely absent from every caller-facing listing -
`.addresses()` after composing a D8 grid reports one `NX`.

Nested sharing: `_ShareScope` and outermost-wins
-------------------------------------------------
A shared node may itself compose another shared node (visu's hillshade group
composing grid under each of two private gradient blocks). Both layers apply at
once, so each node's `.shared` is captured as a `_ShareScope` tagged with the
full address (`start`) its paths are relative to, and scopes are threaded
through the recursion outermost-first. `_resolve_shared` checks every active
scope against each leaf, OUTERMOST first, first match winning outright.

Outermost-first is deliberate and load-bearing: an outer scope's canonical is
always an unconditionally minted address (a node's own top-level slots are
checked only against ENCLOSING scopes, never its own, so a canonical never
itself redirects), so resolving outer-first can never produce a redirect that
points at another redirect. Inner-first could: an inner scope might collapse a
leaf onto an address that the outer scope collapses further, leaving a redirect
whose target is itself redirected - which `value_at`'s single hop would not
chase. `_compress_redirects` resolves any such chain to its terminal minted
address at the end of the walk, so `value_at`'s single hop always lands.

`split` (transitional, no live caller) exempts specific relative paths from one
scope's collapse; an exempted leaf falls through to the next scope or mints
independently.

bind
----
`bind(addr, obj)` fills a leaf; rebinding is normal (immutability belongs to
the compiled artifact, not a slot). PARAM accepts any Parameter of any mode;
DATA checks the dtype declared with data(), if any. `None` is rejected -
it is reserved to mean "unbound" (see value_at). `bind_leaf` is the bulk form;
`bind_into` is the copy-down routine/sequence compile use.

Author: B.G (09/2026)
"""

from typing import Any, NamedTuple

import numpy as np

from ..pool.base import new_uid
from ..pool.base import DataHandle
from .errors import PyFastFlowError
from .frozen import FrozenGroup, FrozenHelper, FrozenKernel, Node
from .slot import SlotKind

Address = tuple[str, ...]


class BindError(PyFastFlowError):
    """
    Raised by anything in the bind phase: an unknown address, a bind() of the
    wrong kind of object or the wrong dtype, or walk() finding a wired HELPER
    slot with nothing composed into it. Every case names the exact address.

    Author: B.G (08/2026)
    """


def parse_address(addr: str) -> Address:
    """`"flux.grad.z"` -> `("flux", "grad", "z")`. Raises on an empty string."""
    if not addr:
        raise BindError("address must not be empty")
    return tuple(addr.split("."))


def format_address(addr: Address) -> str:
    """`("flux", "grad", "z")` -> `"flux.grad.z"`."""
    return ".".join(addr)


def _refcount(obj: Any, delta: int) -> None:
    """
    Adjust `obj._bound_by` by `delta` if `obj` has one (a Parameter or a
    DataHandle). Objects with no `_bound_by` - a raw backend array bound to a
    DATA slot today - are left alone; DATA refcounting activates when DATA binds
    handles (Unit 8). `None` (an unbound slot) is skipped.

    Author: B.G (09/2026)
    """
    if obj is not None and hasattr(obj, "_bound_by"):
        obj._bound_by += delta


class _LeafInfo(NamedTuple):
    """
    The fixed, never-rebound metadata walk() mints for one address: its slot
    kind, and - DATA only - the dtype declared with data() (None if
    left open). Distinct from the bound value, which lives in `_Bound._values`
    and changes freely via bind().

    Author: B.G (08/2026)
    """

    kind: SlotKind
    dtype: Any


class _ShareScope(NamedTuple):
    """
    One node's own build-phase-sharing declarations, active while walking
    anywhere inside that node's composed subtree.

    `start` is the full address this node's root sits at - every path in
    `shared_paths` is relative to it, so a leaf's relative path for this scope
    is `full_addr[len(start):]`. `shared_paths` maps a relative path (a leaf, or
    a child root) to this scope's canonical FULL address.

    Author: B.G (09/2026)
    """

    start: Address
    shared_paths: "dict[Address, Address]"


def _resolve_shared(full_addr: Address, scopes: "list[_ShareScope]") -> "Address | None":
    """
    The full canonical address `full_addr` redirects to under the active
    scopes, or None if none claim it. Scopes are checked outermost-first (see
    the module docstring); within one scope an exact match (a shared leaf, or a
    shared root named exactly) wins over a prefix match (a shared root covering
    this leaf, whose canonical gets the leaf's tail appended). The longest
    prefix wins among prefix matches, so the most specific shared root claims
    the leaf. A returned canonical may itself be redirected further; the walk's
    final compression pass resolves such chains to their terminal address.

    Author: B.G (09/2026)
    """
    for scope in scopes:
        rel = full_addr[len(scope.start) :]
        canonical = scope.shared_paths.get(rel)
        if canonical is not None:
            return canonical
        best_path: "Address | None" = None
        best_canonical: "Address | None" = None
        for path, can in scope.shared_paths.items():
            if len(path) < len(rel) and rel[: len(path)] == path:
                if best_path is None or len(path) > len(best_path):
                    best_path, best_canonical = path, can
        if best_path is not None:
            return best_canonical + rel[len(best_path) :]
    return None


def _mint_leaf(
    full: Address,
    info: _LeafInfo,
    table: "dict[Address, _LeafInfo]",
    redirect: "dict[Address, Address]",
    scopes: "list[_ShareScope]",
) -> None:
    """
    Record one leaf: a `redirect` entry to its canonical if any active scope
    collapses it, an independent `table` entry otherwise. The canonical may be
    another redirect key; `_compress_redirects` resolves the chain at the end.

    Author: B.G (09/2026)
    """
    canonical = _resolve_shared(full, scopes)
    if canonical is not None:
        redirect[full] = canonical
    else:
        table[full] = info


def _walk_node(
    prefix: Address,
    node: Node,
    table: "dict[Address, _LeafInfo]",
    redirect: "dict[Address, Address]",
    roots: "dict[Address, Node]",
    scopes: "list[_ShareScope]",
) -> None:
    """
    One level of the walk. Mints this node's own PARAM/DATA leaves (checked
    against ENCLOSING `scopes` only - a node's canonical is never redirected by
    its own sharing), pushes this node's own scope (from `node.shared`) for the
    descent, walks its synthetic roots/leaves (the targets share(as_=...)
    minted), then recurses each real child. See the module docstring.

    Author: B.G (09/2026)
    """
    for name in node.slots.names(SlotKind.PARAM):
        _mint_leaf(prefix + (name,), _LeafInfo(SlotKind.PARAM, None), table, redirect, scopes)
    for name in node.slots.names(SlotKind.DATA):
        _mint_leaf(prefix + (name,), _LeafInfo(SlotKind.DATA, node.slots[name].dtype), table, redirect, scopes)

    child_scopes = scopes
    if node.shared:
        own_shared_paths = {rel: prefix + canonical for rel, canonical in node.shared.items()}
        child_scopes = list(scopes) + [_ShareScope(prefix, own_shared_paths)]

    for syn_name, syn in node.synthetic.items():
        addr = prefix + (syn_name,)
        if isinstance(syn, Node):
            roots[addr] = syn
            _walk_node(addr, syn, table, redirect, roots, child_scopes)
        else:
            _mint_leaf(addr, _LeafInfo(syn.kind, getattr(syn, "dtype", None)), table, redirect, child_scopes)

    helper_roots = node.slots.names(SlotKind.HELPER) | set(node.children)
    for name in helper_roots:
        addr = prefix + (name,)
        if name not in node.children:
            raise BindError(
                f"'{format_address(addr)}' is a wired HELPER slot with nothing composed into "
                f"it - compose() a frozen helper under that name before build()"
            )
        child = node.children[name]
        roots[addr] = child
        _walk_node(addr, child, table, redirect, roots, child_scopes)


def _compress_redirects(
    table: "dict[Address, _LeafInfo]", redirect: "dict[Address, Address]"
) -> None:
    """
    Resolve every redirect chain to its terminal minted address, in place, so
    each collapsed address points straight at a `table` entry and `value_at`'s
    single hop always lands. A redirect target that is itself redirected
    (nested sharing over an already-internally-sharing node) is followed to the
    end; a cycle, or a terminal that names no minted address, raises BindError.

    Author: B.G (09/2026)
    """
    for src in list(redirect):
        target = redirect[src]
        seen = {src}
        while target in redirect:
            if target in seen:
                raise BindError(f"sharing cycle involving {format_address(src)!r}")
            seen.add(target)
            target = redirect[target]
        if target not in table:
            raise BindError(
                f"share: {format_address(src)!r} redirects to {format_address(target)!r}, "
                f"which is not a minted address"
            )
        redirect[src] = target


def walk(node: Node) -> "tuple[dict[Address, _LeafInfo], dict[Address, Address], dict[Address, Node]]":
    """
    Walk `node`'s whole tree and return `(table, redirect, roots)` - see the
    module docstring for each. The single walker behind every `Node.build()`.

    Author: B.G (09/2026)
    """
    table: dict[Address, _LeafInfo] = {}
    redirect: dict[Address, Address] = {}
    roots: dict[Address, Node] = {}
    _walk_node((), node, table, redirect, roots, [])
    _compress_redirects(table, redirect)
    return table, redirect, roots


def collect_share_decls(node: Node) -> "list[tuple[Address, Address]]":
    """
    This node's OWN build-phase-share declarations as `(source, canonical)`
    address pairs - one per `node.shared` entry (a leaf or a child root). Used
    by `_Bound.inspect()` to report what THIS build level collapsed, one line
    per declaration (a root share renders as one `src.* -> can.*` line, not one
    per leaf under it). Declarations made inside a composed child - a group's
    own internal leaf sharing - are that child's business and stay out of this
    listing; the collapse still shows in `.addresses()` reporting one canonical.

    Author: B.G (09/2026)
    """
    return [(src, can) for src, can in node.shared.items()]


def _format_state(info: _LeafInfo, value: Any) -> str:
    """
    The state column of one inspect() line - see _Bound.inspect.

    Author: B.G (08/2026)
    """
    if value is None:
        return "UNBOUND"
    if info.kind is SlotKind.PARAM:
        mode = getattr(value, "mode", None)
        if mode == "const":
            return f"bound(const {value.value})"
        if mode is not None:
            return f"bound({mode})"
    return "bound"


_DTYPE_SHORT = {
    "float32": "f32", "float64": "f64",
    "int32": "i32", "int64": "i64",
    "uint8": "u8", "uint32": "u32",
}


def _short_dtype(dtype: Any) -> str:
    """
    A dtype in the short spelling this package writes everywhere else ("f32",
    "i64", ...) rather than python's own repr. Tries a numpy coercion first
    (covers numpy dtypes/classes and the cupy backend's dtype objects); falls
    back to `str(dtype)` for a Taichi/Quadrants token, which already prints
    short.

    Author: B.G (08/2026)
    """
    try:
        name = np.dtype(dtype).name
    except TypeError:
        return str(dtype)
    return _DTYPE_SHORT.get(name, name)


def _same_dtype(left: Any, right: Any) -> bool:
    """Compare backend-native and short-tag dtype spellings consistently."""
    return _short_dtype(left) == _short_dtype(right)


class _Bound:
    """
    Shared machinery behind every Bound* kind. Not instantiated directly - see
    `Node.build()` (frozen.py) and `walk` above.

    Author: B.G (09/2026)
    """

    def __init__(
        self,
        node: Node,
        table: "dict[Address, _LeafInfo]",
        redirect: "dict[Address, Address] | None" = None,
        roots: "dict[Address, Node] | None" = None,
    ):
        self._uid = new_uid()
        self._frozen = node
        self._table = table
        self._values: dict[Address, Any] = {}
        # build-phase-collapsed addresses -> canonical (module docstring).
        # Consulted only by value_at(), never by bind()/unmet()/addresses()/
        # inspect(), so a collapsed address stays absent from every listing.
        self._redirect: dict[Address, Address] = dict(redirect) if redirect else {}
        # addressable child roots -> Node; structural metadata, not bindable.
        self._roots: dict[Address, Node] = dict(roots) if roots else {}
        # the Backend recorded from the first bound object that carries one (a
        # Parameter, or a DataHandle once DATA binds handles); every later such
        # object must match. `compile()` infers/checks against it. (address, be)
        self._backend = None
        self._backend_addr: "Address | None" = None
        self._closed = False

    @property
    def frozen(self) -> Node:
        """The Node this object was build()-ed from."""
        return self._frozen

    def addresses(self) -> set[Address]:
        """Every address this object has a slot for - the full, fixed address tree walk() minted."""
        return set(self._table)

    def value_at(self, addr: "Address | str") -> Any:
        """
        The object currently bound at `addr`, or None if unbound. `addr` may be
        a build-phase-collapsed address (resolved through `redirect` in one
        hop) even though it is not one of `.addresses()` - the compile phase's
        structural walks compute a full address at every leaf regardless of
        whether it was minted or collapsed, and this is the read that resolves
        either transparently. `None` unambiguously means "unbound" (bind()
        rejects None), which the copy-down in `bind_into` relies on.

        Author: B.G (09/2026)
        """
        return self._values.get(self._addr_or_redirect(addr))

    def _addr_or_redirect(self, addr: "Address | str") -> Address:
        """
        `addr` as an Address, validated against `.addresses()`, OR - if not
        itself minted - its collapsed canonical. Raises "unknown address" if
        neither. Only value_at() uses this; bind() uses `_addr` (a collapsed
        address is not independently bindable).

        Author: B.G (09/2026)
        """
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if a in self._table:
            return a
        redirected = self._redirect.get(a)
        if redirected is not None:
            return redirected
        raise BindError(
            f"unknown address {format_address(a)!r} - not one of this object's slots "
            f"(see .addresses() for the full set)"
        )

    def slot_info(self, addr: "Address | str") -> _LeafInfo:
        """This address's fixed kind/dtype, as minted by walk() - never changes after that."""
        return self._table[self._addr(addr)]

    def unmet(self) -> list[Address]:
        """
        Every address with no bound value yet, sorted. Empty means every slot
        walk() minted is filled - the precondition compile() checks first
        (compile_shared.check_unmet).

        Author: B.G (09/2026)
        """
        return sorted(addr for addr in self._table if self._values.get(addr) is None)

    def _addr(self, addr: "Address | str") -> Address:
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if a not in self._table:
            canonical = self._redirect.get(a)
            if canonical is not None:
                raise BindError(
                    f"{format_address(a)!r} is build-phase-shared and not independently "
                    f"bindable - bind its canonical {format_address(canonical)!r} instead"
                )
            raise BindError(
                f"unknown address {format_address(a)!r} - not one of this object's slots "
                f"(see .addresses() for the full set)"
            )
        return a

    def root_at(self, path: "Address | str") -> Node:
        """
        The child root Node at `path` - a real composed child or a synthetic
        root minted by share(as_=...) - resolved by exact path. Raises
        BindError for a leaf address or an unknown path. This is the structural
        surface Program bundle validation (Unit 9) resolves a bundle root
        against, rather than inferring roots from string prefixes.

        Author: B.G (09/2026)
        """
        a = parse_address(path) if isinstance(path, str) else tuple(path)
        node = self._roots.get(a)
        if node is None:
            if a in self._table:
                raise BindError(f"{format_address(a)!r} is a leaf address, not a child root")
            raise BindError(f"unknown root {format_address(a)!r} - not a composed or synthetic child root")
        return node

    def bind(self, addr: "Address | str", obj: Any) -> "_Bound":
        """
        Fill the slot at `addr` with `obj`. Rebinding is normal and overwrites.
        PARAM accepts any Parameter of any mode; DATA with a declared dtype
        checks `obj.dtype` when `obj` has one, an open DATA slot accepts
        anything. There is no HELPER case - walk() mints only PARAM/DATA leaves,
        so a prefix naming a child (`flux.grad.grid`) raises "unknown address"
        in `_addr` before any kind dispatch.

        Raises BindError if `addr` is unknown, `obj` is None (reserved for
        "unbound"), or `obj` is the wrong kind/dtype for its slot.

        Author: B.G (09/2026)
        """
        self._check_open("bind")
        if obj is None:
            raise BindError(
                f"{format_address(self._addr(addr))!r}: cannot bind None - None is reserved "
                f"to mean 'unbound' (see value_at). Rebind a real object, or leave the slot unset."
            )
        a = self._addr(addr)
        info = self._table[a]
        if info.kind is SlotKind.PARAM:
            from .parameter import Parameter

            if not isinstance(obj, Parameter):
                raise BindError(
                    f"{format_address(a)!r} is a PARAM slot; expected a Parameter, got "
                    f"{type(obj).__name__}"
                )
        else:
            assert info.kind is SlotKind.DATA
            if not isinstance(obj, DataHandle):
                raise BindError(
                    f"{format_address(a)!r} is a DATA slot; expected a DataHandle, got "
                    f"{type(obj).__name__}. Bind the handle itself, not its raw .array buffer"
                )
            if info.dtype is not None:
                obj_dtype = getattr(obj, "dtype", None)
                if obj_dtype is not None and not _same_dtype(obj_dtype, info.dtype):
                    raise BindError(
                        f"{format_address(a)!r}: dtype mismatch, slot declares {info.dtype}, "
                        f"got {obj_dtype}"
                    )
        obj_be = getattr(obj, "backend", None)
        if obj_be is not None:
            if self._backend is None:
                self._backend = obj_be
                self._backend_addr = a
            elif obj_be != self._backend:
                raise BindError(
                    f"{format_address(a)!r} binds a {obj_be.name!r}-backend object, but "
                    f"{format_address(self._backend_addr)!r} already bound a "
                    f"{self._backend.name!r}-backend object - one bound object is single-backend"
                )
        old = self._values.get(a)
        if old is not obj:
            _refcount(old, -1)
            _refcount(obj, +1)
        self._values[a] = obj
        return self

    def close(self) -> None:
        """
        Release this bound object's hold on its bindings: decrement the
        `_bound_by` of every Parameter/handle it still holds, clear its values,
        and mark it closed. Idempotent. bind()/compile() after close raise. A
        Parameter/handle can be destroyed/released only once every Bound and
        compiled object holding it has been closed. See the module docstring.

        Author: B.G (09/2026)
        """
        if self._closed:
            return
        self._closed = True
        for obj in self._values.values():
            _refcount(obj, -1)
        self._values = {}

    def _check_open(self, action: str) -> None:
        if self._closed:
            raise BindError(f"cannot {action}: this bound object is closed()")

    def bind_into(self, child_bound: "_Bound", prefix: Address) -> None:
        """
        Copy every value bound on this object at `prefix + local` down onto
        `child_bound` at `local`, for each address `child_bound` minted; an
        unbound slot (value_at returns None) is skipped. `value_at` resolves
        redirects, so a child leaf whose canonical lives at this outer level
        receives the value bound at the canonical.

        The copy-down BoundRoutine.compile()/BoundSequence.compile() run to
        fill a freshly-built per-step/per-block bound object from this outer
        address space before compiling it.

        Author: B.G (09/2026)
        """
        for local_addr in child_bound.addresses():
            val = self.value_at(prefix + local_addr)
            if val is not None:
                child_bound.bind(local_addr, val)

    def bind_leaf(
        self, mapping: dict[str, Any], *, prefix: "Address | str" = (), strict: bool = True
    ) -> "_Bound":
        """
        Bind every address under `prefix` whose last segment is a key of
        `mapping`, to that key's value - one bind() per match, same checks as an
        ordinary bind(). `prefix` restricts the match, resolving a leaf name
        recurring under two different meanings at two different prefixes.
        `strict=True` raises if any key matched no address under `prefix`.

        Author: B.G (09/2026)
        """
        p = parse_address(prefix) if isinstance(prefix, str) else tuple(prefix)
        plen = len(p)
        matched: set[str] = set()
        for addr in self._table:
            if addr[:plen] == p and addr[-1] in mapping:
                self.bind(addr, mapping[addr[-1]])
                matched.add(addr[-1])
        if strict:
            unused = sorted(set(mapping) - matched)
            if unused:
                raise BindError(f"bind_leaf(prefix={format_address(p)!r}): {unused} matched no address")
        return self

    def inspect(self) -> str:
        """
        The full binding contract, one line per address, as pasteable
        addresses, columns aligned to what the actual addresses/types need:

            flux.grad.dx       PARAM  -    bound(const 30.0)
            flux.grad.z        PARAM  -    UNBOUND
            flux.acc           DATA   f32  UNBOUND
            update.dt          PARAM  -    bound(scalar)

        PARAM and DATA share one layout; a PARAM's type column reads "-", a
        DATA's its declared dtype short ("f32") or "any" if left open. Any leaf
        name (last segment) shared by two or more addresses is listed once more
        under "Informational" - never an error; two unrelated `z`s at two
        addresses is ordinary.

        A final "Build-phase sharing" section lists each share() declaration
        that collapsed addresses - one line per shared root (`slope.lap.grid.*
        -> grid.*`) or leaf (`slope.z -> z`) - so a caller sees exactly what is
        not independently bindable and which canonical to bind instead.

        Author: B.G (09/2026)
        """
        rows: list[tuple[str, str, str, str]] = []
        for addr in sorted(self._table):
            info = self._table[addr]
            state = _format_state(info, self._values.get(addr))
            if info.kind is SlotKind.DATA:
                type_col = _short_dtype(info.dtype) if info.dtype is not None else "any"
            else:
                type_col = "-"
            rows.append((format_address(addr), info.kind.value.upper(), type_col, state))

        by_leaf: dict[str, list[Address]] = {}
        for addr in self._table:
            by_leaf.setdefault(addr[-1], []).append(addr)
        collisions = []
        for leaf, addrs in sorted(by_leaf.items()):
            if len(addrs) < 2:
                continue
            collisions.append(f"  '{leaf}': {', '.join(format_address(a) for a in sorted(addrs))}")

        if not rows:
            report = "(no slots)"
        else:
            w_addr = max(len(r[0]) for r in rows)
            w_kind = max(len(r[1]) for r in rows)
            w_type = max(len(r[2]) for r in rows)
            lines = []
            for addr_s, kind_s, type_s, state_s in rows:
                lines.append("  ".join([addr_s.ljust(w_addr), kind_s.ljust(w_kind), type_s.ljust(w_type), state_s]))
            report = "\n".join(lines)

        if collisions:
            report += "\n\nInformational - same leaf name at multiple addresses (not an error):\n"
            report += "\n".join(collisions)

        decl_lines = []
        for src, can in sorted(collect_share_decls(self._frozen)):
            if src in self._redirect:  # a shared leaf: src itself redirected
                decl_lines.append(f"  {format_address(src)} -> {format_address(can)}")
            else:  # a shared child root: its leaves redirected, not src itself
                decl_lines.append(f"  {format_address(src)}.* -> {format_address(can)}.*")
        if decl_lines:
            report += "\n\nBuild-phase sharing (collapsed - bind the canonical):\n"
            report += "\n".join(decl_lines)
        return report

    def _resolve_backend(self, backend):
        """
        The Backend to compile on: `backend` if given (a Backend,
        checked against any recorded one), else the backend recorded from a
        bound Parameter/handle. Raises CompileError on a mismatch or when none
        can be determined.

        Author: B.G (09/2026)
        """
        from .backends import Backend
        from .compile_shared import CompileError

        if backend is not None:
            if not isinstance(backend, Backend):
                raise CompileError(
                    f"compile() requires a Backend object, got {type(backend).__name__}"
                )
            be = backend
            if self._backend is not None and be != self._backend:
                raise CompileError(
                    f"compile({be.name!r}) but this object is bound to {self._backend.name!r}-"
                    f"backend objects (first at {format_address(self._backend_addr)!r})"
                )
            return be
        if self._backend is None:
            raise CompileError(
                "compile() needs a backend: none was recorded (no Parameter/handle bound carries "
                "one) - pass a Backend or bind a Parameter first"
            )
        return self._backend

    def __repr__(self) -> str:
        return f"{type(self).__name__}(uid={self._uid}, slots={len(self._table)})"


class BoundKernel(_Bound):
    """
    The bound result of build()-ing a FrozenKernel. See the module docstring.

    Author: B.G (08/2026)
    """

    def compile(self, backend=None) -> Any:
        """
        Produce a frozen, immutable callable from this object's current
        bindings. A snapshot: this BoundKernel stays live and rebindable, and a
        later compile() produces an independent callable (see compile_shared.py
        for CompiledKernel, swap(), and the checks every backend runs first).

        `backend` is a `Backend` (backends.py). It may be omitted once a backend has been recorded by
        binding a Parameter/handle - the recorded one is used. Passing a backend
        that differs from the recorded one raises CompileError.

        Author: B.G (09/2026)
        """
        self._check_open("compile")
        be = self._resolve_backend(backend)
        return be.compile_kernel(self)


class _BoundNonCallable(_Bound):
    """
    Shared base of the Bound kinds with no standalone compiled form - a device
    helper (BoundHelper) and a navigable group (BoundGroup): both are compiled
    only as part of the BoundKernel that composes them. See the module
    docstring.

    Author: B.G (09/2026)
    """

    def compile(self, backend=None) -> Any:
        """
        Always raises: this kind has no standalone compiled form on any
        backend. Compose its frozen node into a KernelBuilder and compile the
        resulting BoundKernel.

        Author: B.G (09/2026)
        """
        raise BindError(
            f"{type(self).__name__}.compile() is not supported: it has no standalone compiled "
            f"form and is compiled as part of the BoundKernel that composes it. Compose its "
            f"frozen node into a KernelBuilder and compile that BoundKernel."
        )


class BoundHelper(_BoundNonCallable):
    """
    The bound result of build()-ing a FrozenHelper. Compiled only as part of
    the BoundKernel that composes it. See the module docstring.

    Author: B.G (09/2026)
    """


class BoundGroup(_BoundNonCallable):
    """
    The bound result of build()-ing a FrozenGroup. A group is a navigable
    composite with no callable form of its own; compiled only as part of the
    BoundKernel that composes it. See the module docstring.

    Author: B.G (09/2026)
    """


# Assign each frozen leaf's Bound* class here (frozen.py cannot, being imported
# by this module). Node.build() imports this module first so these have run.
FrozenKernel.bound_cls = BoundKernel
FrozenHelper.bound_cls = BoundHelper
FrozenGroup.bound_cls = BoundGroup
