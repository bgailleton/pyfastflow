"""
A composite's structural contract - the set of ctx.* chains its template
actually touches - derived, never hand-authored, from the template's own
source.

Two extractors read that source and produce the same kind of thing, a
Contract: a frozen set of chains, each chain the maximal tuple of dotted
segments following `ctx` - every reference through ctx counts, whether or not
it is ever called. `ctx.z.get(i)` and a hypothetical bare `ctx.grid.nx` used
as a value both contribute a chain; nothing about the grammar privileges a
call over any other use. What a chain's own trailing segment means for
device emission - is it a required `.get`, is a bare non-call reference even
legal - is a backend question, deliberately not decided here (see slot.py's
module docstring on PARAM access being uniform across modes, and the compile
phase's `check_legal_accessors`, compile_shared.py, for where "is this
spelling legal" actually gets enforced).

extract_python_contract(template)   a python def, read by inspect.getsource +
                                     ast - STATIC ANALYSIS, the template is
                                     never called. A Taichi/Quadrants template
                                     cannot run outside kernel-trace context,
                                     so this is the only sound way to find out
                                     what it touches, and it is also why `ctx`
                                     must appear only in the template body
                                     itself - a lambda, an exec'd function, or
                                     `ctx` passed into a nested python def the
                                     walk cannot see through, all raise
                                     ContractError rather than silently
                                     under-reporting the contract.
extract_cupy_contract(source)       CUDA text already carrying `$...$` spans.
                                     The template is an f-string fully
                                     materialised at build time, so this reads
                                     the final string directly - no
                                     inspect.getsource, and none of the
                                     python surface's restrictions apply,
                                     since there is no python callable to lose
                                     sight of in the first place.

`ctx.bk` (RESERVED_BK_NAME, bk.py) is grammar the python surface recognises
and drops rather than records: a chain rooted at `bk` is the reserved
backend-intrinsics namespace (`ctx.bk.sqrt(x)`, ...), never a slot
requirement, so extract_python_contract never adds one to the returned
Contract - see bk.py's module docstring for the full mechanism, including why
this namespace exists at all and why cupy's own extractor is deliberately
left untouched (`ctx.bk` is not part of the cupy template surface).

A Contract carries only its chains; nothing here decides whether a chain is
satisfied. That check moved to freeze() (builder.py): every chain rooted at a
composed child is resolved to its end through the child's own children/slots
(the whole tree is known by then), and every other root is classified PARAM
(a two-segment `.get`/`.set_node`) or reported missing.

Author: B.G (08/2026)
"""

import ast
import inspect
import re
import textwrap
from typing import Callable

from .bk import RESERVED_BK_NAME
from .ctx import CTX_PARAM_NAME
from .slot import BuildError

Chain = tuple[str, ...]


class ContractError(BuildError):
    """
    Raised when a template's source cannot be turned into a contract (no
    recoverable source, `ctx` not the first parameter, a malformed span), or
    when a derived contract is checked against a candidate that does not
    satisfy it.

    Author: B.G (08/2026)
    """


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class Contract:
    """
    A composite's derived structural contract: the frozen set of ctx.* chains
    its template touches. See the module docstring.

    Author: B.G (08/2026)
    """

    def __init__(self, chains: frozenset[Chain]):
        self._chains = frozenset(chains)

    @property
    def chains(self) -> frozenset[Chain]:
        """Every chain this contract requires, as a segment tuple each."""
        return self._chains

    @property
    def roots(self) -> set[str]:
        """The first segment of every chain - the ctx.* names this contract references directly."""
        return {chain[0] for chain in self._chains if chain}

    def __repr__(self) -> str:
        if not self._chains:
            return "Contract()"
        body = ", ".join("ctx." + ".".join(c) for c in sorted(self._chains))
        return f"Contract({body})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Contract) and self._chains == other._chains

    def __hash__(self) -> int:
        return hash(self._chains)


# ---------------------------------------------------------------------------
# python surface: static AST walk
# ---------------------------------------------------------------------------


def _ctx_chain(node: ast.AST) -> Chain | None:
    """
    If `node` is an Attribute/Name chain rooted at `ctx`, its segments in
    source order (`ctx.grid.neighbour` -> `("grid", "neighbour")`); else None.

    Author: B.G (08/2026)
    """
    segments: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        segments.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name) and cur.id == CTX_PARAM_NAME:
        segments.reverse()
        return tuple(segments) if segments else None
    return None


class _ChainVisitor(ast.NodeVisitor):
    """
    Collects every maximal ctx.* chain in a template body, called or not.

    Only `visit_Attribute` is overridden: when a node's own chain resolves
    all the way down to `ctx` (see `_ctx_chain`), that is by construction the
    maximal chain at this point in the tree - a `ctx.grid.neighbour` node's
    `.value` is `ctx.grid`, a strict prefix, never a separate reference worth
    recording on its own - so this records the chain and does not descend
    into `node.value`. A node that does not resolve to ctx falls through to
    generic_visit, so a ctx chain nested anywhere within it (a call argument,
    a binary operand, ...) is still found by ordinary recursion. Whether the
    chain is then the func of a Call or used bare as a value makes no
    difference here - both shapes are recorded identically (see the module
    docstring).

    A chain rooted at RESERVED_BK_NAME (`ctx.bk.sqrt(x)`, ...) is dropped
    instead of recorded - see the module docstring and bk.py. Nothing further
    down such a chain needs a visit of its own (it resolves entirely to
    Attribute/Name nodes already fully consumed by `_ctx_chain`), so this is
    a plain early return, not a call into generic_visit.

    Author: B.G (08/2026)
    """

    def __init__(self):
        self.chains: set[Chain] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _ctx_chain(node)
        if chain is not None:
            if chain[0] == RESERVED_BK_NAME:
                return
            self.chains.add(chain)
            return
        self.generic_visit(node)


def _get_function_ast(template: Callable) -> ast.FunctionDef:
    """
    The single FunctionDef node for `template`'s own source, dedented and
    parsed. Raises ContractError - naming the template - if the source
    cannot be recovered (a lambda, an exec'd function) or does not parse down
    to one function definition.

    Author: B.G (08/2026)
    """
    name = getattr(template, "__name__", repr(template))
    try:
        source = inspect.getsource(template)
    except (OSError, TypeError) as exc:
        raise ContractError(
            f"template {name!r}: no recoverable source (a lambda or exec'd function cannot "
            f"be statically analysed - ctx's contract can only be derived from a def with "
            f"real source)"
        ) from exc
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError as exc:
        raise ContractError(f"template {name!r}: source does not parse: {exc}") from exc
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(body) != 1:
        raise ContractError(
            f"template {name!r}: expected source recoverable to exactly one function "
            f"definition, got {len(body)}"
        )
    return body[0]


def extract_python_contract(template: Callable) -> Contract:
    """
    The Contract a python template requires, by static AST walk over its own
    source. Never calls `template` - see the module docstring for why.

    Enforces that `ctx` is the template's first parameter (positional or
    positional-or-keyword) - a template reaching a `ctx` handed to it under
    another name, or not receiving one as its first argument at all, is
    rejected here rather than silently producing an empty or wrong contract.

    Parameters
    ----------
    template : callable
        A plain python `def` with real, statically recoverable source.

    Raises
    ------
    ContractError
        Source cannot be recovered, or `ctx` is not the first parameter.

    Author: B.G (08/2026)
    """
    fn = _get_function_ast(template)
    name = getattr(template, "__name__", fn.name)
    params = fn.args.posonlyargs + fn.args.args
    if not params or params[0].arg != CTX_PARAM_NAME:
        got = params[0].arg if params else "(no parameters)"
        raise ContractError(
            f"template {name!r}: first parameter must be named {CTX_PARAM_NAME!r}, got {got!r}"
        )
    visitor = _ChainVisitor()
    visitor.visit(fn)
    return Contract(frozenset(visitor.chains))


# ---------------------------------------------------------------------------
# cupy surface: span text scan
# ---------------------------------------------------------------------------

_SPAN_RE = re.compile(r"\$(.*?)\$", re.S)
_PATH_RE = re.compile(r"^([\w.]+)")


def extract_cupy_contract(source: str) -> Contract:
    """
    The Contract a cupy (CUDA source text) template requires, by scanning its
    already-materialised `$...$` spans for ones prefixed `ctx.` - see
    compile_cupy.py for the span resolver this reads, and the module
    docstring for why this needs no AST and none of the python surface's
    restrictions: the template is a plain string by the time this runs,
    nothing to lose sight of.

    A span not prefixed `ctx.` (a bound plain value, a bare const name used
    outside any span) contributes nothing - only ctx-rooted spans are part of
    this template's structural contract. A span's own trailing `(...)` is not
    part of the recorded chain and its presence or absence makes no
    difference - `$ctx.grid.neighbour(i, k)$` and a hypothetical bare
    `$ctx.grid.neighbour$` both record `("grid", "neighbour")` (see the
    module docstring: every reference through ctx counts, called or not).

    Parameters
    ----------
    source : str
        Fully materialised CUDA source text.

    Raises
    ------
    ContractError
        A `$...$` span's contents do not start with a dotted path.

    Author: B.G (08/2026)
    """
    chains: set[Chain] = set()
    for match in _SPAN_RE.finditer(source):
        inner = match.group(1).strip()
        path_match = _PATH_RE.match(inner)
        if path_match is None:
            raise ContractError(f"malformed span: ${inner}$")
        parts = path_match.group(1).split(".")
        if parts[0] != CTX_PARAM_NAME:
            continue
        segments = tuple(parts[1:])
        if segments:
            chains.add(segments)
    return Contract(frozenset(chains))
