"""
Taichi/Quadrants compile phase: turns a BoundKernel into a CompiledKernel
by emitting real `ti.func`/`ti.kernel` (or `qd.func`/`qd.kernel`) objects.
Both backends share every line here - only which module `backend` points to
differs - mirroring `_closure_backend.py`'s own Taichi/Quadrants split.

The `ctx` problem and how this solves it
-----------------------------------------
`ctx` is a template's literal first parameter - `def tmpl(ctx,
i): return ctx.grad(ctx.z.get(i), i)` - not a name spliced into the
template's globals the way `_closure_backend.py`'s `specialize_closure`
works for a Parameter's own device view. That distinction
matters here: `ctx` being a real parameter means it is read via `LOAD_FAST`
bytecode, not `LOAD_GLOBAL` - splicing a value into `__globals__` under the
name `ctx`, the `specialize_closure` technique, would have **no effect at
all**, since a local parameter's bytecode never consults globals for that
name regardless of what sits there.

The fix used here is a source-level transform, not a globals trick:
`_compile_dropping_ctx` takes the template's own AST (via `capture_template_
meta`, compile_shared.py's cached inspect.getsource + ast.parse), deletes
`ctx` from the FunctionDef's own
parameter list, unparses the result, and `exec()`s it with `ctx` now bound in
that exec's globals. The body is untouched - every `ctx.foo` reference in it
still reads the name `ctx`, which now resolves via `LOAD_GLOBAL` since the
compiled code object no longer declares it as a local/parameter. Registering
the unparsed source in `linecache` under a synthetic filename before `exec()`
is what lets Taichi/Quadrants re-inspect it via their own `inspect.getsource`
during a later inlined trace (a ti.func is re-traced, not compiled once, each
place it is inlined) - the exact technique `_closure_backend.py`'s
`_fuse_group` already relies on for the same reason, proven working here the
same way.

The ctx tree
------------
What `ctx` resolves to, at every level, is a plain python object built once
per compile() by `_build_ctx_node`, walking the FrozenKernel/FrozenHelper's
own composition tree in lock-step with `BoundKernel`'s address tree:

  - one attribute per PARAM slot at that level, holding the bound
    Parameter's `device_view()` - `ctx.z.get(i)` reaches it exactly as any
    other Taichi/Quadrants-compiled template already does.
  - one attribute per composed HELPER slot, holding the **raw compiled
    `ti.func`/`qd.func` object itself** - not a wrapper - with that helper's
    own children (its PARAM device views and its own composed HELPER
    children, recursively) attached as extra attributes on that same
    function object. A compiled closure-backend function is an ordinary
    python object and accepts arbitrary attribute assignment freely, so
    `ctx.grid` is simultaneously callable (`ctx.grid(...)`, invoking the
    compiled func directly, never a wrapper) and further
    attribute-navigable (`ctx.grid.neighbour(...)`, since `neighbour` was
    attached onto the same object as `.grid`'s own attribute). This is built
    bottom-up: a composed helper's own children are compiled and attached
    before that helper's own template is compiled, since its body may
    reference them.

Every composed FrozenHelper reachable from a BoundKernel is compiled
unconditionally as part of that BoundKernel's compile(), whether or not the
immediate parent's own contract calls it bare - a `ctx.grid.neighbour(...)`
reference needs `neighbour` compiled regardless of whether `grid` itself is
ever called directly, and helper call signatures are fully trusted (no
attempt is made here to prune what nothing in this particular tree happens to
call).

`ctx.bk`, the reserved backend-intrinsics namespace (bk.py) - `ctx.bk.sqrt`,
`ctx.bk.atan2`, `ctx.bk.cast_u32`, ... - is attached to every node this
module builds, at every level of the ctx tree, not just the root: `bk` is
built once per compile() (`make_closure_bk(backend)`) and threaded through
`_build_ctx_node`'s own recursion, so it is reachable from a deeply composed
private block exactly as it is from the kernel's own template. See bk.py's
module docstring for why this namespace exists and contract.py for why it
never appears as a slot requirement.

No memoization
--------------
Every composed helper is retraced per `BoundKernel.compile()` and
`_compile_dropping_ctx` `exec()`s a fresh function each call - nothing is
cached across compiles. This is a deliberate compile-once-run-many choice:
everything in the tree today compiles once at setup and launches many times,
so a memo table would only add a cache-key surface with no payoff. Revisit
only if anything ever compiles per-frame.

The rebuilt function's linecache name is deterministic from its template
qualname and bound address. Fresh-process CUDA probes on 09/2026 compiled the
same Program SFD BoundKernels twice with isolated `offline_cache=True`
directories: Taichi's second process reported `Create kernel ... from cache`,
and Quadrants' reported both `Create kernel ... from cache` and `Loaded PTX
from cache`. Thus neither backend needs a process-unique source location for
these closure kernels, and this deterministic name remains the cache-safe
choice.

Author: B.G (08/2026)
"""

import ast
import copy
import linecache
from types import FunctionType
from typing import Any

from .bk import make_closure_bk
from .bound import Address, BoundKernel, format_address
from .compile_shared import (
    CompiledKernel,
    CompileError,
    capture_template_meta,
    check_data_signature,
    check_legal_accessors,
    check_unmet,
)
from .ctx import CTX_PARAM_NAME
from .frozen import FrozenGroup, Node
from .slot import SlotKind


def _drop_ctx_param(func_def: ast.FunctionDef, label: str) -> None:
    """
    Remove `ctx` from `func_def`'s own parameter list in place - see the
    module docstring for why this, not a globals splice, is what makes `ctx`
    resolve as a global inside the body that is left untouched.

    Author: B.G (08/2026)
    """
    if func_def.args.posonlyargs and func_def.args.posonlyargs[0].arg == CTX_PARAM_NAME:
        func_def.args.posonlyargs = func_def.args.posonlyargs[1:]
    elif func_def.args.args and func_def.args.args[0].arg == CTX_PARAM_NAME:
        func_def.args.args = func_def.args.args[1:]
    else:
        raise CompileError(f"template {label!r}: first parameter must be {CTX_PARAM_NAME!r}")


def _compile_dropping_ctx(template, ctx_obj: Any, label: str, address: Address) -> FunctionType:
    """
    Rebuild `template` with `ctx` removed from its own signature and bound
    instead as a global (`ctx_obj`) the body's untouched `ctx.*` references
    now resolve against. Registers the rebuilt source in `linecache` under a
    synthetic filename, `exec()`s it, and returns the resulting function -
    not yet decorated with `backend.func`/`backend.kernel`, see the callers
    below.

    `exec()` gives the rebuilt function only the globals dict it is handed -
    `template`'s own closure cells (a value captured lexically from an
    enclosing factory function: a baked constant, a composed helper
    reference, ...) are not carried forward by `__globals__` alone and would
    otherwise raise `NameError` the first time the rebuilt body reads that
    name. Fixed here by seeding the exec globals with `template.__code__.
    co_freevars` zipped against `template.__closure__`'s own cell contents,
    laid on top of `template.__globals__` so a free variable wins over a
    same-named module global - the lexical binding is what the template's
    author actually wrote. A captured value becomes an ordinary global in the
    rebuilt function; that is semantically fine here since tracing reads it
    once and a device template never writes back into an enclosing scope, but
    it does mean two templates built from the same closure with different
    captured values are never the same rebuilt function object - each
    compile() call mints its own.

    A data argument's own type annotation (a `ti.template()`/`qd.template()`
    marker, typically) is a second, distinct case closure cells do not cover:
    `def f(x: T): ...` evaluates the name `T` eagerly, in the enclosing
    frame, the moment the original `def` statement runs - confirmed
    empirically - so `T` is never a `LOAD_DEREF` inside `f`'s own code object
    and never appears in `co_freevars`/`__closure__` even when `T` is a local
    of an enclosing factory function, unlike a name the body itself reads.
    What survives instead is `template.__annotations__` (arg name -> already-
    evaluated value), captured by the ORIGINAL `def` statement before this
    function ever ran. Re-executing the unparsed source re-evaluates each
    annotation expression fresh, in the new exec namespace, so it needs the
    same values resolvable under the same names again: for every remaining
    (post ctx-drop) parameter whose annotation unparses to a bare name,
    that name is seeded into the exec globals from `template.__annotations__
    [that parameter's name]` - the value the original template's own
    annotation evaluated to, not a guess.

    Author: B.G (08/2026)
    """
    _, tree = capture_template_meta(template)
    if tree is None:
        raise CompileError(f"template {label!r}: no recoverable source to compile")
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not body:
        raise CompileError(f"template {label!r}: source is not a function definition")
    func_def = copy.deepcopy(body[0])
    _drop_ctx_param(func_def, label)

    module = ast.fix_missing_locations(ast.Module(body=[func_def], type_ignores=[]))
    source = ast.unparse(module)
    qualname = getattr(template, "__qualname__", label)
    filename = f"<pf:{qualname}:{format_address(address)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)

    exec_globals: dict[str, Any] = dict(getattr(template, "__globals__", {}))
    code_obj = getattr(template, "__code__", None)
    closure = getattr(template, "__closure__", None)
    if code_obj is not None and closure:
        exec_globals.update(zip(code_obj.co_freevars, (cell.cell_contents for cell in closure)))

    orig_annotations = getattr(template, "__annotations__", {})
    all_args = (
        list(func_def.args.posonlyargs) + list(func_def.args.args) + list(func_def.args.kwonlyargs)
    )
    for arg in all_args:
        if isinstance(arg.annotation, ast.Name) and arg.arg in orig_annotations:
            exec_globals[arg.annotation.id] = orig_annotations[arg.arg]

    exec_globals["ctx"] = ctx_obj
    code = compile(source, filename, "exec")
    exec(code, exec_globals)
    return exec_globals[func_def.name]


class _CtxNode:
    """
    What `ctx` (or one of its composed-helper children) resolves to inside a
    specialized template body - a plain attribute bag. See the module
    docstring's "The ctx tree" section for what gets attached and why a
    composed HELPER child is the raw compiled func itself rather than an
    instance of this class.

    Author: B.G (08/2026)
    """


def _build_ctx_node(prefix: Address, frozen: Node, bound: BoundKernel, backend: Any, bk: Any) -> _CtxNode:
    """
    Recursively build the ctx tree rooted at `frozen` (found at `prefix` in
    `bound`'s address tree), compiling every composed HELPER child - bottom
    up, so a child's own compiled func exists before its parent's template
    (which may call it) is compiled - and attaching each as both a callable
    and a further-navigable node on the returned object. See the module
    docstring.

    `bk` (bk.py's `make_closure_bk(backend)`, built once per compile() and
    threaded through every recursive call) is attached to every node at
    every level - the reserved `ctx.bk` namespace is reachable from the
    kernel's own root template and from any composed helper's, however deep,
    since a private block many levels down is exactly where noise's/visu's
    own use of it lives. See bk.py's module docstring.

    Author: B.G (08/2026)
    """
    node = _CtxNode()
    node.bk = bk
    for name in frozen.slots.names(SlotKind.PARAM):
        addr = prefix + (name,)
        param = bound.value_at(addr)
        setattr(node, name, param.device_view())

    for name in frozen.slots.names(SlotKind.HELPER) | set(frozen.children):
        child_addr = prefix + (name,)
        child_frozen = frozen.children[name]
        child_node = _build_ctx_node(child_addr, child_frozen, bound, backend, bk)
        if isinstance(child_frozen, FrozenGroup):
            # A FrozenGroup has no template of its own to compile - it is a
            # passive, non-callable composite (frozen.py). `ctx.<name>` is
            # attached exactly as built: navigable (`ctx.<name>.<member>`),
            # never callable.
            setattr(node, name, child_node)
            continue
        label = format_address(child_addr)
        raw = _compile_dropping_ctx(child_frozen.template, child_node, label, child_addr)
        compiled = backend.func(raw)
        # child_node's own attributes (its PARAM device views, its own
        # composed HELPER children) are copied onto the compiled func object
        # itself, so `ctx.<name>` is simultaneously callable (invokes this
        # func) and navigable (`ctx.<name>.<grandchild>`) - see the module
        # docstring.
        for attr_name, attr_val in vars(child_node).items():
            setattr(compiled, attr_name, attr_val)
        setattr(node, name, compiled)

    return node


def compile_kernel(bound: BoundKernel, backend: Any) -> CompiledKernel:
    """
    Checks unmet slots and legal PARAM accessors first (compile_shared.py),
    then builds the whole ctx tree and compiles the kernel's own template as
    `backend.kernel(...)`.

    Parameters
    ----------
    bound : BoundKernel
    backend : module
        `taichi` or `quadrants`.

    Returns
    -------
    CompiledKernel

    Author: B.G (08/2026)
    """
    check_unmet(bound)
    check_legal_accessors(bound)

    frozen = bound.frozen
    data_names = check_data_signature(frozen.template)
    bk = make_closure_bk(backend)
    root_node = _build_ctx_node((), frozen, bound, backend, bk)
    raw = _compile_dropping_ctx(frozen.template, root_node, "root", ())
    compiled = backend.kernel(raw)

    data_order = [(name,) for name in data_names]
    return CompiledKernel(bound, compiled, data_order, needs_launch_dims=False)
