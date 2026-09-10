"""Compile Python-template kernels for Taichi and Quadrants."""

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
    """Remove ``ctx`` so template accesses resolve through supplied globals."""
    if func_def.args.posonlyargs and func_def.args.posonlyargs[0].arg == CTX_PARAM_NAME:
        func_def.args.posonlyargs = func_def.args.posonlyargs[1:]
    elif func_def.args.args and func_def.args.args[0].arg == CTX_PARAM_NAME:
        func_def.args.args = func_def.args.args[1:]
    else:
        raise CompileError(f"template {label!r}: first parameter must be {CTX_PARAM_NAME!r}")


def _compile_dropping_ctx(template, ctx_obj: Any, label: str, address: Address) -> FunctionType:
    """Rebuild a template so ``ctx`` resolves to the supplied context tree.

    Closure values and annotations are copied into the execution namespace
    before the rebuilt function is traced.
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
    specialized template body - a plain attribute namespace. See the module
    docstring's "The ctx tree" section for what gets attached and why a
    composed HELPER child is the raw compiled func itself rather than an
    instance of this class.

    """


def _build_ctx_node(prefix: Address, frozen: Node, bound: BoundKernel, backend: Any, bk: Any) -> _CtxNode:
    """Build the nested context object used while tracing one kernel.

Groups become namespaces; helpers become compiled device functions.
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
    return CompiledKernel(bound, compiled, data_order)
