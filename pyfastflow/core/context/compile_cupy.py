"""
cupy compile phase: turns a BoundKernel into a CompiledKernel by assembling
CUDA source text and building a `cp.RawModule` from it.

Reuses cupy_backend.py's pure text/emission utilities - dtype/literal
formatting, the `pf_params` constant-block and `__restrict__`-local
machinery, `__global__`/`__device__` function-name extraction - which know
only about Parameter objects and plain text (see cupy_backend.py's module
docstring for the block's exact shape). The span *resolver* lives here: a
`$ctx.path$` span resolves against one BoundKernel's address tree (bound.py)
- `$ctx.z.get(i)$`, `$ctx.grid.neighbour(i, k)$`, the same grammar
contract.py derives.

Composed helpers become `__device__` functions
------------------------------------------------
Every composed FrozenHelper reachable from `bound` gets its own `__device__`
function, unconditionally (mirroring compile_closure.py's reasoning: a
`ctx.grid.neighbour(...)` span needs `neighbour` emitted regardless of
whether `grid` itself is ever called bare). Its C name is derived from its
own full address (`pf_flux_grad_grid_neighbour` for address `flux.grad.grid.
neighbour`), which is unique within one compile by construction (build()
never mints two different composed subtrees under the same address) - no
uid-based mangling needed, unlike `_cupy_blocks.py`'s per-make_grid-call
`new_uid()` tag, since there is exactly one BoundKernel's address tree per
compile here, not several independently-built grids sharing one module.
`_emit_device_func` renames the template's own declared function name to
that address-derived name in the emitted text (the template author's own
choice of name in source is never seen by the caller); a helper already
emitted once in this compile (reachable from two different addresses -
uncommon here since addresses are already unique, but the memo guards a
cycle regardless) is reused, not re-emitted.

Author: B.G (08/2026)
"""

import re
from typing import Any

import cupy as cp

from .bound import Address, BoundKernel, format_address
from .compile_shared import CompiledKernel, CompileError, check_legal_accessors, check_unmet
from .cupy_backend import (
    _DEVICE_NAME_RE,
    _KERNEL_NAME_RE,
    _KERNEL_SIG_RE,
    _cuda_literal,
    _ctype,
    _extract_name,
    _insert_locals,
    _param_argname,
    _param_block_source,
    _split_args,
    _upload_param_block,
)
from .ctx import CTX_PARAM_NAME
from .frozen import FrozenGroup, Node
from .parameter import Parameter
from .slot import SlotKind

# Default cupy threads-per-block when a kernel's `block=` is unset. Moves to
# Backend.default_block in Unit 5.
DEFAULT_BLOCK = 256

_SPAN_RE = re.compile(r"\$(.*?)\$", re.S)
_CALL_RE = re.compile(r"([\w.]+)\s*(?:\((.*)\))?\s*$", re.S)
_CONSTANT_DECL_RE = re.compile(r"__constant__\s+[\w:\*&]+\s+(\w+)\s*(?:\[[^\]]*\])?\s*=")


class _EmitState:
    """
    Everything one compile() accumulates across every `__device__`/
    `__global__` body it parses - the pointer registry and its
    first-encounter local-index map (handed straight to cupy_backend.py's
    `_param_block_source`/`_upload_param_block`/`_insert_locals`), and the
    dependency-first, dedup-by-name map of every composed helper's own
    `__device__` source.

    Author: B.G (08/2026)
    """

    def __init__(self):
        self.registry: dict[int, dict] = {}
        self.local_index: dict[int, int] = {}
        self.device_srcs: dict[str, "str | None"] = {}
        # Finalization order, distinct from `device_srcs`' own insertion
        # order: a name is reserved (`= None`) in `device_srcs` *before*
        # `_ensure_emitted` recurses into whatever it calls, so a name's
        # position in `device_srcs` itself is caller-before-callee - the
        # wrong direction for C, which needs a callee's definition (or at
        # least a declaration) above its caller. `emit_order` instead
        # records a name only once its own body is fully resolved, i.e.
        # child-before-parent, which is what `compile_kernel` must emit in.
        self.emit_order: list[str] = []


def _register_ptr(state: _EmitState, param: Parameter, write: bool, local_ptrs: dict[int, dict]) -> str:
    uid = param.uid
    entry = state.registry.get(uid)
    if entry is None:
        entry = {"ctype": _ctype(param.dtype), "write": False, "array": param.handle().array}
        state.registry[uid] = entry
    if write:
        entry["write"] = True
    if uid not in state.local_index:
        state.local_index[uid] = len(state.local_index)
    local = local_ptrs.get(uid)
    if local is None:
        local_ptrs[uid] = {"ctype": entry["ctype"], "write": write}
    elif write:
        local["write"] = True
    return _param_argname(param, state.local_index)


def _expand_param(state: _EmitState, param: Parameter, method: str, call_args: list[str], local_ptrs: dict) -> str:
    if method == "get":
        if param.mode == "const":
            return _cuda_literal(param.value)
        argname = _register_ptr(state, param, write=False, local_ptrs=local_ptrs)
        if param.mode == "scalar":
            return f"{argname}[0]"
        idx = call_args[0] if call_args else "0"
        return f"{argname}[{idx}]"
    if param.mode == "const":
        raise CompileError(f"{param.name}: const parameter is read-only")
    if len(call_args) != 2:
        raise CompileError(f"{param.name}: set_node(node, value) takes two arguments")
    argname = _register_ptr(state, param, write=True, local_ptrs=local_ptrs)
    node, val = call_args
    return f"{argname}[0] = {val}" if param.mode == "scalar" else f"{argname}[{node}] = {val}"


def _c_name(addr: Address) -> str:
    return "pf_" + "_".join(addr)


def _resolve_chain(
    state: _EmitState,
    segs: list[str],
    call_args: list[str],
    argstr: "str | None",
    prefix: Address,
    frozen: Node,
    bound: BoundKernel,
    local_ptrs: dict,
) -> str:
    """
    Resolve one span's `ctx.<segs...>` path, rooted at `frozen`/`prefix` in
    `bound`'s address tree - see the module docstring for the two shapes
    (PARAM leaf, composed HELPER call/descent).

    Author: B.G (08/2026)
    """
    if not segs:
        raise CompileError("span '$ctx$' names nothing")
    root = segs[0]
    addr = prefix + (root,)

    if root in frozen.slots.names(SlotKind.PARAM):
        if len(segs) != 2 or segs[1] not in ("get", "set_node"):
            raise CompileError(
                f"{format_address(addr)!r}: illegal PARAM accessor 'ctx.{'.'.join(segs)}' - "
                f"legal accessors are .get(...) and .set_node(...)"
            )
        param = bound.value_at(addr)
        return _expand_param(state, param, segs[1], call_args, local_ptrs)

    if root in frozen.slots.names(SlotKind.HELPER) or root in frozen.children:
        child_frozen = frozen.children[root]
        if len(segs) == 1:
            if isinstance(child_frozen, FrozenGroup):
                raise CompileError(
                    f"{format_address(addr)!r}: a FrozenGroup composite is not callable - "
                    f"reference one of its members instead (ctx.{'.'.join(segs)}.<member>)"
                )
            fname = _ensure_emitted(state, addr, child_frozen, bound)
            return f"{fname}({argstr if argstr is not None else ''})"
        return _resolve_chain(state, segs[1:], call_args, argstr, addr, child_frozen, bound, local_ptrs)

    raise CompileError(f"{format_address(addr)!r}: no such PARAM/HELPER slot on 'ctx.{'.'.join(segs)}'")


def _make_repl(state: "_EmitState", prefix: Address, frozen: Node, bound: BoundKernel, local_ptrs: dict):
    def _repl(match: re.Match) -> str:
        cm = _CALL_RE.match(match.group(1).strip())
        if cm is None:
            raise CompileError(f"malformed span: ${match.group(1)}$")
        path = cm.group(1).split(".")
        argstr = cm.group(2)
        call_args = _split_args(argstr) if argstr is not None else []
        if path[0] != CTX_PARAM_NAME:
            raise CompileError(f"span '${match.group(1)}$' is not ctx-rooted")
        return _resolve_chain(state, path[1:], call_args, argstr, prefix, frozen, bound, local_ptrs)

    return _repl


def _mangle_constants(body: str, c_name: str) -> str:
    """
    Rename every `__constant__` symbol `body` itself declares to a name
    derived from `c_name` (this device block's own address-mangled function
    name), consistently everywhere it appears in `body` - the declaration
    and every use, both already in `body` since this runs on one block's own
    text. Extends `_ensure_emitted`'s existing per-address renaming (until
    this, applied only to the block's own `__device__` function name) to any
    *other* top-level symbol a template happens to declare - a `__constant__`
    lookup table backing a runtime-data if-ladder, in practice (grid's own
    `delta` block, _cupy_blocks.py) - which needs exactly the same
    per-address uniqueness the function name already gets: a FrozenHelper
    composed at two different addresses in one compile is emitted twice
    (`_ensure_emitted` memoizes by the mangled *function* name, which already
    differs per address), and without this, both emissions would declare the
    identical `__constant__` symbol name and collide at NVRTC compile time.

    Author: B.G (08/2026)
    """
    for orig in dict.fromkeys(_CONSTANT_DECL_RE.findall(body)):
        body = re.sub(rf"\b{re.escape(orig)}\b", f"{c_name}_{orig}", body)
    return body


def _ensure_emitted(state: _EmitState, addr: Address, frozen: Node, bound: BoundKernel) -> str:
    """
    This composed helper's own `__device__` C function name, emitting its
    source into `state.device_srcs` on first reach (memoized by name, so a
    cycle - or the same address reached twice, which cannot currently happen
    since addresses are already unique per compile - never re-emits).

    Author: B.G (08/2026)
    """
    name = _c_name(addr)
    if name in state.device_srcs:
        return name
    state.device_srcs[name] = None  # reserve, guards a helper cycle
    orig_match = _DEVICE_NAME_RE.search(frozen.template)
    if orig_match is None:
        raise CompileError(f"{format_address(addr)!r}: template has no recoverable __device__ function name")
    renamed = frozen.template[: orig_match.start(1)] + name + frozen.template[orig_match.end(1) :]
    renamed = _mangle_constants(renamed, name)
    local_ptrs: dict[int, dict] = {}
    body = _SPAN_RE.sub(_make_repl(state, addr, frozen, bound, local_ptrs), renamed)
    body = _insert_locals(body, local_ptrs, state.local_index)
    state.device_srcs[name] = body
    state.emit_order.append(name)
    return name


def _check_cupy_data_signature(template: str) -> list[str]:
    """
    The `__global__` kernel's own C parameter names, in source order. The
    SOURCE of a cupy kernel's DATA slot set (derived at freeze, builder.py),
    cupy's text-source counterpart to compile_shared.py's check_data_signature
    (there is no python `inspect.signature` to read here).

    Author: B.G (08/2026)
    """
    match = _KERNEL_SIG_RE.search(template)
    if match is None:
        raise CompileError("template has no recoverable __global__ signature")
    argstr = match.group(2).strip()
    parts = _split_args(argstr) if argstr else []
    return [p.strip().rsplit(None, 1)[-1].lstrip("*") for p in parts]


def compile_kernel(bound: BoundKernel, *, grid: Any = None, block: Any = None) -> CompiledKernel:
    """
    Compile `bound` to a cupy `cp.RawModule`. Checks unmet slots and legal
    PARAM accessors first (compile_shared.py), then emits the kernel's own
    `__global__` body and every composed helper's `__device__` source it
    reaches, in dependency order, assembles the `pf_params` constant block,
    builds the module and uploads the block.

    Parameters
    ----------
    bound : BoundKernel
    grid, block : optional
        Launch-dimension defaults for the returned CompiledKernel (see
        CompiledKernel.__call__) - cupy has no auto-ranging equivalent to
        Taichi/Quadrants, so a caller must supply them here or at call time.

    Returns
    -------
    CompiledKernel

    Author: B.G (08/2026)
    """
    check_unmet(bound)
    check_legal_accessors(bound)

    frozen = bound.frozen
    template = frozen.template
    data_names = _check_cupy_data_signature(template)

    state = _EmitState()
    kernel_name = _extract_name(_KERNEL_NAME_RE, template, "__global__")
    local_ptrs: dict[int, dict] = {}
    kbody = _SPAN_RE.sub(_make_repl(state, (), frozen, bound, local_ptrs), template)
    kbody = _insert_locals(kbody, local_ptrs, state.local_index)
    if 'extern "C"' not in kbody:
        kbody = kbody.replace("__global__", 'extern "C" __global__', 1)

    source = "\n".join(
        [_param_block_source(state.registry, state.local_index)]
        + [state.device_srcs[name] for name in state.emit_order]
        + [kbody]
    )

    module = cp.RawModule(code=source)
    _upload_param_block(module, state.registry, state.local_index)
    raw = module.get_function(kernel_name)

    def launch(*args, grid, block):
        g = (grid,) if isinstance(grid, int) else tuple(grid)
        b = (block,) if isinstance(block, int) else tuple(block)
        return raw(g, b, tuple(args))

    data_order = [(name,) for name in data_names]

    # Launch domain (Unit 4): domain names one of this kernel's DATA args (the
    # launch extent is that buffer's length at launch time) or is a fixed int;
    # block is threads/block (default DEFAULT_BLOCK - moves to Backend.
    # default_block in Unit 5). A domain of None falls back to the compile-time
    # grid/block compat path.
    domain = frozen.domain
    domain_addr = None
    extent = None
    kernel_block = block
    if isinstance(domain, str):
        if domain not in data_names:
            raise CompileError(
                f"domain={domain!r} is not one of this kernel's DATA arguments {data_names}"
            )
        domain_addr = (domain,)
        kernel_block = frozen.block or DEFAULT_BLOCK
    elif isinstance(domain, int):
        extent = domain
        kernel_block = frozen.block or DEFAULT_BLOCK
    return CompiledKernel(
        bound, launch, data_order,
        needs_launch_dims=True, grid=grid, block=kernel_block,
        domain_addr=domain_addr, extent=extent,
    )
