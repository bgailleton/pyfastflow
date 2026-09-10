"""Shared checks and wrappers for compiled kernels."""

import ast
import inspect
import textwrap
from functools import lru_cache
from typing import Any, Callable

from .bound import Address, BindError, _Bound, _refcount, _same_dtype, format_address, parse_address
from .ctx import CTX_PARAM_NAME
from .errors import PyFastFlowError
from .slot import SlotKind

_LEGAL_PARAM_ACCESSORS = ("get", "set_node")


@lru_cache(maxsize=256)
def capture_template_meta(template) -> tuple[str | None, ast.AST | None]:
    """
    Return (source_text, ast) for a template. A python def is introspected; a
    raw string (CUDA source) is kept verbatim and has no AST. The source
    returned as `source_text` is `inspect.getsource`'s own indentation
    (whatever the def's nesting produced); the source parsed into `tree` is
    dedented first, so a nested def's indented body parses instead of
    raising an IndentationError - a no-op for a module-level def, which is
    already at column 0. A def with no recoverable source at all (a lambda,
    an exec'd function) still comes back with `tree = None`.

    Cached because compile_closure.py's `_compile_dropping_ctx` asks once per
    template to recover its AST, and a miss costs an inspect.getsource plus a
    parse. The tree handed back is shared by every caller of this function
    against that template: treat it as read-only.

    The cache key is the template object itself, so the bound size matters -
    unbounded, it would pin every dynamically generated template and every CUDA
    source string for the life of the process. An eviction only costs one
    re-parse.

    Parameters
    ----------
    template : callable or str
        A python def, or raw CUDA source text.

    Returns
    -------
    source_text : str or None
        None only if a python def had no recoverable source.
    tree : ast.AST or None
        None for CUDA source text, or a python def with no recoverable/
        parseable source.

    """
    if isinstance(template, str):
        return template, None
    try:
        source = inspect.getsource(template)
    except (OSError, TypeError):
        return None, None
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        tree = None
    return source, tree


class CompileError(PyFastFlowError):
    """
    Raised by the compile phase: unmet slots, an illegal PARAM accessor,
    a data-argument signature mismatch between a template and its data()
    slots, or any other structural problem caught before/while emitting
    device code. Every case names the exact address (and, for accessors, the
    exact chain) involved.

    """


def check_unmet(bound: _Bound) -> None:
    """
    Raise, listing every still-unbound address exactly as inspect() would
    print it, if `bound` has any. Every concrete `compile()` calls this
    first.

    """
    missing = bound.unmet()
    if missing:
        listing = ", ".join(format_address(a) for a in missing)
        raise CompileError(
            f"compile: unbound slot(s): {listing}\n\nfull binding contract:\n{bound.inspect()}"
        )


def check_legal_accessors(bound: _Bound) -> None:
    """
    Walk `bound`'s whole composition tree (the kernel's own frozen object,
    then every composed FrozenHelper, recursively) and raise on the first
    illegal PARAM accessor found - see the module docstring for the exact
    legal set and why it is identical on every backend.

    """
    _walk_accessors((), bound.frozen, bound)


def _walk_accessors(prefix: Address, frozen, bound: _Bound) -> None:
    param_names = frozen.slots.names(SlotKind.PARAM)
    for chain in frozen.contract.chains:
        root = chain[0]
        if root not in param_names:
            continue
        addr = prefix + (root,)
        if len(chain) != 2 or chain[1] not in _LEGAL_PARAM_ACCESSORS:
            raise CompileError(
                f"{format_address(addr)!r}: illegal PARAM accessor 'ctx.{'.'.join(chain)}' - "
                f"legal accessors are .get(...) and .set_node(...)"
            )
        if chain[1] == "set_node":
            value = bound.value_at(addr)
            if value is not None and getattr(value, "mode", None) == "const":
                raise CompileError(
                    f"{format_address(addr)!r}: 'ctx.{'.'.join(chain)}' - set_node against a "
                    f"const-mode PARAM slot is illegal (const is a baked-in literal, nothing "
                    f"to write)"
                )

    for name in frozen.slots.names(SlotKind.HELPER) | set(frozen.children):
        _walk_accessors(prefix + (name,), frozen.children[name], bound)


def check_data_signature(template) -> list[str]:
    """
    A python template's own data-argument names, in declaration order (its
    parameters after `ctx`). This is the SOURCE of a kernel/host-block's DATA
    slot set (derived at freeze, builder.py), not a check against a declared
    one; the order returned is what CompiledKernel resolves DATA addresses
    against for positional launch.

    Parameters
    ----------
    template : callable

    Returns
    -------
    list[str]
        `template`'s data-argument names, in declaration order.

    Raises
    ------
    CompileError
        `template`'s first parameter is not `ctx`.

    """
    label = getattr(template, "__name__", "?")
    params = list(inspect.signature(template).parameters)
    if not params or params[0] != CTX_PARAM_NAME:
        raise CompileError(f"template {label!r}: first parameter must be {CTX_PARAM_NAME!r}")
    return params[1:]


class CompiledKernel:
    """Callable produced by compiling one bound kernel."""

    def __init__(
        self,
        bound: _Bound,
        launch: Callable,
        data_order: list[Address],
        *,
        block: Any = None,
        domain_addr: "Address | None" = None,
        extent: "int | None" = None,
    ):
        self._bound = bound
        self._launch = launch
        self._data_order = list(data_order)
        self._data: dict[Address, Any] = {addr: bound.value_at(addr) for addr in self._data_order}
        self._block = block
        # The CuPy launch extent is a fixed int, or the length
        # of the DATA buffer at `domain_addr` read live at every launch (so
        # swap() to a shorter buffer shrinks the launch), and grid = ceil(n /
        # block). Both are None for closure backends, whose template loop owns
        # its iteration space.
        self._domain_addr = domain_addr
        self._extent = extent
        # This compiled kernel independently holds its
        # DATA bindings (swap() can re-point them after compile, so they are not
        # only the Bound's), refcounted here and released in close().
        self._closed = False
        for buf in self._data.values():
            _refcount(buf, +1)

    @property
    def data_order(self) -> list[Address]:
        """This kernel's DATA addresses, in the fixed positional order `launch` is called with."""
        return list(self._data_order)

    def data_at(self, addr: "Address | str") -> Any:
        """The buffer `swap()` currently has parked at `addr` (or the value it was compiled with)."""
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if a not in self._data:
            raise BindError(f"data_at: {format_address(a)!r} is not one of this compiled kernel's data addresses")
        return self._data[a]

    def swap(self, addr: "Address | str", buf: Any) -> "CompiledKernel":
        """
        Re-point DATA address `addr` at `buf` - a dict write, nothing else:
        no re-trace, no recompile, the compiled kernel itself is untouched.

        Parameters
        ----------
        addr : Address or str
        buf : Any
            Validated against the slot's declared dtype
            (data(..., dtype=...)), if one was declared - the same
            check bind() runs.

        """
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if a not in self._data:
            raise BindError(
                f"swap: {format_address(a)!r} is not one of this compiled kernel's data "
                f"addresses ({', '.join(format_address(x) for x in self._data_order)})"
            )
        info = self._bound.slot_info(a)
        if info.dtype is not None:
            obj_dtype = getattr(buf, "dtype", None)
            if obj_dtype is not None and not _same_dtype(obj_dtype, info.dtype):
                raise BindError(
                    f"swap({format_address(a)!r}, ...): dtype mismatch, slot declares "
                    f"{info.dtype}, got {obj_dtype}"
                )
        old = self._data.get(a)
        if old is not buf:
            _refcount(old, -1)
            _refcount(buf, +1)
        self._data[a] = buf
        return self

    def close(self) -> None:
        """
        Release this compiled kernel's hold on its DATA bindings (decrement each
        `_bound_by`) and mark it closed. Idempotent. It does not close the
        caller-owned Bound it was compiled from - that is the caller's to close.

        """
        if self._closed:
            return
        self._closed = True
        for buf in self._data.values():
            _refcount(buf, -1)

    def __call__(self):
        """
        Launch with whatever `swap()` currently holds for every DATA address,
        in `data_order`. Takes no arguments in normal use: a closure backend
        ranges over the template's own loop, and cupy computes its grid from the
        kernel's launch domain.

        """
        # DATA is represented by a DataHandle throughout the bind graph. Only
        # the backend call boundary unwraps it to the native buffer.
        args = [getattr(self._data[addr], "array", self._data[addr]) for addr in self._data_order]
        if self._domain_addr is None and self._extent is None:
            return self._launch(*args)
        if self._extent is not None:
            n = self._extent
        else:
            buf = self._data[self._domain_addr]
            n = int(buf.array.shape[0])
        b = self._block
        g = (n + b - 1) // b
        return self._launch(*args, grid=(g,), block=(b,))

    def __repr__(self) -> str:
        return f"CompiledKernel(data={[format_address(a) for a in self._data_order]})"
