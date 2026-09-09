"""
cupy implementation of Parameter, plus the text/emission utilities
compile_cupy.py reuses to compile a BoundKernel to a `cp.RawModule`.

A template is CUDA source text rather than a python function, since
cp.RawModule compiles source and there is no function whose globals could be
patched. Bound objects are written into that source as `$...$` spans holding a
dotted path, which keeps the in-kernel spelling the same as on the other
backends:

    $p.get(i)$        read parameter p at flat index i
    $p.set_node(i,v)$ write parameter p at flat index i
    $grid.nx.get(i)$  reach a bag member
    $helper(a, b)$    call a bound device helper

Compiling substitutes each span according to the parameter's mode - a CUDA
literal for const, or a read/write through a pointer for scalar and field.
That pointer never travels as a kernel argument: every scalar/field Parameter
a compilation unit reaches - the kernel's own bindings plus, recursively, its
helpers' - is collected once, deduplicated by uid, into a module-scope
constant block:

    struct pf_params_t { float* p_<idx>; const float* p_<idx2>; ... };
    __constant__ pf_params_t pf_params;

`<idx>` is a per-compilation-unit local index (0, 1, 2, ...) assigned the
first time this compile's traversal reaches a given Parameter, not its
process-global `uid` - `uid` still identifies the Parameter for dedup, cycle
detection and the ptr registry's keys, but never appears in emitted text, so
an unrelated allocation upstream that shifts every uid does not change this
source at all. See compile_cupy.py's `_register_ptr`.

uploaded once per compile() via cp.RawModule.get_global. A member is `const`
when nothing in the unit writes that parameter, `T*` otherwise. Every
`__global__` and `__device__` function in the module sees the same block, so a
helper reaches a bound Parameter exactly the way its caller does - there is no
argument to thread through and no call site to rewrite. At the top of each
function body one local is declared per pointer that function's own spans
reference:

    const float* __restrict__ p_<idx> = pf_params.p_<idx>;

read (or written, dropping `const`) through for the rest of that body. This is
what keeps a function's own accesses provably non-aliasing to the compiler,
the same guarantee a `__restrict__` kernel argument used to carry - reading
`pf_params.p_<idx>` directly, span by span, would lose it.

A const parameter is reached only through a `$...$` span, like every other
parameter - spans are the only grammar. A const's span expands to its CUDA
literal in place; there is no `#define` (a macro for a common identifier - N,
DIM, EPS, min - would risk silently rewriting unrelated code in the unit).

One `cp.RawModule` is built per compilation unit: the constant block, every
`__device__` helper the unit reaches (each emitted once, however many call
sites share it), and the unit's `__global__` kernel. See compile_cupy.py's
module docstring for how a unit's source is assembled and cached.

Author: B.G (07/2026)
"""

import re
from typing import Any

import cupy as cp
import numpy as np

from .parameter import MODES, Parameter

_KERNEL_NAME_RE = re.compile(r"__global__\s+void\s+(\w+)\s*\(")
# the return type is one-or-more tokens, matched non-greedily so the LAST one
# before the parameter list is the function name - `__device__ unsigned int f(`
# names f, not int.
_DEVICE_NAME_RE = re.compile(r"__device__\s+(?:[\w:\*&]+\s+)+?(\w+)\s*\(")
_KERNEL_SIG_RE = re.compile(r"(__global__\s+void\s+\w+\s*\()(.*?)(\))", re.S)

_CTYPE = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.int32): "int",
    np.dtype(np.int64): "long long",
    np.dtype(np.uint8): "unsigned char",
    np.dtype(np.uint32): "unsigned int",
}


def _ctype(dtype) -> str:
    """
    CUDA scalar type name for a (numpy) dtype.

    Author: B.G (07/2026)
    """
    return _CTYPE[np.dtype(dtype)]


def _cuda_literal(value) -> str:
    """
    Format a resolved const value as a CUDA literal.

    Author: B.G (07/2026)
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value)}f"
    return str(value)


def _extract_name(pattern: re.Pattern, template: str, kind: str) -> str:
    """
    The `__global__`/`__device__` function's own name, read out of the source
    text - that is the entry point cp.RawModule.get_function is looked up by.

    Author: B.G (07/2026)
    """
    match = pattern.search(template)
    if not match:
        raise ValueError(f"could not find a {kind} function name in template source")
    return match.group(1)


def _split_args(argstr: str) -> list[str]:
    """
    Split a call-argument string on top-level commas (respecting nesting).

    Author: B.G (07/2026)
    """
    parts, depth, cur = [], 0, ""
    for ch in argstr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _param_argname(param: Parameter, local_index: dict[int, int]) -> str:
    """
    The struct member / local variable name a Parameter's pointer is reached
    through - stable for the object's whole lifetime *within this compile*
    since it is derived from `local_index[param.uid]`, a per-compilation-unit
    index assigned in first-encounter order (see compile_cupy.py's
    `_register_ptr`), not from `uid` itself. `uid` still identifies the Parameter for dedup (two
    spans reaching the same Parameter under two different handles look up the
    same local index and therefore compute the same argname/struct member),
    but the emitted name no longer carries the process-global uid, which is
    what keeps generated source byte-stable across runs regardless of
    allocation order upstream.

    Author: B.G (07/2026)
    """
    return f"p_{local_index[param.uid]}"


def _insert_locals(body: str, local_ptrs: dict[int, dict], local_index: dict[int, int]) -> str:
    """
    Prepend one `__restrict__` local per pointer `body` itself references,
    reading through the module's `pf_params` constant block, right after the
    function's opening brace.

    Declared `const` unless this body writes that parameter anywhere - kept
    per function rather than read off the struct member (which is `const`
    only when *no* function in the whole unit writes it), so a function that
    only reads a parameter another function in the same unit writes still
    gets the non-aliasing benefit of a const-qualified local.

    Ordered by local index ascending (first-encounter order for this compile,
    see compile_cupy.py's `_register_ptr`) rather than by uid, so this declaration
    block's text does not depend on the process-global uid values a run
    happened to assign upstream.

    Author: B.G (07/2026)
    """
    if not local_ptrs:
        return body
    idx = body.find("{")
    if idx == -1:
        raise ValueError("could not find a function body to insert parameter locals into")
    decls = "".join(
        f"    {'' if e['write'] else 'const '}{e['ctype']}* __restrict__ {_argname_for(local_index[uid])} = pf_params.{_argname_for(local_index[uid])};\n"
        for uid, e in sorted(local_ptrs.items(), key=lambda kv: local_index[kv[0]])
    )
    return f"{body[: idx + 1]}\n{decls}{body[idx + 1 :]}"


def _argname_for(local_idx: int) -> str:
    """
    The struct member / local name for a pointer already assigned local index
    `local_idx` in this compile - see _param_argname, which this must stay in
    lockstep with.

    Author: B.G (07/2026)
    """
    return f"p_{local_idx}"


def _param_block_source(registry: dict[int, dict], local_index: dict[int, int]) -> str:
    """
    The `pf_params_t` struct and its `__constant__` instance for one
    compilation unit's pointer registry - empty when the unit reaches no
    scalar/field Parameter, so a unit with only consts and bare helpers emits
    no block at all.

    Member order is by local index, ascending - i.e. first-encounter order
    during this compile's traversal (see compile_cupy.py's `_register_ptr`), not by
    uid. This is what keeps the struct's text (and therefore the whole
    generated source) independent of the process-global uid values, so an
    unrelated allocation upstream that shifts every uid does not change this
    text. _upload_param_block writes pointers in the same order.

    Author: B.G (07/2026)
    """
    if not registry:
        return ""
    members = "".join(
        f"    {'' if e['write'] else 'const '}{e['ctype']}* {_argname_for(local_index[uid])};\n"
        for uid, e in sorted(registry.items(), key=lambda kv: local_index[kv[0]])
    )
    return f"struct pf_params_t {{\n{members}}};\n__constant__ pf_params_t pf_params;\n"


def _upload_param_block(module: "cp.RawModule", registry: dict[int, dict], local_index: dict[int, int]) -> None:
    """
    Copy the current pointer for every registered Parameter into the module's
    `pf_params` constant block, in the same local-index order the struct was
    emitted in (see _param_block_source).

    Runs once per compile(), synchronously - safe as an ordinary host->device
    copy anywhere a kernel launch would be.

    Author: B.G (07/2026)
    """
    if not registry:
        return
    global_ptr = module.get_global("pf_params")
    ptrs = np.array(
        [e["array"].data.ptr for _, e in sorted(registry.items(), key=lambda kv: local_index[kv[0]])],
        dtype=np.uint64,
    )
    view = cp.ndarray(ptrs.shape, dtype=np.uint64, memptr=global_ptr)
    view.set(ptrs)


class CupyParameter(Parameter):
    """
    Parameter backed by a const python value or a pooled CupyDataHandle.

    dtypes are numpy dtypes throughout, so they need no translation. There is
    no device_view() either: a parameter reaches device code when the span
    parser substitutes it into the source.

    Author: B.G (07/2026)
    """

    _BACKEND_NAME = "cupy"

    def __init__(self, name: str, *, dtype, mode: str, value, pool, shape: tuple = (), n_flat: int | None = None):
        """
        Declare and initialize one parameter. "scalar"/"field" modes allocate
        pooled storage immediately via `pool`; "const" stays a plain python
        value, expanded to its CUDA literal wherever a `$...$` span reads it.

        Parameters
        ----------
        name : str
        dtype : str or numpy dtype
            Short dtype tag (``"f32"``, ``"i32"``, ...) or the temporary
            numpy dtype spelling accepted during the feature migration.
        mode : str
            "const", "scalar" or "field".
        value : Any
            Initial value.
        pool : DataPool
            Backing store for "scalar"/"field" modes.
        shape : tuple, optional
            Field storage shape. Field parameters require a non-empty shape.
        n_flat : int, optional
            Temporary compatibility spelling for ``shape=(n_flat,)``.

        Author: B.G (07/2026)
        """
        if mode not in MODES:
            raise ValueError(f"{name}: mode must be one of {sorted(MODES)}, got {mode!r}")
        if isinstance(dtype, str):
            try:
                dtype = {"i32": np.int32, "i64": np.int64, "f32": np.float32,
                         "u8": np.uint8, "u32": np.uint32}[dtype]
            except KeyError as exc:
                raise ValueError(f"{name}: unknown dtype tag {dtype!r}") from exc
        shape = tuple(shape)
        if n_flat is not None:
            if shape:
                raise ValueError(f"{name}: pass shape= or n_flat=, not both")
            shape = (int(n_flat),)

        super().__init__()
        self.name = name
        self.dtype = dtype
        self.mode = mode
        self._pool = pool
        self._const_value: Any = None
        self._handle = None

        if mode == "scalar":
            self._handle = pool.get_data(dtype, ())
        elif mode == "field":
            if not shape:
                raise ValueError(f"{name}: field mode requires shape=(...)")
            self._handle = pool.get_data(dtype, shape)

        self._store(value)

    def _host_value(self):
        """
        The python value for const mode, the backing CupyDataHandle otherwise.

        Author: B.G (07/2026)
        """
        return self._const_value if self.mode == "const" else self._handle

    def set(self, value) -> None:
        """
        Overwrite the whole value: a device write for scalar, a full
        host->device copy for field. const is immutable - see Parameter.set.

        Author: B.G (07/2026)
        """
        if self.mode == "const":
            from .errors import ParameterError
            raise ParameterError(
                f"{self.name}: const parameter is immutable; build a new Parameter and "
                f"replace() it into the bag, then recompile"
            )
        self._store(value)

    def _store(self, value) -> None:
        """
        Write `value` according to the mode, with no immutability check - the
        one path that may set a const, used by __init__ to place its initial
        value.

        Author: B.G (07/2026)
        """
        if self.mode == "const":
            self._const_value = np.dtype(self.dtype).type(value).item()
        elif self.mode == "scalar":
            self._handle.array[...] = value
        else:  # field
            arr = np.asarray(value, dtype=self.dtype).reshape(-1)
            self._handle.from_numpy(arr)

    def set_node(self, node, value) -> None:
        """
        Host-side single-cell write. scalar ignores node; const is read-only.

        Author: B.G (07/2026)
        """
        if self.mode == "const":
            from .errors import ParameterError
            raise ParameterError(f"{self.name}: const parameter is read-only")
        if self.mode == "scalar":
            self._handle.array[...] = value
        else:  # field
            self._handle.array[node] = value

    def read(self):
        """
        Host-side scalar read - see Parameter.read for the contract. dtypes
        are numpy dtypes already here, so no translation is needed.

        Author: B.G (07/2026)
        """
        if self.mode == "const":
            return self._const_value
        if self.mode == "field":
            from .errors import ParameterError
            raise ParameterError(
                f"{self.name}: read() is for scalar/const only; a field is not meant to be "
                f"read back to the host as a whole"
            )
        return np.dtype(self.dtype).type(self._handle.array.get()).item()

    def destroy(self) -> None:
        """
        Return any pooled storage to the pool. const mode owns none, so this
        is a no-op there. Raises ParameterError while a bound object still holds
        this Parameter (see Parameter._assert_unbound).

        Author: B.G (07/2026)
        """
        self._assert_unbound("destroy")
        if self._handle is not None:
            self._pool.release_data(self._handle)
            self._handle = None
