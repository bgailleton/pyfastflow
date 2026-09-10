"""CuPy Parameters and CUDA-source emission helpers."""

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

    """
    return _CTYPE[np.dtype(dtype)]


def _cuda_literal(value) -> str:
    """
    Format a resolved const value as a CUDA literal.

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

    """
    match = pattern.search(template)
    if not match:
        raise ValueError(f"could not find a {kind} function name in template source")
    return match.group(1)


def _split_args(argstr: str) -> list[str]:
    """
    Split a call-argument string on top-level commas (respecting nesting).

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

    """

    _BACKEND_NAME = "cupy"

    def __init__(self, name: str, *, dtype: str, mode: str, value, pool, shape: tuple = ()):
        """
        Declare and initialize one parameter. "scalar"/"field" modes allocate
        pooled storage immediately via `pool`; "const" stays a plain python
        value, expanded to its CUDA literal wherever a `$...$` span reads it.

        Parameters
        ----------
        name : str
        dtype : str
            Short dtype tag (``"f32"``, ``"i32"``, ...).
        mode : str
            "const", "scalar" or "field".
        value : Any
            Initial value.
        pool : DataPool
            Backing store for "scalar"/"field" modes.
        shape : tuple, optional
            Field storage shape. Field parameters require a non-empty shape.
        """
        if mode not in MODES:
            raise ValueError(f"{name}: mode must be one of {sorted(MODES)}, got {mode!r}")
        if not isinstance(dtype, str):
            raise TypeError(f"{name}: dtype must be a short tag string, got {type(dtype).__name__}")
        try:
            backend_dtype = {"i32": np.dtype(np.int32), "i64": np.dtype(np.int64),
                             "f32": np.dtype(np.float32), "u8": np.dtype(np.uint8),
                             "u32": np.dtype(np.uint32)}[dtype]
        except KeyError as exc:
            raise ValueError(f"{name}: unknown dtype tag {dtype!r}") from exc
        shape = tuple(shape)

        super().__init__()
        self.name = name
        self.dtype = dtype
        self.backend_dtype = backend_dtype
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

        """
        return self._const_value if self.mode == "const" else self._handle

    def set(self, value) -> None:
        """
        Overwrite the whole value: a device write for scalar, a full
        host->device copy for field. const is immutable - see Parameter.set.

        """
        if self.mode == "const":
            from .errors import ParameterError
            raise ParameterError(
                f"{self.name}: const parameter is immutable; build a new Parameter and "
                f"bind a new Parameter and recompile"
            )
        self._store(value)

    def _store(self, value) -> None:
        """
        Write `value` according to the mode, with no immutability check - the
        one path that may set a const, used by __init__ to place its initial
        value.

        """
        if self.mode == "const":
            self._const_value = self.backend_dtype.type(value).item()
        elif self.mode == "scalar":
            self._handle.array[...] = value
        else:  # field
            arr = np.asarray(value, dtype=self.backend_dtype).reshape(-1)
            self._handle.from_numpy(arr)

    def set_node(self, node, value) -> None:
        """
        Host-side single-cell write. scalar ignores node; const is read-only.

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

        """
        if self.mode == "const":
            return self._const_value
        if self.mode == "field":
            from .errors import ParameterError
            raise ParameterError(
                f"{self.name}: read() is for scalar/const only; a field is not meant to be "
                f"read back to the host as a whole"
            )
        return self.backend_dtype.type(self._handle.array.get()).item()

    def destroy(self) -> None:
        """
        Return any pooled storage to the pool. const mode owns none, so this
        is a no-op there. Raises ParameterError while a bound object still holds
        this Parameter (see Parameter._assert_unbound).

        """
        self._assert_unbound("destroy")
        if self._handle is not None:
            self._pool.release_data(self._handle)
            self._handle = None
