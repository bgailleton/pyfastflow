"""Shared Parameter support for Python-template backends."""

from types import FunctionType
from typing import Any, ClassVar

import numpy as np

from .parameter import MODES, Parameter


def specialize_closure(template, globals_: dict[str, Any]) -> FunctionType:
    """
    Rebuild `template` as a new function whose globals carry `globals_`,
    leaving the original untouched.

    The code object is reused as-is; only the globals differ, which is what
    makes a name in the template body resolve to a bound object. Defaults,
    annotations and the rest are copied over so the result still introspects
    like the template it came from.

    Parameters
    ----------
    template : FunctionType
        Function to rebuild.
    globals_ : dict[str, Any]
        Names to inject into the rebuilt function's globals.

    Returns
    -------
    FunctionType
        A new function sharing `template`'s code object but with `globals_`
        merged into its globals.

    """
    source = getattr(template, "__wrapped__", template)
    func_globals = dict(source.__globals__)
    func_globals.update(globals_)

    specialised = FunctionType(
        source.__code__,
        func_globals,
        source.__name__,
        source.__defaults__,
        source.__closure__,
    )
    specialised.__kwdefaults__ = source.__kwdefaults__
    specialised.__annotations__ = dict(source.__annotations__)
    specialised.__doc__ = source.__doc__
    specialised.__qualname__ = source.__qualname__
    return specialised


class ClosureParamDeviceView:
    """
    What a Parameter looks like from inside device code.

    `.get` and `.set_node` are compiled device funcs, so a template body reads
    `p.get(i)` and writes `p.set_node(i, v)` the same way whatever the
    parameter's mode. A const parameter is read-only and carries no `.set_node`
    at all, which turns a write to one into a trace-time error.

    """

    def __init__(self, name: str, get_fn, set_fn=None):
        self._name = name
        self.get = get_fn
        if set_fn is not None:
            self.set_node = set_fn


class ClosureBackendParameter(Parameter):
    """
    Parameter backed by a const python value or by pooled device storage.

    Concrete backends subclass this and pin `_backend` to their module; the
    dtype mapping and the device view are written once here against the names
    both modules share.

    """

    _backend: ClassVar[Any]

    def __init__(self, name: str, *, dtype: str, mode: str, value, pool, shape: tuple = ()):
        """
        Declare one parameter and give it its initial value.

        scalar and field take pooled storage straight away; const stays a
        plain python value.

        Parameters
        ----------
        name : str
        dtype : str
            Short dtype tag (``"f32"``, ``"i32"``, ...).
        mode : str
            One of MODES ("const", "scalar", "field").
        value : Any
            Initial value.
        pool : Pool
            Device-buffer pool backing scalar/field storage.
        shape : tuple, optional
            Field storage shape. Field parameters require a non-empty shape.
        Raises
        ------
        ValueError
            If `mode` is not in MODES, or field mode is given without a
            shape.

        """
        if mode not in MODES:
            raise ValueError(f"{name}: mode must be one of {sorted(MODES)}, got {mode!r}")
        if not isinstance(dtype, str):
            raise TypeError(f"{name}: dtype must be a short tag string, got {type(dtype).__name__}")
        try:
            backend_dtype = getattr(self._backend, dtype)
        except AttributeError as exc:
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
        self._device_view: "ClosureParamDeviceView | None" = None

        if mode == "scalar":
            self._handle = pool.get_data(dtype, ())
        elif mode == "field":
            if not shape:
                raise ValueError(f"{name}: field mode requires shape=(...)")
            self._handle = pool.get_data(dtype, shape)

        self._store(value)

    @classmethod
    def _numpy_dtype(cls, dtype):
        """
        Map a backend dtype (`ti.*`/`qd.*`) to the numpy dtype used for
        host-side (de)serialization.

        """
        backend = cls._backend
        if dtype == backend.u8:
            return np.uint8
        if dtype == backend.i32:
            return np.int32
        if dtype == backend.i64:
            return np.int64
        return np.float32

    def _host_value(self):
        """
        The python value for const mode, the backing DataHandle otherwise.

        """
        return self._const_value if self.mode == "const" else self._handle

    def set(self, value) -> None:
        """
        Overwrite the whole value: a device write for scalar, a full
        host->device copy for field. const is immutable - see Parameter.set.

        Raises
        ------
        ValueError
            If this parameter's mode is const.

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
            self._const_value = self._numpy_dtype(self.backend_dtype)(value).item()
        elif self.mode == "scalar":
            self._handle.array[None] = value
        else:  # field
            arr = np.asarray(value, dtype=self._numpy_dtype(self.backend_dtype)).reshape(-1)
            self._handle.array.from_numpy(arr)

    def set_node(self, node, value) -> None:
        """
        Host-side single-cell write. scalar ignores node; const is read-only.

        Raises
        ------
        ValueError
            If this parameter's mode is const.

        """
        if self.mode == "const":
            from .errors import ParameterError

            raise ParameterError(f"{self.name}: const parameter is read-only")
        if self.mode == "scalar":
            self._handle.array[None] = value
        else:  # field
            self._handle.array[node] = value

    def read(self):
        """
        Host-side scalar read - see Parameter.read for the contract.

        Raises
        ------
        ValueError
            If this parameter's mode is field.

        """
        if self.mode == "const":
            return self._const_value
        if self.mode == "field":
            from .errors import ParameterError

            raise ParameterError(
                f"{self.name}: read() is for scalar/const only; a field is not meant to be "
                f"read back to the host as a whole"
            )
        return self._numpy_dtype(self.backend_dtype)(self._handle.array.to_numpy()).item()

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
            self._device_view = None  # a cached view closes over the released handle

    def device_view(self) -> ClosureParamDeviceView:
        """
        This parameter's device accessor, built on first use and kept.

        The compiled funcs come out identical every time, so one view serves
        every kernel that binds this parameter, for the parameter's whole
        life. A const's literal is fixed at construction, and a scalar or
        field set() writes through the very storage the view already reads, so
        neither can stale it. Only destroy() drops the view, having released
        that storage - and that does not reach kernels compiled earlier, which
        still hold it (see parameter.py, "Lifetime of a compiled object").

        """
        if self._device_view is None:
            self._device_view = self._build_device_view()
        return self._device_view

    def _build_device_view(self) -> ClosureParamDeviceView:
        """
        Compile this parameter's device accessors as backend funcs.

        get(node) branches on the mode through `_backend.static`, which
        resolves at trace time, so only one arm survives into the generated
        code: a baked literal for const, HANDLE[None] for scalar, HANDLE[node]
        for field. set_node is built for scalar and field only. MODE, VALUE and
        HANDLE are ordinary python values spliced in as globals.

        """
        backend = self._backend
        mode = self.mode
        value = self._const_value
        handle = self._handle.array if self._handle is not None else None

        def get_template(node):
            if STATIC(MODE == "const"):
                return VALUE
            elif STATIC(MODE == "scalar"):
                return HANDLE[None]
            else:
                return HANDLE[node]

        get_fn = backend.func(
            specialize_closure(get_template, {"MODE": mode, "VALUE": value, "HANDLE": handle, "STATIC": backend.static})
        )

        set_fn = None
        if mode != "const":

            def set_node_template(node, val):
                if STATIC(MODE == "scalar"):
                    HANDLE[None] = val
                else:
                    HANDLE[node] = val

            set_fn = backend.func(
                specialize_closure(set_node_template, {"MODE": mode, "HANDLE": handle, "STATIC": backend.static})
            )

        return ClosureParamDeviceView(self.name, get_fn, set_fn)
