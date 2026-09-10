"""Backend selection and backend-specific runtime wiring."""

from typing import Any

import numpy as np

from ..pool.base import DataHandle, new_uid

_NAMES = ("taichi", "quadrants", "cupy")
_NP_DTYPES = {"i32": np.int32, "i64": np.int64, "f32": np.float32, "u8": np.uint8, "u32": np.uint32}


class _ForeignDataHandle(DataHandle):
    """A non-owning :class:`DataHandle` around storage supplied by an embedder.

    It deliberately is not registered with a Pool: releasing an adopted buffer
    only drops this wrapper, while the caller retains ownership of the actual
    device allocation.
    """

    def __init__(self, be: "Backend", array, *, owned: bool):
        self._uid = new_uid()
        self._be = be
        self._BACKEND_NAME = be.name
        self._array = array
        self._owned = owned
        self.in_use = True
        self.shape = tuple(array.shape)
        native = getattr(array, "dtype", None)
        if native is None:
            raise TypeError("adopted backend storage must expose dtype and shape")
        for tag, dtype in be.dtypes.items():
            try:
                compatible = native == dtype or np.dtype(native) == np.dtype(be.np_dtypes[tag])
            except TypeError:
                compatible = native == dtype
            if compatible:
                self.dtype = tag
                self.backend_dtype = dtype
                break
        else:
            raise TypeError(f"unsupported adopted dtype {native!r}")

    @property
    def array(self):
        return self._array

    def acquire(self):
        self.in_use = True

    def release(self):
        self._assert_unbound("release")
        self.in_use = False

    def destroy(self):
        self._assert_unbound("destroy")
        # Caller-owned storage must never be freed by Program.close().  Even
        # owned=True is only wrapper ownership until a backend supplies a real
        # foreign-allocation destructor.
        self._array = None
        self.in_use = False

    def to_numpy(self):
        if self._be.name == "cupy":
            import cupy as cp
            return cp.asnumpy(self._array)
        return self._array.to_numpy()

    def from_numpy(self, arr):
        if self._be.name == "cupy":
            import cupy as cp
            self._array[...] = cp.asarray(arr)
        else:
            self._array.from_numpy(arr)


class Backend:
    """
    One backend's wiring, cached per name. Construct
    through `from_name`, never directly.

    """

    _cache: "dict[str, Backend]" = {}

    def __init__(
        self,
        name: str,
        family: str,
        module: Any,
        ParameterCls: type,
        PoolCls: type,
        dtypes: dict,
        np_dtypes: dict,
        tensor: Any,
        bk: Any,
        default_block: int = 256,
    ):
        self.name = name
        self.family = family
        self.module = module
        self.ParameterCls = ParameterCls
        self.PoolCls = PoolCls
        self.dtypes = dtypes
        self.np_dtypes = np_dtypes
        self.tensor = tensor
        self.bk = bk
        self.default_block = default_block

    @classmethod
    def from_name(cls, name) -> "Backend":
        """
        The cached Backend for `name` ("taichi"/"quadrants"/"cupy"). Idempotent:
        a Backend passed in is returned unchanged, so callers can accept either
        a Backend unchanged when one is already supplied.

        """
        if isinstance(name, Backend):
            return name
        if name in cls._cache:
            return cls._cache[name]
        if name not in _NAMES:
            raise ValueError(f"unknown backend {name!r}, expected one of {_NAMES}")
        be = cls._build(name)
        cls._cache[name] = be
        return be

    @staticmethod
    def _build(name: str) -> "Backend":
        if name == "taichi":
            import taichi as ti

            from ..pool.taichi_pool import TaichiPool
            from .bk import make_closure_bk
            from .taichi_backend import TaichiParameter

            return Backend(
                "taichi", "closure", ti, TaichiParameter, TaichiPool,
                {"i32": ti.i32, "i64": ti.i64, "f32": ti.f32, "u8": ti.u8, "u32": ti.u32},
                _NP_DTYPES, ti.template(), make_closure_bk(ti),
            )
        if name == "quadrants":
            import quadrants as qd

            from ..pool.quadrants_pool import QuadrantsPool
            from .bk import make_closure_bk
            from .quadrants_backend import QuadrantsParameter

            return Backend(
                "quadrants", "closure", qd, QuadrantsParameter, QuadrantsPool,
                {"i32": qd.i32, "i64": qd.i64, "f32": qd.f32, "u8": qd.u8, "u32": qd.u32},
                _NP_DTYPES, qd.Tensor, make_closure_bk(qd),
            )
        # cupy
        from ..pool.cupy_pool import CupyPool
        from .cupy_backend import CupyParameter

        return Backend(
            "cupy", "cupy", None, CupyParameter, CupyPool,
            dict(_NP_DTYPES), _NP_DTYPES, None, None,
        )

    def compile_kernel(self, bound):
        """
        Compile `bound` on this backend - dispatches to compile_closure (with
        this backend's module) or compile_cupy. Launch dimensions belong to the
        frozen kernel's `domain`/`block` declaration.

        """
        if self.family == "closure":
            from . import compile_closure

            return compile_closure.compile_kernel(bound, self.module)
        from . import compile_cupy

        return compile_cupy.compile_kernel(bound)

    def pool(self):
        """A fresh Pool for this backend."""
        return self.PoolCls()

    def wrap(self, array, *, owned: bool = False):
        """
        Wrap caller-owned backend storage in a DataHandle without copying.

        """
        if self.name == "cupy":
            import cupy as cp
            if not isinstance(array, cp.ndarray):
                raise TypeError("Backend('cupy').wrap() requires a cupy.ndarray")
        elif not (hasattr(array, "to_numpy") and hasattr(array, "from_numpy")):
            raise TypeError(
                f"Backend({self.name!r}).wrap() requires a {self.name} field exposing to_numpy/from_numpy"
            )
        return _ForeignDataHandle(self, array, owned=owned)

    def __eq__(self, other) -> bool:
        return isinstance(other, Backend) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"Backend({self.name!r})"


def require_backend(be) -> Backend:
    """Return ``be`` when it is a Backend; feature factories reject strings.

    """
    if not isinstance(be, Backend):
        raise TypeError(f"feature factories require a Backend, got {type(be).__name__}")
    return be
