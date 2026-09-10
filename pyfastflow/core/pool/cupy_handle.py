"""CuPy implementation of DataHandle."""

from typing import Any

import cupy as cp
import numpy as np

from .base import DataHandle, new_uid


class CupyDataHandle(DataHandle):
    """Handle backed by one CuPy array."""

    _BACKEND_NAME = "cupy"

    @classmethod
    def normalize_dtype(cls, dtype):
        """Return a numpy dtype for a short tag or numpy-compatible dtype."""
        tags = {"i32": np.int32, "i64": np.int64, "f32": np.float32, "u8": np.uint8, "u32": np.uint32}
        if isinstance(dtype, str):
            try:
                return np.dtype(tags[dtype])
            except KeyError as exc:
                raise ValueError(f"unknown dtype tag {dtype!r}") from exc
        return np.dtype(dtype)

    @classmethod
    def short_dtype(cls, dtype) -> str:
        """Return a numpy-compatible dtype's stable public short tag."""
        names = {"int32": "i32", "int64": "i64", "float32": "f32", "uint8": "u8", "uint32": "u32"}
        try:
            return names[np.dtype(dtype).name]
        except KeyError as exc:
            raise ValueError(f"unsupported dtype {dtype!r}") from exc

    def __init__(self, dtype: Any, shape: tuple[int, ...]):
        """Allocate a CuPy array of the requested dtype and shape."""
        self._uid = new_uid()
        self.backend_dtype = self.normalize_dtype(dtype)
        self.dtype = self.short_dtype(self.backend_dtype)
        self.shape = tuple(shape)
        self.in_use = False
        self._array = cp.empty(self.shape, dtype=self.backend_dtype)

    @property
    def array(self):
        """Underlying CuPy array."""
        return self._array

    def acquire(self) -> None:
        self.in_use = True

    def release(self) -> None:
        self._assert_unbound("release")
        self.in_use = False

    def destroy(self) -> None:
        """Release this handle's reference to its CuPy array."""
        self._assert_unbound("destroy")
        self._array = None

    def to_numpy(self):
        return cp.asnumpy(self._array)

    def from_numpy(self, arr) -> None:
        self._array[...] = cp.asarray(arr)
