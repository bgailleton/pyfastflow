"""Shared field-backed DataHandle for Taichi and Quadrants."""

from typing import Any, ClassVar

from .base import DataHandle, new_uid


class FieldsBuilderDataHandle(DataHandle):
    """Handle backed by one field allocated through ``FieldsBuilder``."""

    _backend: ClassVar[Any]

    @classmethod
    def normalize_dtype(cls, dtype):
        """Return this backend's dtype object for a short tag or native dtype."""
        if isinstance(dtype, str):
            try:
                return getattr(cls._backend, dtype)
            except AttributeError as exc:
                raise ValueError(f"unknown dtype tag {dtype!r}") from exc
        return dtype

    @classmethod
    def short_dtype(cls, dtype) -> str:
        """Return this backend dtype's stable public short tag."""
        dtype = cls.normalize_dtype(dtype)
        for tag in ("i32", "i64", "f32", "u8", "u32"):
            if dtype == getattr(cls._backend, tag):
                return tag
        raise ValueError(f"unsupported dtype {dtype!r}")

    def __init__(self, dtype: Any, shape: tuple[int, ...]):
        """Allocate a field; ``shape=()`` creates a scalar field."""
        self._uid = new_uid()
        self.backend_dtype = self.normalize_dtype(dtype)
        self.dtype = self.short_dtype(self.backend_dtype)
        self.shape = tuple(shape)
        self.in_use = False

        backend = self._backend
        self._builder = backend.FieldsBuilder()
        self._field = backend.field(self.backend_dtype)

        if len(self.shape) == 0:
            self._builder.place(self._field)
        elif len(self.shape) == 1:
            self._builder.dense(backend.i, self.shape).place(self._field)
        elif len(self.shape) == 2:
            self._builder.dense(backend.ij, self.shape).place(self._field)
        else:
            raise ValueError(f"Unsupported field dimensionality: {len(self.shape)}D. Only 0D, 1D, 2D supported.")

        self._snodetree = self._builder.finalize()

    @property
    def array(self):
        """Underlying Taichi or Quadrants field."""
        return self._field

    def acquire(self) -> None:
        self.in_use = True

    def release(self) -> None:
        self._assert_unbound("release")
        self.in_use = False

    def destroy(self) -> None:
        """Destroy the underlying field."""
        self._assert_unbound("destroy")
        if self._snodetree is not None:
            self._snodetree.destroy()
            self._snodetree = None

    def to_numpy(self):
        return self._field.to_numpy()

    def from_numpy(self, arr) -> None:
        self._field.from_numpy(arr)
