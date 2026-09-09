"""
Shared DataHandle implementation for FieldsBuilder-based backends.

Taichi and Quadrants expose an identical field/FieldsBuilder API; subclasses
only pin `_backend` to their module (ti or qd).

Known cost - Taichi offline cache: each handle finalizes its own
FieldsBuilder/SNode tree, and Taichi's kernel cache key includes field
identity, so churning pooled buffers makes Taichi recompile textually
identical kernels. No fix here; revisit only if Taichi cold-start compile
time becomes a real problem. The cupy path has no equivalent issue (its
constant-block machinery was made field-identity-robust).

Author: B.G (07/2026)
"""

from typing import Any, ClassVar

from .base import DataHandle, new_uid


class FieldsBuilderDataHandle(DataHandle):
    """
    DataHandle backed by one field allocated via FieldsBuilder.

    Composition, not inheritance: kernels take the raw field via `.array`,
    not the handle itself - see pool/base.py design notes on why
    subclassing a field type was rejected.

    Author: B.G (07/2026)
    """

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
        """
        Allocate a field of the given dtype/shape via FieldsBuilder.

        shape=() allocates a 0D scalar field, indexed as field[None].

        Author: B.G (07/2026)
        """
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
        """
        Return the underlying field, for passing straight into kernels or
        binding as a global.

        Author: B.G (07/2026)
        """
        return self._field

    def acquire(self) -> None:
        self.in_use = True

    def release(self) -> None:
        self._assert_unbound("release")
        self.in_use = False

    def destroy(self) -> None:
        """
        Free the field's GPU memory. Unusable afterwards. Raises PoolError while
        a bound object still holds this handle directly (see _assert_unbound).

        Author: B.G (07/2026)
        """
        self._assert_unbound("destroy")
        if self._snodetree is not None:
            self._snodetree.destroy()
            self._snodetree = None

    def to_numpy(self):
        return self._field.to_numpy()

    def from_numpy(self, arr) -> None:
        self._field.from_numpy(arr)
