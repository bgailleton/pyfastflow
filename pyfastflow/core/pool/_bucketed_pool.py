"""
Shared bucketed Pool implementation, parameterized by a DataHandle subclass.

Author: B.G (07/2026)
"""

from typing import Any, ClassVar

from .base import DataHandle, Pool, PoolError


class BucketedPool(Pool):
    """
    Pool manager bucketed by (dtype, shape). Subclasses only pin
    `_handle_cls` to the DataHandle implementation they allocate.

    Author: B.G (07/2026)
    """

    _handle_cls: ClassVar[type]

    def __init__(self):
        self._buckets: dict[tuple[Any, tuple[int, ...]], list[DataHandle]] = {}

    def get_data(self, dtype, shape) -> DataHandle:
        backend_dtype = self._handle_cls.normalize_dtype(dtype)
        key = (backend_dtype, tuple(shape))
        bucket = self._buckets.setdefault(key, [])
        for handle in bucket:
            if not handle.in_use:
                handle.acquire()
                return handle
        handle = self._handle_cls(backend_dtype, key[1])
        handle.acquire()
        bucket.append(handle)
        return handle

    def release_data(self, handle: DataHandle) -> None:
        bucket = self._buckets.get((handle.backend_dtype, tuple(handle.shape)), [])
        if not any(h is handle for h in bucket):
            raise PoolError(
                f"release_data: handle uid={handle.uid} (dtype={handle.dtype}, "
                f"shape={tuple(handle.shape)}) was not minted by this pool"
            )
        if not handle.in_use:
            raise PoolError(
                f"release_data: handle uid={handle.uid} is already available - "
                f"double release would let its buffer be handed out twice"
            )
        handle.release()

    def clear_unused(self) -> None:
        for bucket in self._buckets.values():
            for handle in bucket[:]:
                if not handle.in_use:
                    handle.destroy()
                    bucket.remove(handle)

    def clear_all(self, force: bool = False) -> None:
        if not force:
            in_use = [h.uid for bucket in self._buckets.values() for h in bucket if h.in_use]
            if in_use:
                raise PoolError(
                    f"clear_all: {len(in_use)} handle(s) still in use (uids {in_use}); "
                    f"release them or pass force=True"
                )
        for bucket in self._buckets.values():
            for handle in bucket[:]:
                handle.destroy()
                bucket.remove(handle)

    def stats(self) -> dict:
        total = sum(len(bucket) for bucket in self._buckets.values())
        in_use = sum(1 for bucket in self._buckets.values() for h in bucket if h.in_use)
        return {"total": total, "in_use": in_use, "available": total - in_use}
