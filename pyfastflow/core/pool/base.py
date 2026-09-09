"""
Backend-agnostic pool contracts.

Defines the blueprint that every pool backend (Taichi fields, ndarrays,
quadrants, cupy, ...) must implement. No allocation logic here
- this is the interface only.

Author: B.G (07/2026)
"""

import itertools
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from ..context.errors import PyFastFlowError

_uid_counter = itertools.count()


class PoolError(PyFastFlowError):
    """
    Raised on a pool lifecycle misuse: releasing a foreign or already-free
    handle, or clearing a pool that still has handles in use.

    Author: B.G (08/2026)
    """


def new_uid() -> int:
    """
    Return the next value from the process-wide identity counter.

    Every Parameter, Bag, Helper (device-function builder and its compiled
    artifact) and pool DataHandle is assigned one of these at construction,
    exposed as a read-only `uid` property. uids are plain integers drawn from
    this single shared counter - not stable across processes, and
    deliberately so: they identify an object within one running process and
    must never appear in generated code or a cache key.

    Author: B.G (07/2026)
    """
    return next(_uid_counter)


class DataHandle(ABC):
    """
    Opaque handle to one pooled backend resource (a Taichi field, ndarray, ...).

    Owns the acquire/release lifecycle: `release()` returns the handle to its
    pool for reuse without freeing memory; `destroy()` actually frees it.

    Attributes:
        uid: Process-wide identity from the shared counter (new_uid()) - unique
            across every Parameter, Bag, Helper and DataHandle regardless of
            backend. Concrete handles set self._uid in their own __init__.
        dtype: Short dtype tag (``"f32"``, ``"i32"``, ...).
        backend_dtype: Backend-native dtype used only by pool allocation and
            device compilation.
        shape: Resource dimensions. () for a scalar.
        in_use: True between acquire() and release().

    Author: B.G (07/2026)
    """

    dtype: str
    backend_dtype: Any
    shape: tuple[int, ...]
    in_use: bool

    _BACKEND_NAME: str = None  # set by each concrete handle subclass
    # how many bound objects hold this handle directly (a DATA slot bound to the
    # handle itself). bind()/close()/swap() keep it; release()/destroy() refuse
    # while non-zero. A handle owned as a Parameter's storage is not counted
    # here (the Parameter's own _bound_by guards that).
    _bound_by = 0

    def _assert_unbound(self, action: str) -> None:
        """Raise PoolError if a bound object still holds this handle directly - guards release()/destroy(). See _bound_by."""
        if self._bound_by > 0:
            raise PoolError(
                f"handle uid={getattr(self, '_uid', '?')}: cannot {action} while still bound by "
                f"{self._bound_by} object(s) - close() every Bound/compiled object holding it first"
            )

    @property
    def backend(self):
        """
        The `Backend` this handle belongs to (context.backends). Read by
        `_Bound.bind()` to keep one bound object's storage on a single backend.
        Imported lazily to avoid a pool <-> Backend import cycle.

        Author: B.G (09/2026)
        """
        from ..context.backends import Backend

        return Backend.from_name(self._BACKEND_NAME)

    @property
    def uid(self) -> int:
        """
        Process-wide identity assigned at construction. See new_uid().

        Author: B.G (07/2026)
        """
        return self._uid

    @property
    @abstractmethod
    def array(self):
        """
        Return the raw backend object (ti.field, np.ndarray, ...).

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def acquire(self) -> None:
        """
        Mark this handle in_use. Called by the owning pool on checkout.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """
        Mark this handle available for reuse. Backend memory is kept.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def destroy(self) -> None:
        """
        Free the underlying backend memory. Handle is unusable afterwards.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def to_numpy(self):
        """
        Copy the resource out to a numpy array.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def from_numpy(self, arr) -> None:
        """
        Copy a numpy array into the resource in place.

        Author: B.G (07/2026)
        """
        ...


class Pool(ABC):
    """
    Blueprint for a backend-specific pool manager.

    Implementations keep handles bucketed by (dtype, shape) and reuse
    released handles before allocating new ones.

    Author: B.G (07/2026)
    """

    @abstractmethod
    def get_data(self, dtype, shape) -> DataHandle:
        """
        Return an available handle matching (dtype, shape), allocating one if needed.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def release_data(self, handle: DataHandle) -> None:
        """
        Return a handle to the pool for reuse.

        Raises on a handle this pool never minted, and on a double release (a
        handle already marked available) - either would let the same backing
        buffer be handed out to two owners at once.

        Author: B.G (07/2026)
        """
        ...

    @contextmanager
    def data(self, dtype, shape):
        """
        Scoped acquire/release: `with pool.data(dtype, shape) as h:` checks a
        handle out and returns it on block exit, including on exception. The
        primary acquisition API - use `get_data`/`release_data` directly only
        when a handle must outlive the acquiring scope.

        Author: B.G (08/2026)
        """
        handle = self.get_data(dtype, shape)
        try:
            yield handle
        finally:
            self.release_data(handle)

    @abstractmethod
    def clear_unused(self) -> None:
        """
        Destroy and drop all handles currently not in_use.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def clear_all(self, force: bool = False) -> None:
        """
        Destroy and drop every handle.

        Raises if any handle is still `in_use` unless `force=True` is passed -
        destroying a live handle leaves its holder with a dangling device
        resource.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def stats(self) -> dict:
        """
        Return {"total", "in_use", "available"} handle counts.

        Author: B.G (07/2026)
        """
        ...
