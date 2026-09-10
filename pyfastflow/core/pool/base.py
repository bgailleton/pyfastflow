"""Interfaces for reusable backend arrays."""

import itertools
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from ..context.errors import PyFastFlowError

_uid_counter = itertools.count()


class PoolError(PyFastFlowError):
    """Raised when a pool or handle is used outside its lifetime."""


def new_uid() -> int:
    """Return a new process-local object identity."""
    return next(_uid_counter)


class DataHandle(ABC):
    """A checked-out backend buffer owned by a Pool.

    Release returns its memory to the pool for reuse; destroy discards that
    memory. A handle bound as ``DATA`` cannot be released or destroyed until
    every bound or compiled object using it has been closed.
    """

    dtype: str
    backend_dtype: Any
    shape: tuple[int, ...]
    in_use: bool

    _BACKEND_NAME: str = None
    # Number of bound or compiled objects that hold this handle as DATA.
    _bound_by = 0

    def _assert_unbound(self, action: str) -> None:
        """Reject a lifecycle change while this handle is bound as DATA."""
        if self._bound_by > 0:
            raise PoolError(
                f"handle uid={getattr(self, '_uid', '?')}: cannot {action} while still bound by "
                f"{self._bound_by} object(s) - close() every Bound/compiled object holding it first"
            )

    @property
    def backend(self):
        """Backend that owns this handle."""
        from ..context.backends import Backend

        return Backend.from_name(self._BACKEND_NAME)

    @property
    def uid(self) -> int:
        """Process-local identity of this handle."""
        return self._uid

    @property
    @abstractmethod
    def array(self):
        """Underlying backend array."""
        ...

    @abstractmethod
    def acquire(self) -> None:
        """Mark this handle as checked out."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Mark this handle available for reuse without freeing memory."""
        ...

    @abstractmethod
    def destroy(self) -> None:
        """Discard the underlying backend memory."""
        ...

    @abstractmethod
    def to_numpy(self):
        """Copy this buffer to a NumPy array."""
        ...

    @abstractmethod
    def from_numpy(self, arr) -> None:
        """Copy a NumPy array into this buffer."""
        ...


class Pool(ABC):
    """Manager for reusable backend buffers.

    A pool returns a matching available handle or allocates one when needed.
    Released handles retain their memory and can be checked out again.
    """

    @abstractmethod
    def get_data(self, dtype, shape) -> DataHandle:
        """Check out a buffer of the requested dtype and shape."""
        ...

    @abstractmethod
    def release_data(self, handle: DataHandle) -> None:
        """Return a checked-out handle to this pool for reuse."""
        ...

    @contextmanager
    def data(self, dtype, shape):
        """Yield a checked-out handle and return it on exit.

        This is the usual acquisition API. Use ``get_data`` only when a
        handle must outlive the surrounding scope.
        """
        handle = self.get_data(dtype, shape)
        try:
            yield handle
        finally:
            self.release_data(handle)

    @abstractmethod
    def clear_unused(self) -> None:
        """Destroy every available handle in this pool."""
        ...

    @abstractmethod
    def clear_all(self, force: bool = False) -> None:
        """Destroy every handle, rejecting checked-out ones unless forced."""
        ...

    @abstractmethod
    def stats(self) -> dict:
        """Return counts for total, checked-out, and available handles."""
        ...
