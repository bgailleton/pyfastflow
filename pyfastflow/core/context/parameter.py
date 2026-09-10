"""Typed model values used by PyFastFlow computations.

A Parameter is a compile-time constant, a writable device scalar, or a
writable device field. Bind it to a PARAM slot; device templates then read it
through the same ``get`` interface regardless of its storage mode.

Use ``value`` for a constant, ``handle()`` for stored values, and ``set`` or
``read`` when host code must update or inspect a value. DATA arrays are
separate DataHandles passed directly to kernels.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..pool.base import new_uid

MODES = ("const", "scalar", "field")
"""The storage kinds a Parameter's `mode` may take, common to every backend."""


class Parameter(ABC):
    """One named, typed model value."""

    name: str
    dtype: Any

    def __init__(self):
        """Initialize the internal identity and storage mode."""
        self._uid = new_uid()
        self._mode: str | None = None
        # Bound and compiled objects retain this Parameter until they close.
        self._bound_by = 0

    @property
    def uid(self) -> int:
        """Process-local identity of this Parameter."""
        return self._uid

    _BACKEND_NAME: str = None  # set by each concrete subclass

    @property
    def backend(self):
        """Backend that owns this Parameter."""
        from .backends import Backend

        return Backend.from_name(self._BACKEND_NAME)

    @property
    def mode(self) -> str:
        """Storage mode: ``const``, ``scalar``, or ``field``."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if self._mode is not None:
            raise AttributeError(
                f"{getattr(self, 'name', '?')}: Parameter.mode is immutable once set (already "
                f"{self._mode!r}); create and bind a new Parameter instead"
            )
        self._mode = value

    @abstractmethod
    def _host_value(self):
        """Return a literal for ``const`` or a DataHandle for stored values."""
        ...

    def handle(self):
        """Return this scalar or field DataHandle."""
        from .errors import ParameterError

        if self.mode == "const":
            raise ParameterError(f"{self.name}: const has no handle() - it is a baked-in literal; use .value")
        return self._host_value()

    @property
    def value(self):
        """Return this constant's Python value."""
        from .errors import ParameterError

        if self.mode != "const":
            raise ParameterError(f"{self.name}: .value is const-only; {self.mode!r} lives in storage - use handle()/read()")
        return self._host_value()

    def _assert_unbound(self, action: str) -> None:
        """Reject destruction while this Parameter remains bound."""
        from .errors import ParameterError

        if self._bound_by > 0:
            raise ParameterError(
                f"{self.name}: cannot {action} while still bound by {self._bound_by} object(s) - "
                f"close() every Bound/compiled object holding it first"
            )

    @abstractmethod
    def set(self, value) -> None:
        """Update a scalar or field value without recompiling its users."""
        ...

    def set_node(self, node, value) -> None:
        """
        Host-side single-cell write. scalar ignores node; const is read-only.
        Overridden by concrete backends; device-side writes go through
        device_view().set_node instead.

        """
        raise NotImplementedError(f"{type(self).__name__} does not implement host set_node")

    def device_view(self):
        """
        An object whose .get(node) / .set_node(node, val) work inside device
        code. Taichi and Quadrants compile one out of ti/qd funcs. cupy leaves
        this unimplemented, having no use for it: its parser substitutes
        parameters into the source directly.

        """
        raise NotImplementedError(f"{type(self).__name__} does not implement device_view")

    def read(self):
        """
        Host-side scalar read, returned as a plain python value regardless of
        mode - unlike get(), which hands back a DataHandle for scalar/field.

        const mode: the stored python value, no device traffic.
        scalar mode: a device->host read that synchronizes. That sync is the
        whole cost model of any host-driven loop built on top of this - call
        it only where a step actually needs the value on the host.
        field mode: raises. Reading a whole field back to the host is not
        what this is for; use device_view()/get() from device code, or copy
        the field explicitly if the host genuinely needs all of it.

        """
        raise NotImplementedError(f"{type(self).__name__} does not implement read")

    @abstractmethod
    def destroy(self) -> None:
        """Release storage owned by this Parameter."""
        ...
