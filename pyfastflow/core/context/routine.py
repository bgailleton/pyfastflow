"""Ordered device-only kernel routines."""

from typing import Any

from ..pool.base import new_uid
from .bound import Address, BindError, _Bound, format_address, parse_address
from .builder import _ShareMixin
from .compile_shared import check_unmet
from .contract import Contract
from .frozen import FrozenError, FrozenHelper, FrozenKernel, Node
from .slot import BuildError, SlotGroup


class RoutineBuilderError(BuildError):
    """
    Raised by the RoutineBuilder build phase: a step name reused, an
    attempt to compose a non-FrozenKernel, or a mutation after freeze().

    """


class RoutineBuilder(_ShareMixin):
    """
    Collects an ordered set of named kernel steps and freeze()s them into a
    FrozenRoutine. `share()`/`share_identical()` (from _ShareMixin, builder.py)
    collapse a bundle - a grid's helpers and their params - shared across the
    routine's steps, addressed `step.<...>`.

    """

    def __init__(self):
        self._uid = new_uid()
        self._order: list[str] = []
        self._composed: dict[str, FrozenKernel] = {}
        self._shared: dict[tuple, tuple] = {}
        self._synthetic: dict[str, Any] = {}
        self._shared_seen: set[tuple] = set()
        self._frozen = False

    def _check_mutable(self) -> None:
        if self._frozen:
            raise FrozenError(
                f"RoutineBuilder(uid={self._uid}) has already been freeze()-ed and is frozen - "
                f"build a new RoutineBuilder instead of reusing this one"
            )

    def step(self, name: str, frozen_kernel: FrozenKernel) -> "RoutineBuilder":
        """
        Append a step named `name`, launching `frozen_kernel` at this
        position in the routine's launch order.

        Parameters
        ----------
        name : str
            Address prefix for this step. Must be unique within the routine.
        frozen_kernel : FrozenKernel
        """
        self._check_mutable()
        if isinstance(frozen_kernel, FrozenHelper):
            raise RoutineBuilderError(
                f"step({name!r}, ...): got a FrozenHelper, not a FrozenKernel - a helper has "
                f"no standalone launch and cannot be a routine step. Compose it into a "
                f"KernelBuilder first, then compose that kernel's FrozenKernel here."
            )
        if not isinstance(frozen_kernel, FrozenKernel):
            raise TypeError(f"step({name!r}, ...): expected a FrozenKernel, got {type(frozen_kernel).__name__}")
        if name in self._composed:
            raise RoutineBuilderError(f"'{name}' is already registered on this routine")
        self._composed[name] = frozen_kernel
        self._order.append(name)
        return self

    def freeze(self) -> "FrozenRoutine":
        """
        Close out the build phase and return the resulting FrozenRoutine.

        Raises
        ------
        RoutineBuilderError
            No step was ever composed - an empty routine has nothing to
            launch.

        """
        self._check_mutable()
        if not self._order:
            raise RoutineBuilderError("freeze: routine has no steps - compose() at least one kernel first")
        self._frozen = True
        return FrozenRoutine(self._order, self._composed, self._shared, self._synthetic)


class FrozenRoutine(Node):
    """
    The frozen result of a RoutineBuilder's freeze(): a `Node` of kind
    "routine" whose `children` are its steps, insertion order = launch order,
    It has no template/contract/slots of its own; `.order` reports its step names in
    launch order. build() is inherited from Node.

    """

    KIND = "routine"

    def __init__(
        self,
        order: list,
        composed: dict,
        shared: "dict | None" = None,
        synthetic: "dict | None" = None,
    ):
        super().__init__(
            template=None,
            slots=SlotGroup(),
            children=composed,
            contract=Contract(frozenset()),
            shared=shared,
            synthetic=synthetic,
            order=tuple(order),
        )
    def __repr__(self) -> str:
        return f"FrozenRoutine(uid={self._uid}, steps={list(self.order)})"


class BoundRoutine(_Bound):
    """
    The bound result of build()-ing a FrozenRoutine - bind()/wire()/
    inspect() work exactly as on a BoundKernel (_Bound, bound.py), over the
    routine's whole `name.*` address space. See the module docstring's
    "Compiling" section for compile().

    """

    def compile(self, backend=None) -> "CompiledRoutine":
        """
        Compile every step and return the resulting CompiledRoutine. See
        the module docstring's "Compiling" section.

        `backend` is a `Backend`, or may be omitted to use the one recorded
        from bound Parameters and data handles.

        """
        self._check_open("compile")
        check_unmet(self)
        be = self._resolve_backend(backend)
        frozen: FrozenRoutine = self._frozen
        steps: list[tuple[str, Any]] = []
        step_bounds: list[_Bound] = []
        for name in frozen.order:
            step_frozen = frozen.children[name]
            step_bound = step_frozen.build()
            self.bind_into(step_bound, (name,))
            compiled = step_bound.compile(be)
            steps.append((name, compiled))
            step_bounds.append(step_bound)
        return CompiledRoutine(steps, step_bounds)


class CompiledRoutine:
    """
    An immutable, ordered sequence of already-compiled kernels, ready to
    launch as one unit.

    """

    def __init__(self, steps: list, step_bounds: "list | None" = None):
        self._steps = list(steps)
        self._by_name = dict(steps)
        # the per-step BoundKernels this routine built and owns (destroy safety,
        # close() releases their hold on the shared Parameters and handles.
        self._step_bounds = list(step_bounds) if step_bounds else []
        self._closed = False

    def close(self) -> None:
        """
        Close every step's compiled kernel and the per-step Bounds this routine
        owns, releasing their hold on the shared Parameters/handles. Idempotent.
        Does not close the caller-owned routine Bound this was compiled from.

        """
        if self._closed:
            return
        self._closed = True
        for _, compiled in self._steps:
            if hasattr(compiled, "close"):
                compiled.close()
        for b in self._step_bounds:
            b.close()

    @property
    def step_names(self) -> list:
        """Step names in launch order."""
        return [name for name, _ in self._steps]

    def swap(self, addr: "Address | str", buf: Any) -> "CompiledRoutine":
        """
        Re-point one step's DATA address at `buf`.

        Parameters
        ----------
        addr : Address or str
            `name.*`, routed to step `name`'s own CompiledKernel.swap().
        buf : Any
            Replacement buffer.

        """
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if not a:
            raise BindError("swap: address must not be empty")
        name, local = a[0], a[1:]
        if name not in self._by_name:
            raise BindError(
                f"swap: {format_address(a)!r} - no such routine step {name!r} "
                f"(steps: {sorted(self._by_name)})"
            )
        self._by_name[name].swap(local, buf)
        return self

    def __call__(self) -> None:
        """Launch every step in order, each with whatever its own swap() state currently holds."""
        for _, compiled in self._steps:
            compiled()

    def __repr__(self) -> str:
        return f"CompiledRoutine(steps={[n for n, _ in self._steps]})"


# Node.build() reads this to mint a BoundRoutine (frozen.py); set here since
# BoundRoutine is defined in this module alongside FrozenRoutine.
FrozenRoutine.bound_cls = BoundRoutine
