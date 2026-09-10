"""Named PARAM, DATA, and HELPER slots used by builders."""

from enum import Enum
from typing import Iterator

from .errors import PyFastFlowError


class SlotKind(Enum):
    """Kinds of value a builder can declare."""

    PARAM = "param"
    HELPER = "helper"
    DATA = "data"


class BuildError(PyFastFlowError):
    """
    Base of every build-phase exception across the core - slot-namespace
    misuse, contract derivation/checking, frozen-object mutation, and the
    RoutineBuilder/SequenceBuilder build phases all raise a subclass of this.
    Catch `BuildError` to catch any build-phase mistake regardless of which
    level raised it. Genuine `TypeError`s (e.g. compose() handed the wrong
    frozen type) are not build-phase errors and stay `TypeError`.

    """


class SlotGroupError(BuildError):
    """
    Raised when a builder's local slot namespace is misused - wiring a name
    twice (as a slot or as a compose() root), or looking up a name that was
    never wired.

    """


class ProgramBuilderError(BuildError):
    """
    Raised by the ProgramBuilder build phase (program.py): a name reused
    across the program's one flat namespace (dim/param/data/sequence), a name
    that is not a valid python identifier, a shape referencing an undeclared
    dim, or a malformed spec.

    """


class ProgramError(PyFastFlowError):
    """
    Raised at program run time (program.py), after build: feeding a wrong
    dtype or shape, a conflicting dim binding, touching a value before its
    shapes have resolved, writing a const, or attaching a pool whose backend
    does not match.

    Not a BuildError - these are use-phase mistakes on a built program, not
    recipe-construction mistakes.

    """


class Slot:
    """
    One named place of a given SlotKind, local to the builder that declared
    it.

    """

    __slots__ = ("name", "kind")

    def __init__(self, name: str, kind: SlotKind):
        self.name = name
        self.kind = kind

    def __repr__(self) -> str:
        return f"Slot({self.name!r}, kind={self.kind.value})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Slot) and self.name == other.name and self.kind is other.kind

    def __hash__(self) -> int:
        return hash((self.name, self.kind))


class ParamSlot(Slot):
    """A PARAM slot."""

    __slots__ = ()

    def __init__(self, name: str):
        super().__init__(name, SlotKind.PARAM)


class HelperSlot(Slot):
    """A HELPER slot."""

    __slots__ = ()

    def __init__(self, name: str):
        super().__init__(name, SlotKind.HELPER)


class DataSlot(Slot):
    """A DATA slot, optionally constrained to one dtype."""

    __slots__ = ("dtype",)

    def __init__(self, name: str, dtype=None):
        super().__init__(name, SlotKind.DATA)
        self.dtype = dtype

    def __repr__(self) -> str:
        if self.dtype is None:
            return f"Slot({self.name!r}, kind=data)"
        return f"Slot({self.name!r}, kind=data, dtype={self.dtype})"


class SlotGroup:
    """The flat ``{name: Slot}`` namespace of one builder."""

    def __init__(self):
        self._slots: dict[str, Slot] = {}

    def add(self, slot: Slot) -> None:
        """
        Register ``slot`` under its name, rejecting duplicates.

        """
        if slot.name in self._slots:
            raise SlotGroupError(
                f"'{slot.name}' is already wired on this builder "
                f"(as {self._slots[slot.name]!r})"
            )
        self._slots[slot.name] = slot

    def __contains__(self, name: str) -> bool:
        return name in self._slots

    def __getitem__(self, name: str) -> Slot:
        return self._slots[name]

    def __iter__(self) -> Iterator[Slot]:
        return iter(self._slots.values())

    def __len__(self) -> int:
        return len(self._slots)

    def names(self, kind: SlotKind | None = None) -> set[str]:
        """
        Every wired name, or just those of one `kind` if given.

        """
        if kind is None:
            return set(self._slots)
        return {name for name, slot in self._slots.items() if slot.kind is kind}

    def copy(self) -> "SlotGroup":
        """
        A fresh SlotGroup holding the same Slot objects (Slot is itself
        immutable data, so nothing needs a deeper copy).

        """
        new = SlotGroup()
        new._slots = dict(self._slots)
        return new

    def __repr__(self) -> str:
        if not self._slots:
            return "SlotGroup()"
        body = ", ".join(repr(s) for s in self._slots.values())
        return f"SlotGroup({body})"
