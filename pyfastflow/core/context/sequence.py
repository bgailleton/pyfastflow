"""Host-driven schedules of kernels, routines, and host blocks."""

from typing import Any

from ..pool.base import new_uid
from .bound import Address, BindError, _Bound, format_address, parse_address
from .builder import _ShareMixin
from .compile_shared import check_unmet
from .contract import Contract
from .frozen import FrozenError, FrozenHelper, FrozenKernel, Node
from .host_block import BoundHostBlock, FrozenHostBlock
from .routine import BoundRoutine, FrozenRoutine
from .slot import BuildError, SlotGroup


class SequenceBuilderError(BuildError):
    """
    Raised by the SequenceBuilder build phase: a name reused or unknown, an
    attempt to compose an unsupported frozen type, a malformed loop, or a
    mutation after freeze().

    """


class SequenceBuilder(_ShareMixin):
    """Register blocks, then schedule them as steps and host-driven loops."""

    def __init__(self):
        self._uid = new_uid()
        self._composed: dict[str, Any] = {}
        self._order: list[tuple] = []
        self._shared: dict[tuple, tuple] = {}
        self._synthetic: dict[str, Any] = {}
        self._shared_seen: set[tuple] = set()
        self._frozen = False

    def _check_mutable(self) -> None:
        if self._frozen:
            raise FrozenError(
                f"SequenceBuilder(uid={self._uid}) has already been freeze()-ed and is frozen - "
                f"build a new SequenceBuilder instead of reusing this one"
            )

    def _require_registered(self, name: str) -> Any:
        if name not in self._composed:
            raise SequenceBuilderError(f"{name!r} is not registered on this sequence - call add({name!r}, ...) first")
        return self._composed[name]

    def add(self, name: str, frozen: Any) -> "SequenceBuilder":
        """Register a block without adding it to execution order."""
        self._check_mutable()
        if isinstance(frozen, FrozenHelper):
            raise SequenceBuilderError(
                f"add({name!r}, ...): got a FrozenHelper, not a FrozenKernel/FrozenRoutine/"
                f"FrozenHostBlock - a helper has no standalone host-callable form. Compose it "
                f"into a KernelBuilder first."
            )
        if not isinstance(frozen, (FrozenKernel, FrozenRoutine, FrozenHostBlock)):
            raise TypeError(
                f"add({name!r}, ...): expected a FrozenKernel, FrozenRoutine or "
                f"FrozenHostBlock, got {type(frozen).__name__}"
            )
        if name in self._composed:
            raise SequenceBuilderError(f"'{name}' is already registered on this sequence")
        self._composed[name] = frozen
        return self

    def step(self, name: str) -> "SequenceBuilder":
        """Append a registered block to the top-level execution order."""
        self._check_mutable()
        self._require_registered(name)
        self._order.append(("step", name))
        return self

    def loop(self, body, max_times, until: "str | None" = None) -> "SequenceBuilder":
        """
        Append a loop running the composed blocks named in `body`, in order,
        `max_times` times, stopping early once `until` reports True.

        Parameters
        ----------
        body : Sequence[str]
            Names of already-composed blocks, run in order each iteration.
        max_times : int or str
            Trip count, or the name of a composed host block whose
            zero-argument callable is invoked once on entry and coerced
            with `int()`.
        until : str, optional
            Name of a composed host block invoked after each iteration and
            coerced with `bool()`; `None` runs to completion.

        """
        self._check_mutable()
        body = tuple(body)
        if not body:
            raise SequenceBuilderError("loop: body is empty")
        for name in body:
            self._require_registered(name)
        if isinstance(max_times, str):
            frozen = self._require_registered(max_times)
            if not isinstance(frozen, FrozenHostBlock):
                raise TypeError(f"loop: max_times={max_times!r} must name a host block, got {type(frozen).__name__}")
        elif not isinstance(max_times, int):
            raise TypeError("loop: max_times must be an int or the name of a composed host block")
        if until is not None:
            if not isinstance(until, str):
                raise TypeError("loop: until must be None or the name of a composed host block")
            frozen = self._require_registered(until)
            if not isinstance(frozen, FrozenHostBlock):
                raise TypeError(f"loop: until={until!r} must name a host block, got {type(frozen).__name__}")
        self._order.append(("loop", body, max_times, until))
        return self

    def freeze(self) -> "FrozenSequence":
        """
        Close out the build phase and return the resulting FrozenSequence.

        Raises
        ------
        SequenceBuilderError
            No step()/loop() was ever recorded.

        """
        self._check_mutable()
        if not self._order:
            raise SequenceBuilderError("freeze: sequence has no steps - call step()/loop() at least once")
        self._frozen = True
        return FrozenSequence(self._composed, self._order, self._shared, self._synthetic)


class FrozenSequence(Node):
    """
    The frozen result of a SequenceBuilder's freeze(): a `Node` of kind
    "sequence" whose `children` are the block registry ({name: FrozenKernel|
    FrozenRoutine|FrozenHostBlock}) and whose `.order` is the ordered step/loop
    schedule. It has no template/contract/slots of its own. build() is inherited from Node - the
    walk recurses a composed FrozenRoutine into its own steps, so a routine
    composed under `saddlesort` stepping `label`/`sort` reaches
    `saddlesort.label.*`.

    """

    KIND = "sequence"

    def __init__(
        self,
        composed: dict,
        order: list,
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
        return f"FrozenSequence(uid={self._uid}, blocks={sorted(self.children)})"


class BoundSequence(_Bound):
    """
    The bound result of build()-ing a FrozenSequence - bind()/wire()/
    inspect() work exactly as on a BoundKernel (_Bound, bound.py), over the
    sequence's whole `name.*` address space. See the module docstring's
    "Compiling" section for compile().

    """

    def compile(self, backend=None) -> "CompiledSequence":
        """
        Compile every composed block and return the resulting
        CompiledSequence. See the module docstring's "Compiling" section.

        `backend` is a `Backend`, or may be omitted to use the one recorded
        from bound Parameters and data handles.

        """
        self._check_open("compile")
        check_unmet(self)
        be = self._resolve_backend(backend)
        frozen: FrozenSequence = self._frozen
        compiled_blocks: dict[str, Any] = {}
        block_bounds: list[_Bound] = []

        def _compile_name(name: str) -> Any:
            if name in compiled_blocks:
                return compiled_blocks[name]
            child = frozen.children[name]
            child_bound = child.build()
            self.bind_into(child_bound, (name,))
            compiled = child_bound.compile() if isinstance(child, FrozenHostBlock) else child_bound.compile(be)
            compiled_blocks[name] = compiled
            block_bounds.append(child_bound)
            return compiled

        entries: list[_SeqEntry] = []
        for item in frozen.order:
            if item[0] == "step":
                _, name = item
                entries.append(_SeqEntry("run", run=_compile_name(name)))
            else:
                _, body, max_times, until = item
                body_compiled = tuple(_compile_name(n) for n in body)
                mt = max_times if isinstance(max_times, int) else _compile_name(max_times)
                un = None if until is None else _compile_name(until)
                entries.append(_SeqEntry("loop", body=body_compiled, max_times=mt, until=un))

        return CompiledSequence(entries, compiled_blocks, block_bounds)


class _SeqEntry:
    """
    One entry of a CompiledSequence: a "run" entry wraps a single already-
    resolved zero-argument callable; a "loop" entry carries its own compiled
    body, max_times and until (each itself a plain value or a zero-argument
    callable). Not constructed directly outside BoundSequence.compile().

    """

    __slots__ = ("kind", "run", "body", "max_times", "until")

    def __init__(self, kind: str, run: Any = None, body: tuple = (), max_times: Any = None, until: Any = None):
        self.kind = kind
        self.run = run
        self.body = body
        self.max_times = max_times
        self.until = until


class CompiledSequence:
    """
    An immutable, ordered list of resolved blocks and host-evaluated loops,
    ready to run.

    `last_trip_counts` reports how many body iterations each loop entry took
    on the most recent call, in the order the loop entries appear.

    """

    def __init__(self, entries: list, compiled_blocks: dict, block_bounds: "list | None" = None):
        self._entries = entries
        self._compiled_blocks = compiled_blocks
        # the per-block Bounds this sequence built and owns (destroy safety,
        # close() releases their hold on the shared Parameters and handles.
        self._block_bounds = list(block_bounds) if block_bounds else []
        self._closed = False
        self._last_trip_counts: tuple = ()

    @property
    def last_trip_counts(self) -> tuple:
        """Body iterations taken by each loop entry on the most recent call, in entry order."""
        return self._last_trip_counts

    def close(self) -> None:
        """
        Close every composed block's compiled object and the per-block Bounds
        this sequence owns, releasing their hold on the shared Parameters/
        handles. Idempotent. Does not close the caller-owned sequence Bound.

        """
        if self._closed:
            return
        self._closed = True
        for compiled in self._compiled_blocks.values():
            if hasattr(compiled, "close"):
                compiled.close()
        for b in self._block_bounds:
            b.close()

    def swap(self, addr: "Address | str", buf: Any) -> "CompiledSequence":
        """
        Re-point one composed block's DATA address at `buf`.

        Parameters
        ----------
        addr : Address or str
            `name.*`, routed to that block's own compiled `.swap()`.
        buf : Any
            Replacement buffer.

        Raises
        ------
        BindError
            `name` is not composed, or the named block has nothing to swap
            (a compiled host block is a plain callable with no data
            addresses of its own).

        """
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if not a:
            raise BindError("swap: address must not be empty")
        name, local = a[0], a[1:]
        if name not in self._compiled_blocks:
            raise BindError(
                f"swap: {format_address(a)!r} - no such composed block {name!r} "
                f"(blocks: {sorted(self._compiled_blocks)})"
            )
        target = self._compiled_blocks[name]
        if not hasattr(target, "swap"):
            raise BindError(f"swap: {format_address(a)!r} - block {name!r} has no data to swap (it is a host block)")
        target.swap(local, buf)
        return self

    def __call__(self) -> None:
        """Run every entry in order - see the module docstring's cost-model paragraph."""
        trips: list[int] = []
        for entry in self._entries:
            if entry.kind == "loop":
                trips.append(self._run_loop(entry))
            else:
                entry.run()
        self._last_trip_counts = tuple(trips)

    def _run_loop(self, entry: _SeqEntry) -> int:
        """
        Evaluate `max_times` once on entry, run the body that many times,
        evaluating `until` after each iteration and stopping when it returns
        True. Returns the number of iterations actually run.

        """
        max_times = entry.max_times
        times = int(max_times()) if callable(max_times) else int(max_times)
        taken = 0
        for _ in range(max(0, times)):
            for inner in entry.body:
                inner()
            taken += 1
            if entry.until is not None and bool(entry.until()):
                break
        return taken

    def __repr__(self) -> str:
        return f"CompiledSequence(entries={len(self._entries)})"


# Node.build() reads this to mint a BoundSequence (frozen.py); set here since
# BoundSequence is defined in this module alongside FrozenSequence.
FrozenSequence.bound_cls = BoundSequence
