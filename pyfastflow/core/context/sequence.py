"""
The host-driven layer above Routine (routine.py): an ordered list of blocks
(kernel, whole routine, host block) plus loops whose trip count and break
are evaluated on the host.

`SequenceBuilder` / `FrozenSequence` / `BoundSequence` / `CompiledSequence`
follow the same build -> freeze -> bind -> compile lifecycle as everything
else in this package.

What this is for
------------------
A Routine is device-only and linear - fixed steps, no python between them,
nothing about repeat count decided at run time. That is the wrong shape for
an outer pass whose trip count is not known until the device has been asked -
depression routing reads a pass count back from the device and either goes
round again or stops. A Sequence runs blocks in order, calls host code
between them, and loops with a host-evaluated predicate; `Parameter.read()`
(parameter.py) underpins every such predicate, and it synchronizes - the
layer's whole cost model, paid at block boundaries, never inside a block.

Composition vs. order
-----------------------
Unlike RoutineBuilder, composing a block here (`compose(name, frozen)`) does
not by itself place it in execution order - a name may be composed once and
then referenced from `step(name)` and/or from inside `loop(...)`'s body more
than once (a callback run once before a loop and again every iteration is
exactly this shape). `step(name)` appends `name` to the top-level order;
`loop(body, max_times, until=None)` appends one loop entry whose `body` is a
sequence of already-composed names, run in order, `max_times` times, stopping
early when `until` returns True.

`max_times`/`until` are each either a plain value (an int for `max_times`,
`None` for `until`, meaning "run to completion") or the *name* of an
already-composed host block (host_block.py) - a FrozenHostBlock, never a bare
python callable, since a name is the only handle this layer's addressing
scheme has to bind that block's own Parameters through. That host block's
compiled, zero-argument callable is invoked once per check; its return value
is coerced with `int()` for `max_times`, `bool()` for `until`.

compose() accepts a FrozenKernel, a FrozenRoutine (routine.py) or a
FrozenHostBlock (host_block.py); a FrozenHelper raises, matching
RoutineBuilder.compose() - a device helper has no standalone host-callable
form on its own.

Addressing
-----------
`build()` is inherited from Node and uses the one walker (bound.py's `walk`),
which recurses `children` under each block's compose() name. A composed
FrozenRoutine is itself a Node whose children are its steps, so the walk
descends one more level through them - a routine composed under `saddlesort`
and internally stepping `label`/`sort` reaches `saddlesort.label.*`/
`saddlesort.sort.*`, exactly the address a standalone Routine's own `build()`
would mint, with the sequence-level compose() name prefixed on top.

Compiling
----------
`BoundSequence.compile(backend)` checks this sequence's own unmet
slots first, then compiles each composed name at most once (cached by name,
since one composed block may be referenced from several places in order): a
fresh bound object from that block's own `.build()`, filled from this
BoundSequence's current values at that block's addresses, then that object's
own `.compile()` - a FrozenHostBlock ignores `backend` (host_block.py),
everything else takes it. The result, CompiledSequence, is an ordered list
of zero-argument callables (blocks already resolved to their own compiled
form) plus loop entries carrying their own body/max_times/until, evaluated
on the host at call time.

`CompiledSequence.swap(addr, buf)` routes `name.*` to the matching compiled
block's own `.swap()` (CompiledKernel.swap / CompiledRoutine.swap); raises if
that block has nothing to swap (a host block has no DATA of its own - see
host_block.py).

Author: B.G (08/2026)
"""

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

    Author: B.G (08/2026)
    """


class SequenceBuilder(_ShareMixin):
    """
    Collects a set of named blocks and an ordered list of steps/loops over
    them, and freeze()s them into a FrozenSequence. `share()`/
    `share_identical()` (from _ShareMixin, builder.py) collapse a bundle shared
    across the sequence's blocks, addressed `block.<...>`. See the module
    docstring.

    Author: B.G (08/2026)
    """

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
        """
        Register `frozen` under `name`, without placing it in execution
        order - see step()/loop() for that, and the module docstring for why
        the two are separate calls here (unlike RoutineBuilder.compose()).

        Parameters
        ----------
        name : str
        frozen : FrozenKernel, FrozenRoutine or FrozenHostBlock
        Author: B.G (08/2026)
        """
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
        """
        Append a top-level step launching the block composed under `name`.

        Author: B.G (08/2026)
        """
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

        Author: B.G (08/2026)
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

        Author: B.G (08/2026)
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
    `saddlesort.label.*`. See the module docstring.

    Author: B.G (09/2026)
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

    Author: B.G (08/2026)
    """

    def compile(self, backend=None) -> "CompiledSequence":
        """
        Compile every composed block and return the resulting
        CompiledSequence. See the module docstring's "Compiling" section.

        `backend` is a `Backend`, or may be omitted to use the one recorded
        from bound Parameters and data handles.

        Author: B.G (09/2026)
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

    Author: B.G (08/2026)
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
    ready to run. See the module docstring.

    `last_trip_counts` reports how many body iterations each loop entry took
    on the most recent call, in the order the loop entries appear.

    Author: B.G (08/2026)
    """

    def __init__(self, entries: list, compiled_blocks: dict, block_bounds: "list | None" = None):
        self._entries = entries
        self._compiled_blocks = compiled_blocks
        # the per-block Bounds this sequence built and owns (destroy safety,
        # Unit 6); close() releases their hold on the shared Parameters/handles.
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

        Author: B.G (09/2026)
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

        Author: B.G (08/2026)
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

        Author: B.G (08/2026)
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
