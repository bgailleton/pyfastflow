"""
Backend-agnostic building blocks for describing GPU work once and compiling it
against Taichi, Quadrants or cupy (or any future one).

What this is for
----------------
A physics model rarely needs a new numerical scheme just because one of its
parameters changed shape. Take heat diffusion: the update is identical whether
the diffusion coefficient K is a spatially variable field, a single value the
host retunes between steps, or a constant fixed for the whole run. That choice
matters enormously on a GPU - a compile-time constant costs no memory traffic
and can be folded into the generated code, a field costs a fetch per node - but
it does not change the maths. Boundary conditions and stencils behave the same
way: making a grid periodic alters the neighbour logic, not the scheme built on
top of it.

Writing one kernel per combination is the obvious way to handle this, and it
becomes unmanageable fast. So instead a template reads a Parameter the same way
whatever its mode, and calls a neighbour helper without knowing which topology
implements it. Which mode, and which helper, is settled at compile time - where
it can still turn into a literal or a specialised routine - and the kernel code
never changes.

Jargon
----------
Parameter       One named, typed value. Its `mode` says where the value lives:
                "const" (baked into the generated code, fixed at
                construction), "scalar" (a single device cell, writable) or
                "field" (a device array, one value per node, writable).
KernelBuilder   The recipe for a launchable kernel: PARAM/HELPER/DATA slots
                declared (`.compose()`/`.data()`), a template supplied at
                construction, then frozen (`.freeze()` -> FrozenKernel) and
                bound (`.build()`/`.bind()` -> BoundKernel) before
                `.compile(backend)` emits the real ti.kernel/qd.kernel/CUDA
                __global__.
HelperBuilder   Same recipe shape, for a device-side helper callable only
                from other device code (ti.func, qd.func, CUDA __device__) -
                composed into an enclosing kernel's tree rather than compiled
                standalone.
FrozenGroup     A pure-structure composite with no data of its own (e.g. a
                grid's neighbour/distance helpers) - composed into a kernel's
                tree the same way a FrozenHelper is, reached in-kernel by
                dotted path: ctx.phys.dx.get(i), ctx.ops.neighbour(i).

There is deliberately no stateful context class: a `make_*` factory returns a
FrozenGroup (pure structure) plus, separately, the concrete Parameters a
caller owns and binds itself - see grid/__init__.py's own module docstring
for that structure/data split.

Compiling something
-------------------
A template is written once, generically, with `ctx` as its first parameter -
the tree it composes from, PARAM/HELPER slots reached as `ctx.name.get(i)`/
`ctx.name(...)` - and turned into something callable in three phases: a
KernelBuilder declares its slots and composes children (`.data()`,
`.compose()`), `.freeze()` freezes that recipe into an inert
FrozenKernel, and `.build()` again (this time on the frozen object, via
`.build()`'s own BoundKernel) plus `.bind()` fills every slot with a concrete
Parameter/helper/data buffer before `.compile(backend)` emits the real
ti.kernel/qd.kernel/CUDA source:

    kernel = KernelBuilder(update_height).compose("phys", phys_group).freeze()
    bound = kernel.build()
    bound.bind(("phys", "dx"), dx_p)
    compiled = bound.compile(be)
    compiled()

See core/context/builder.py, frozen.py and bound.py for the three phases in
full.

A helper composed anywhere in a kernel's tree - directly or nested under
another composed group - is specialized once as part of that kernel's own
`.compile()`, against that same compile's bindings; reaching the same
FrozenHelper from two addresses in one kernel still specializes it once
(build-phase-shared via `.share()`) if the two occurrences were unified into
one address to begin with, or twice if they were composed independently -
either way there is no standalone compiled Helper object to hold onto
outside of a particular kernel's compile.

Data at call time, configuration at compile time
------------------------------------------------
Bound objects are injected into the template body and never appear in the call
signature. A compiled Kernel takes exactly the arguments its template declares,
and that is where bulk data travels - the buffers read and written each step.
Everything that *describes* the problem rather than *being* it - grid spacing,
timestep, gravity, which helper implements the neighbour lookup - is bound.

Reading a Parameter in device code is uniform across modes: p.get(node) to
read, p.set_node(node, value) to write.

What a device helper may bind
-----------------------------
A helper binds whatever a kernel binds, in any mode, on every backend.

On Taichi and Quadrants, bound objects reach device code as globals, and a
helper is traced as part of the kernel that calls it, so alpha.get(i) reads
the same inside a helper as it does in the kernel body.

On cupy, every scalar/field Parameter a compilation unit reaches - the
kernel's own bindings plus, recursively, every helper's - is collected into
one module-scope `__constant__` block, uploaded once per compile(). Every
`__global__` and `__device__` function compiled into that module sees the
same block, so a helper reaches a bound Parameter exactly the way its caller
does, with no pointer argument to thread through and no call site to rewrite.
See cupy_backend.py's module docstring for the block's exact shape.

Lifetime of a compiled object
-----------------------------
compile() freezes what it was given: const Parameters are baked in as literals,
scalar and field Parameters as the storage behind their DataHandle. What may
change afterwards follows from that, and splits cleanly along the mode:

  - Writing to a scalar or field Parameter - set(), set_node(), or a device
    write from inside a kernel - *is* visible to every kernel that binds it,
    including ones compiled beforehand, since they all hold that same storage.
    This is the normal way to feed changing data, and it needs no recompile.
  - A const Parameter is immutable: its value is fixed at construction and
    set() raises. To change one, build a new Parameter, replace() it into the
    bag, and recompile whatever bound the old one.
  - destroy() returns storage to the pool, which may hand the same buffer out
    again. Never destroy a Parameter that a live kernel still binds. This one
    is not enforced at runtime.

So a Parameter's build-time identity - name, dtype, mode, const value - is
fixed at construction, and only its device storage is writable. Which is also
the line to design along: if a quantity changes per step, it is scalar or
field and you simply write it; if changing it demands a recompile, const says
so rather than silently missing the kernels already built.

Where things live
-----------------
This module defines Parameter and the modes it may take. The rest of the
scheme described above is split by concern:

  builder.py  KernelBuilder/HelperBuilder/GroupBuilder - declaring slots and
              composing children, the build phase.
  frozen.py   FrozenKernel/FrozenHelper/FrozenGroup - the inert, immutable
              recipe a builder's `.build()` produces.
  bound.py    BoundKernel - filling a frozen recipe's slots with concrete
              Parameters/helpers/data before compile().
  compile_closure.py / compile_cupy.py / compile_shared.py
              backend-specific and shared compile-phase logic, turning a
              BoundKernel into the real callable.
  bag.py      Bag and its operators (merge, extract, trim, replace, ...), a
              lighter-weight named-collection convenience some factories
              still return alongside the above, unrelated to compilation.

Author: B.G (07/2026)
"""

from abc import ABC, abstractmethod
from typing import Any

from ..pool.base import new_uid

MODES = ("const", "scalar", "field")
"""The storage kinds a Parameter's `mode` may take, common to every backend."""


class Parameter(ABC):
    """
    One named, typed value owned by a context.

    `mode` decides where the value lives - "const" in the generated code,
    "scalar" in a single device cell, "field" in a device array - and every
    backend offers all three (see MODES).

    Two surfaces. From the host: `.value` for const, `.handle()` for stored
    values, plus `set(value)` and `read()`.
    From device code: device_view(), which returns a backend object whose
    .get(node) / .set_node(node, val) let a kernel read and write the
    parameter identically whatever its mode.

    Author: B.G (07/2026)
    """

    name: str
    dtype: Any

    def __init__(self):
        """
        Assign this parameter's process-wide uid and open its `mode` slot.
        Concrete backends call this first, then set `self.mode = ...` once as
        part of their own __init__ - see the `mode` property below.

        Author: B.G (07/2026)
        """
        self._uid = new_uid()
        self._mode: str | None = None
        # how many bound objects (compiled or not) currently hold this
        # Parameter - _Bound.bind()/close() and compiled swap() keep it. destroy()
        # refuses while non-zero. See the module docstring, "Lifetime".
        self._bound_by = 0

    @property
    def uid(self) -> int:
        """
        Process-wide identity assigned at construction, from the same counter
        as every other Parameter, Bag, Helper and pool data handle. Two
        references to one Parameter share a uid; two different Parameters
        never do, even if they hold equal values. Not stable across processes
        and never meant to appear in generated code or a cache key - see the
        module docstring, "uid vs handle".

        Author: B.G (07/2026)
        """
        return self._uid

    _BACKEND_NAME: str = None  # set by each concrete subclass

    @property
    def backend(self):
        """
        The `Backend` this Parameter belongs to (backends.py). `_Bound.bind()`
        reads it to keep one bound object's Parameters/handles on a single
        backend. Imported lazily to avoid a Parameter <-> Backend import cycle.

        Author: B.G (09/2026)
        """
        from .backends import Backend

        return Backend.from_name(self._BACKEND_NAME)

    @property
    def mode(self) -> str:
        """
        Where the value lives - "const", "scalar" or "field". Set once, by
        the backend's __init__; reassigning it raises. To change a
        parameter's mode, construct a new Parameter and swap it into the bag
        in place of this one.

        Author: B.G (07/2026)
        """
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if self._mode is not None:
            raise AttributeError(
                f"{getattr(self, 'name', '?')}: Parameter.mode is immutable once set (already "
                f"{self._mode!r}); construct a new Parameter and swap it into the bag instead"
            )
        self._mode = value

    @abstractmethod
    def _host_value(self):
        """
        Internal host representation: a python scalar for const mode, a
        DataHandle for scalar/field. The public surface is `value` for const
        and `handle()` for stored parameters.

        Author: B.G (07/2026)
        """
        ...

    def handle(self):
        """
        This parameter's DataHandle (scalar/field storage). Raises
        ParameterError for a const parameter, which has no device storage - its
        literal is the `value` property.

        Author: B.G (09/2026)
        """
        from .errors import ParameterError

        if self.mode == "const":
            raise ParameterError(f"{self.name}: const has no handle() - it is a baked-in literal; use .value")
        return self._host_value()

    @property
    def value(self):
        """
        A const parameter's python literal. Raises ParameterError for
        scalar/field, whose value lives in device storage - use handle()/read().

        Author: B.G (09/2026)
        """
        from .errors import ParameterError

        if self.mode != "const":
            raise ParameterError(f"{self.name}: .value is const-only; {self.mode!r} lives in storage - use handle()/read()")
        return self._host_value()

    def _assert_unbound(self, action: str) -> None:
        """Raise ParameterError if a bound object still holds this Parameter - guards destroy(). See _bound_by."""
        from .errors import ParameterError

        if self._bound_by > 0:
            raise ParameterError(
                f"{self.name}: cannot {action} while still bound by {self._bound_by} object(s) - "
                f"close() every Bound/compiled object holding it first"
            )

    @abstractmethod
    def set(self, value) -> None:
        """
        Update the whole parameter value in place, according to its mode: one
        device cell for scalar, a full host->device copy for field. The write
        lands in storage every kernel binding this parameter already reads, so
        no recompile is needed.

        const mode raises: its value is fixed at construction. Build a new
        Parameter, replace() it into the bag and recompile - see the module
        docstring, "Lifetime of a compiled object".

        Author: B.G (07/2026)
        """
        ...

    def set_node(self, node, value) -> None:
        """
        Host-side single-cell write. scalar ignores node; const is read-only.
        Overridden by concrete backends; device-side writes go through
        device_view().set_node instead.

        Author: B.G (07/2026)
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement host set_node")

    def device_view(self):
        """
        An object whose .get(node) / .set_node(node, val) work inside device
        code. Taichi and Quadrants compile one out of ti/qd funcs. cupy leaves
        this unimplemented, having no use for it: its parser substitutes
        parameters into the source directly.

        Author: B.G (07/2026)
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

        Author: B.G (07/2026)
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement read")

    @abstractmethod
    def destroy(self) -> None:
        """
        Release any backing storage owned by this parameter. Unsafe while a
        compiled kernel still binds it - see the module docstring, "Lifetime
        of a compiled object".

        Author: B.G (07/2026)
        """
        ...
