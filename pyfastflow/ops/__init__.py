"""
Small device-op factories built on the builder/frozen/bound stack
(../core/context/builder.py, frozen.py, bound.py; see parameter.py for
Parameter) - bit packing, missing math, generic flat-buffer kernels, a
grid-aware slope helper, a parallel inclusive scan / stream compaction,
plain reductions, and (cupy only) a cub::BlockReduce wrapper.

Four shapes, mirroring grid/noise/visu's own split
-------------------------------------------------------
Helper-group factories (`make_bitpack_group`, `make_math_group`,
`make_slope_group`, `make_block_reduce_group`) return a FrozenGroup composed
by name into whatever kernel/helper needs it (`kb.compose("bitpack",
bitpack_group)`, `ctx.bitpack.pack(f, i)` in a template) - the same
two-call structure/data split grid/__init__.py's own docstring explains,
just with an empty data half for every one of these (none of the four owns
any Parameter). `make_elementwise` is the kernel-builder shape: a
dict[str, FrozenKernel], unbuilt - a caller `.build()`s the member it wants,
binds its DATA addresses, `.compile()`s. `make_scan`/`make_reduce` are
host-orchestrated: each returns a plain python object (Scan/Reduce)
wrapping one or more compiled kernels/routines plus device-side 0-d scalar
Parameters for their results - not representable as a single composable
structure, since calling them runs a short host-side sequence. Unlike the
helper-group factories, `make_scan`/`make_reduce` DO take a `pool` and hand
Parameters back to the caller, unlike `flow/`'s factories (which take no
pool at all): a scan/reduce result is a value the caller consumes, not
scratch state the caller already owns.

`make_slope_group` is this pass's first nested-FrozenGroup-in-FrozenGroup
case within `ops/` itself (mirroring visu's own hillshade gradient blocks,
../visu/__init__.py): `sumslope_downstream`/`slope_dir` each independently
compose the caller's own grid FrozenGroup as a child (a device template can
only reach what is composed directly onto its own scope), and `make_slope_
group` promotes every one of grid's own top-level PARAM names to its own top
level, build-phase-shared into both nested occurrences via `_share_leaf` -
identical mechanism to grid/visu's own, copied rather than imported for the
same reason those two modules copy it from each other (an explicit,
itemized, per-factory declaration, never name-matching across independently
authored composites).

`make_block_reduce_group` is cupy-only (raises otherwise) - Taichi/Quadrants
have no block-level reduction primitive this wraps.

`make_scan`'s returned Parameter (`Scan.count_param`) is handed back bare,
not wrapped in anything - there is no Need-shaped wrapper in this stack to
reach for; a caller wanting it bound into its own KernelBuilder does so
exactly like any other Parameter (`kb.param("count"); bound.bind(
"count", scan.count_param)`). `make_reduce`'s returned handles
(`Reduce.{sum,min,max,argmin}_data`) are bare DataHandles instead, not
Parameters - see "Reduce's DATA-not-PARAM accumulator" below for why a
Parameter wrapper has nothing to add there.

Reduce's DATA-not-PARAM accumulator
-------------------------------------
PARAM access in this stack is strict get()/set_node() (compile_shared.
check_legal_accessors) - a plain, non-atomic single write, never an atomic
accumulate. sum/min/max/argmin genuinely need atomic accumulation across
threads (`ctx.bk.atomic_min(acc[None], x[i])`, Taichi's automatic atomic `+=`
for sum), which needs the raw backend field/ndarray itself, not a Parameter's
device_view - so the running total is wired as a DATA slot on the kernel
(`acc`), never a PARAM one, and is never reached through a PARAM slot
anywhere in this factory's own kernels. A Parameter's only two jobs beyond
raw storage - a device_view() for PARAM-slot binding, and dtype/mode
bookkeeping for that binding - are therefore both unused here, so
`make_reduce` hands its own caller the pooled DataHandle directly
(`sum_data`, ...) rather than a Parameter wrapping it: `.from_numpy()`/
`.to_numpy()` cover the host write/read this module itself needs, and a
caller wanting to chain a result into another kernel binds the handle
straight into that kernel's own DATA slot, no Parameter indirection either
side. See _closure_blocks.py's own module docstring for the fuller reasoning
on why atomic accumulation forces DATA over PARAM in the first place.

Scan's cupy compaction
-----------------------
`.inclusive()` on cupy stays `cp.cumsum` (CUB's own DeviceScan is already
the accelerator cupy dispatches to by default on this build). Compaction's
count-read and scatter are a 2-step FrozenRoutine (routine.py): "read_count"
declares a one-thread domain and "scatter" declares an `n`-thread domain; see
_cupy_blocks.build_count_and_scatter_routine's own docstring.

One consequence: scatter always launches on cupy, even when `count == 0`
(Taichi/Quadrants' `compact_fn` short-circuits before its own scatter
launch, since those two kernels are not bundled into one Routine call and
can be skipped independently) - a harmless extra launch in the all-flags-
zero case, not a correctness difference.

Author: B.G (08/2026)
"""

import numpy as np

from ..core import Backend, require_backend


def _blocks_for(be: Backend):
    """
    The private block module implementing one of this package's factories
    for a given backend name: the closure blocks (shared by Taichi and
    Quadrants) or the cupy blocks.

    Author: B.G (08/2026)
    """
    if be.family == "closure":
        from . import _closure_blocks as blocks
    elif be.family == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"ops: unsupported backend family {be.family!r}")
    return blocks


def make_bitpack_group(be: Backend) -> "FrozenGroup":
    """
    pack(f, i) -> i64, unpack_value(p) -> f32, unpack_index(p) -> i32: pack a
    float and an int32 index into one i64 so that an atomic_min over the
    packed value behaves as a lexicographic argmin over (value, index) -
    composed by name into whatever kernel/helper needs it
    (`kb.compose("bitpack", group)`, `ctx.bitpack.pack(f, i)`). No PARAM
    slots - see this module's own docstring.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".

    Returns
    -------
    FrozenGroup

    Author: B.G (08/2026)
    """
    return _blocks_for(require_backend(be)).build_bitpack_group()


def make_math_group(be: Backend) -> "FrozenGroup":
    """
    atan(x) and nextafter(x, y) (f32), filling in for the two functions
    Taichi/Quadrants/CUDA device code has no direct equivalent for - composed
    by name (`ctx.math.atan(x)`, `ctx.math.nextafter(x, y)`). No PARAM slots.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".

    Returns
    -------
    FrozenGroup

    Author: B.G (08/2026)
    """
    return _blocks_for(require_backend(be)).build_math_group()


def make_elementwise(be: Backend, *, n: "int | None" = None) -> dict:
    """
    swap, add_B_to_A, add_B_to_weighted_A, weighted_mean_B_in_A, arange,
    multiply_by_scalar over a flat buffer, as unbuilt FrozenKernels - call
    `.build()` then bind DATA addresses then `.compile()` on the member you
    want. Buffers (and, on cupy, the buffer length `n`, required here since a
    `cp.RawModule` kernel has no auto-ranging equivalent to Taichi/Quadrants'
    `for i in array`) are DATA slots, not bound Parameters.

    `n` is required on cupy (baked into the generated launch-bounds check at
    build time - see _cupy_blocks.build_elementwise) and ignored on Taichi/
    Quadrants (whose `for i in array` ranges over the buffer's own runtime
    shape, needing no baked length at all).

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    n : int, optional
        Buffer length, required on cupy.

    Returns
    -------
    dict
        {"swap": ..., "add_B_to_A": ..., "add_B_to_weighted_A": ...,
        "weighted_mean_B_in_A": ..., "arange": ..., "multiply_by_scalar":
        ...}, each an unbuilt FrozenKernel.

    Raises
    ------
    ValueError
        On cupy, if `n` is not given.

    Author: B.G (08/2026)
    """
    be = require_backend(be)
    blocks = _blocks_for(be)
    if be.family == "cupy":
        if n is None:
            raise ValueError("make_elementwise: cupy requires n (the buffer length)")
        return blocks.build_elementwise(n)
    return blocks.build_elementwise(be.name, be.module)


def make_slope_group(be: Backend, grid: "FrozenGroup") -> "FrozenGroup":
    """
    sumslope_downstream(z, i): sum of (z[i]-z[j])/dx over every downstream
    neighbour of i. slope_dir(z, i, k): the signed slope towards neighbour k,
    0 where there is none. Both walk `grid`'s own neighbour/dx/n_neighbours
    surface (../grid's make_grid_group result), so they follow whatever
    topology/boundary/nodata `grid` was built with - see this module's own
    docstring for the nested-FrozenGroup mechanism involved.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    grid : FrozenGroup
        Grid structure to walk (../grid's make_grid_group result).

    Returns
    -------
    FrozenGroup

    Author: B.G (08/2026)
    """
    return _blocks_for(require_backend(be)).build_slope_group(grid)


def make_block_reduce_group(be: Backend, *, block_size: int = 128) -> "FrozenGroup":
    """
    cupy only: `sum(val)`, one cub::BlockReduce<float, 128>::Sum() per
    calling CUDA block - composed under the name "sum". Raises on Taichi/
    Quadrants, which have no block-level primitive this wraps.

    The first compile that reaches this triggers a one-time jitify header
    cache warm-up for <cub/block/block_reduce.cuh>, roughly two minutes; that
    is expected, not a hang.

    Parameters
    ----------
    backend : str
        Must be "cupy".
    block_size : int, optional
        CUDA block size, default 128.

    Returns
    -------
    FrozenGroup

    Raises
    ------
    ValueError
        If `backend` is not "cupy".

    Author: B.G (08/2026)
    """
    be = require_backend(be)
    if be.family != "cupy":
        raise ValueError(f"make_block_reduce_group: only supported on cupy, got {be.name!r}")
    from . import _cupy_blocks as blocks

    return blocks.build_block_reduce_group(block_size=block_size)


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


class Scan:
    """
    Inclusive prefix-sum and flag-based stream compaction over a fixed-size
    i32 buffer, as a plain python object rather than a Routine/Sequence -
    built once by make_scan for one backend/size, reused for every call.

    `.inclusive(input_handle, output_handle)` fills `output_handle` with the
    inclusive prefix sum of `input_handle`. `.compact(flags_handle,
    ids_handle)` scatters the indices where `flags_handle` is nonzero into
    `ids_handle[0:count]` and returns `count` as a python int.

    `count_param` is a device-side 0-d scalar Parameter holding the count
    from the most recent `.compact()` call, readable by another kernel via
    `.get(0)` with no host sync; `.count()` is the syncing host equivalent.

    Author: B.G (08/2026)
    """

    def __init__(self, inclusive_fn, compact_fn, count_param):
        self._inclusive_fn = inclusive_fn
        self._compact_fn = compact_fn
        self.count_param = count_param

    def inclusive(self, input_handle, output_handle) -> None:
        """Fill `output_handle` with the inclusive prefix sum of `input_handle`."""
        self._inclusive_fn(input_handle, output_handle)

    def compact(self, flags_handle, ids_handle) -> int:
        """Scatter indices where `flags_handle` is nonzero into `ids_handle[0:count]`; returns count."""
        return self._compact_fn(flags_handle, ids_handle)

    def count(self) -> int:
        """Host-syncing read of the most recent `.compact()` count."""
        return int(self.count_param.read())


def make_scan(be: Backend, pool, n: int) -> Scan:
    """
    Build one Scan over i32 buffers of length `n`.

    Taichi/Quadrants: `.inclusive` drives an internal Blelloch FrozenRoutine
    over a `next_pow2(n)` work buffer, allocated once here - see
    _closure_blocks.build_scan_routine. cupy: `.inclusive` is `cp.cumsum`;
    `.compact`'s count-read/scatter are a 2-step FrozenRoutine - see this
    module's own docstring and _cupy_blocks.build_count_and_scatter_routine.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    pool : Pool
        Device-buffer pool backing scratch storage.
    n : int
        Buffer length.

    Returns
    -------
    Scan

    Author: B.G (08/2026)
    """
    be = require_backend(be)
    ParamCls, dtypes = be.ParameterCls, be.dtypes
    blocks = _blocks_for(be)

    if be.family == "cupy":
        import cupy as cp

        scan_out_h = pool.get_data(np.int32, (n,))
        count_p = ParamCls("SCAN_COUNT", dtype="i32", mode="scalar", value=0, pool=pool)

        routine_frozen = blocks.build_count_and_scatter_routine(n)
        bound = routine_frozen.build()
        bound.bind("read_count.scan_out", scan_out_h)
        bound.bind("read_count.COUNT", count_p)
        bound.bind("scatter.flags", scan_out_h)  # placeholder, swapped every .compact() call
        bound.bind("scatter.scan_out", scan_out_h)
        bound.bind("scatter.ids", scan_out_h)  # placeholder, swapped every .compact() call
        compiled_routine = bound.compile(be)

        def inclusive_fn(input_handle, output_handle):
            cp.cumsum(input_handle.array, out=output_handle.array)

        def compact_fn(flags_handle, ids_handle):
            cp.cumsum(flags_handle.array, out=scan_out_h.array)
            compiled_routine.swap("scatter.flags", flags_handle)
            compiled_routine.swap("scatter.ids", ids_handle)
            compiled_routine()
            return int(count_p.read())

        return Scan(inclusive_fn, compact_fn, count_p)

    # Taichi / Quadrants
    backend_mod = be.module
    work_size = blocks.next_pow2(n)
    work_h = pool.get_data(dtypes["i32"], (work_size,))
    scan_out_scratch = pool.get_data(dtypes["i32"], (n,))
    count_p = ParamCls("SCAN_COUNT", dtype="i32", mode="scalar", value=0, pool=pool)

    routine_frozen = blocks.build_scan_routine(be.name, backend_mod, n, work_size)
    bound = routine_frozen.build()
    for name in routine_frozen.order:
        bound.bind(f"{name}.work", work_h)
    bound.bind("copy_in.src", work_h)  # placeholder, swapped every call
    bound.bind("inclusive_copy.inp", work_h)  # placeholder, swapped every call
    bound.bind("inclusive_copy.out", work_h)  # placeholder, swapped every call
    compiled_routine = bound.compile(be)

    read_count_frozen, scatter_frozen = blocks.build_count_and_scatter_kernels(be.name, backend_mod, n)
    read_count_bound = read_count_frozen.build()
    read_count_bound.bind("scan_out", scan_out_scratch)
    read_count_bound.bind("COUNT", count_p)
    read_count_compiled = read_count_bound.compile(be)

    scatter_bound = scatter_frozen.build()
    scatter_bound.bind("flags", scan_out_scratch)  # placeholder, swapped every .compact() call
    scatter_bound.bind("scan_out", scan_out_scratch)
    scatter_bound.bind("ids", scan_out_scratch)  # placeholder, swapped every .compact() call
    scatter_compiled = scatter_bound.compile(be)

    def inclusive_fn(input_handle, output_handle):
        compiled_routine.swap("copy_in.src", input_handle)
        compiled_routine.swap("inclusive_copy.inp", input_handle)
        compiled_routine.swap("inclusive_copy.out", output_handle)
        compiled_routine()

    def compact_fn(flags_handle, ids_handle):
        compiled_routine.swap("copy_in.src", flags_handle)
        compiled_routine.swap("inclusive_copy.inp", flags_handle)
        compiled_routine.swap("inclusive_copy.out", scan_out_scratch)
        compiled_routine()
        read_count_compiled()
        count = int(count_p.read())
        if count <= 0:
            return 0
        scatter_compiled.swap("flags", flags_handle)
        scatter_compiled.swap("ids", ids_handle)
        scatter_compiled()
        return count

    return Scan(inclusive_fn, compact_fn, count_p)


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------


class Reduce:
    """
    sum/min/max/argmin over a fixed-size f32 buffer, each backed by its own
    device-side 0-d scalar DataHandle plus a syncing host getter - built once
    by make_reduce for one backend/size, reused for every call.

    The four handles this hands back (`sum_data`, ...) are bare pooled
    storage, not Parameters: the accumulator is wired as a DATA slot on the
    underlying kernel (see the module docstring, "Reduce's DATA-not-PARAM
    accumulator"), and a Parameter wrapping storage that is never read
    through a PARAM slot anywhere has nothing left to add over the DataHandle
    itself - see `sum_value`/etc. for the host read, or bind a handle
    directly into another kernel's DATA slot to chain results device-side
    with no host sync.

    Every handle holds the same thing on every backend - `argmin_data` a bare
    i64 index, not the packed (value, index) pair the Taichi/Quadrants
    reduction accumulates into on its way there.

    Author: B.G (08/2026)
    """

    def __init__(self, sum_h, min_h, max_h, argmin_h, run, host):
        self.sum_data = sum_h
        self.min_data = min_h
        self.max_data = max_h
        self.argmin_data = argmin_h
        self._run = run
        self._host = host

    def sum(self, handle) -> None:
        """Accumulate `handle`'s sum into `self.sum_data` (device-side, no host sync)."""
        self._run["sum"](handle)

    def min(self, handle) -> None:
        """Accumulate `handle`'s min into `self.min_data` (device-side, no host sync)."""
        self._run["min"](handle)

    def max(self, handle) -> None:
        """Accumulate `handle`'s max into `self.max_data` (device-side, no host sync)."""
        self._run["max"](handle)

    def argmin(self, handle) -> None:
        """Accumulate `handle`'s argmin into `self.argmin_data` (device-side, no host sync)."""
        self._run["argmin"](handle)

    def sum_value(self) -> float:
        """Host-syncing read of `self.sum_data`."""
        return self._host["sum"]()

    def min_value(self) -> float:
        """Host-syncing read of `self.min_data`."""
        return self._host["min"]()

    def max_value(self) -> float:
        """Host-syncing read of `self.max_data`."""
        return self._host["max"]()

    def argmin_value(self) -> int:
        """Host-syncing read of `self.argmin_data`."""
        return self._host["argmin"]()


def make_reduce(be: Backend, pool, n: int) -> Reduce:
    """
    Build one Reduce over f32 buffers of length `n`.

    cupy: each op is `cp.sum`/`cp.min`/`cp.max`/`cp.argmin`, written into its
    device DataHandle via an async device-to-device copy. Taichi/Quadrants:
    one atomic-accumulate FrozenKernel per op (sum via Taichi/Quadrants'
    automatic atomic `+=`, min/max via `ctx.bk.atomic_min`/`atomic_max`,
    argmin via atomic_min over ops.make_bitpack_group's packed (value, index)
    i64), each preceded by writing the identity element into that op's
    DataHandle from the host. See this module's own docstring for why the
    running total is a DATA argument, not a PARAM slot - and so pooled
    storage handed back bare, not wrapped in a Parameter.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    pool : Pool
        Device-buffer pool backing the accumulator storage.
    n : int
        Buffer length.

    Returns
    -------
    Reduce

    Author: B.G (08/2026)
    """
    be = require_backend(be)
    dtypes = be.dtypes

    if be.family == "cupy":
        import cupy as cp

        sum_h = pool.get_data(dtypes["f32"], ())
        min_h = pool.get_data(dtypes["f32"], ())
        max_h = pool.get_data(dtypes["f32"], ())
        argmin_h = pool.get_data(np.int64, ())

        def run_sum(handle):
            sum_h.array[...] = cp.sum(handle.array)

        def run_min(handle):
            min_h.array[...] = cp.min(handle.array)

        def run_max(handle):
            max_h.array[...] = cp.max(handle.array)

        def run_argmin(handle):
            argmin_h.array[...] = cp.argmin(handle.array).astype(cp.int64)

        run = {"sum": run_sum, "min": run_min, "max": run_max, "argmin": run_argmin}
        host = {
            "sum": lambda: float(sum_h.array.get()),
            "min": lambda: float(min_h.array.get()),
            "max": lambda: float(max_h.array.get()),
            "argmin": lambda: int(argmin_h.array.get()),
        }
        return Reduce(sum_h, min_h, max_h, argmin_h, run, host)

    # Taichi / Quadrants
    backend_mod = be.module
    blocks = _blocks_for(be)

    sum_h = pool.get_data(dtypes["f32"], ())
    min_h = pool.get_data(dtypes["f32"], ())
    max_h = pool.get_data(dtypes["f32"], ())
    argmin_h = pool.get_data(backend_mod.i64, ())
    # internal: the atomic_min accumulator, holding a packed (value, index)
    # pair that argmin_unpack resolves into argmin_h's bare index.
    argmin_packed_h = pool.get_data(backend_mod.i64, ())

    bitpack_group = blocks.build_bitpack_group()
    sum_frozen, min_frozen, max_frozen, argmin_frozen, argmin_unpack_frozen = blocks.build_reduce_kernels(
        be.name, backend_mod, bitpack_group, n
    )

    sum_bound = sum_frozen.build()
    sum_bound.bind("acc", sum_h)
    sum_bound.bind("x", sum_h)  # placeholder, swapped every call
    sum_compiled = sum_bound.compile(be)

    min_bound = min_frozen.build()
    min_bound.bind("acc", min_h)
    min_bound.bind("x", min_h)  # placeholder, swapped every call
    min_compiled = min_bound.compile(be)

    max_bound = max_frozen.build()
    max_bound.bind("acc", max_h)
    max_bound.bind("x", max_h)  # placeholder, swapped every call
    max_compiled = max_bound.compile(be)

    argmin_bound = argmin_frozen.build()
    argmin_bound.bind("acc", argmin_packed_h)
    argmin_bound.bind("x", argmin_packed_h)  # placeholder, swapped every call
    argmin_compiled = argmin_bound.compile(be)

    argmin_unpack_bound = argmin_unpack_frozen.build()
    argmin_unpack_bound.bind("packed_acc", argmin_packed_h)
    argmin_unpack_bound.bind("out", argmin_h)
    argmin_unpack_compiled = argmin_unpack_bound.compile(be)

    _argmin_identity = _closure_pack_identity()

    def run_sum(handle):
        sum_h.from_numpy(np.array(0.0, dtype=np.float32))
        sum_compiled.swap("x", handle)
        sum_compiled()

    def run_min(handle):
        min_h.from_numpy(np.array(float("inf"), dtype=np.float32))
        min_compiled.swap("x", handle)
        min_compiled()

    def run_max(handle):
        max_h.from_numpy(np.array(float("-inf"), dtype=np.float32))
        max_compiled.swap("x", handle)
        max_compiled()

    def run_argmin(handle):
        argmin_packed_h.from_numpy(np.array(_argmin_identity, dtype=np.int64))
        argmin_compiled.swap("x", handle)
        argmin_compiled()
        argmin_unpack_compiled()

    run = {"sum": run_sum, "min": run_min, "max": run_max, "argmin": run_argmin}
    host = {
        "sum": lambda: float(sum_h.to_numpy()),
        "min": lambda: float(min_h.to_numpy()),
        "max": lambda: float(max_h.to_numpy()),
        "argmin": lambda: int(argmin_h.to_numpy()),
    }
    return Reduce(sum_h, min_h, max_h, argmin_h, run, host)


def _closure_pack_identity() -> int:
    """
    pack(+inf, 0), computed host-side with the same bit arithmetic as
    _closure_blocks._pack_tmpl - the identity element atomic_min starts
    argmin's accumulator field at, so any real (value, index) pair wins.

    Author: B.G (07/2026)
    """
    f = np.float32(float("inf"))
    u = int(f.view(np.uint32))
    # +inf has a clear sign bit, so flip_float_bits' positive branch applies:
    # invert every bit.
    f_enc = np.uint32(~np.uint32(u))
    i_enc = np.uint32(0)
    packed = (np.int64(f_enc) << np.int64(32)) | np.int64(i_enc)
    flipped_upper = (~packed) & (np.int64(0xFFFFFFFF) << np.int64(32))
    unchanged_lower = packed & np.int64(0xFFFFFFFF)
    return int(flipped_upper | unchanged_lower)
