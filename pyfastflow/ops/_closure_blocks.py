"""
Taichi/Quadrants (closure) block templates behind ops's make_bitpack_group/
make_scan/make_reduce, on the new builder/frozen/bound stack
(core/context/builder.py, frozen.py, bound.py) - mirrors grid/_closure_blocks.py's
own split (private blocks as plain python defs, public members composed under
an explicit name) and reuses its `_helper` assembly verbatim.

bitpack has no PARAM slots at all - every private block below only ever needs
`ctx.bk` (bit_cast/select/cast/the raw u32/i64/i32/f32 dtype tokens - see
bk.py) plus whatever it composes. `make_bitpack_group` therefore returns a
FrozenGroup with three composed HELPER members (pack/unpack_value/
unpack_index) and zero top-level PARAM slots - composing the whole group into
any kernel or helper mints no new bindable address at all, exactly like
composing a Contract-empty leaf.

Reduce's accumulator (`sum`/`min`/`max`/`argmin`'s running total) is wired as
a DATA slot, not a PARAM one: PARAM access is strict get()/set_node() only
(compile_shared.check_legal_accessors), which is a plain non-atomic
overwrite, not the atomic accumulation a parallel reduction needs across
threads - `ctx.bk.atomic_min(acc[None], x[i])` needs `acc` as the raw backend
field/ndarray `ti.template()`/`qd.Tensor` already gives a DATA argument, the
same reasoning ops/__init__.py's old (pre-rewrite) `_SUM`/`_MIN`/`_MAX`/
`_ARGMIN` raw-field binds already used - just expressed as a DATA slot instead
of a raw, Need-less bind, since a PARAM slot in the new stack always requires
an actual Parameter object (bound.py's bind() checks `isinstance(obj,
Parameter)`) and there is deliberately no atomic accessor on Parameter's
device_view. The Parameter objects `make_reduce` hands back to its own caller
(`sum_param`, ...) still exist and still own that same storage
(`sum_p.handle().array` IS the array bound to the "acc" DATA address) - reduce's
own caller reads them exactly as before; only the device-side wiring differs.

`ctx.bk` supplies `bit_cast`/`select`/`cast`/`atomic_min`/`atomic_max` and the
`u32`/`i64` dtype tokens bitpack/reduce need - not part of the original grid/
noise/visu surface, extended here (see bk.py's own module docstring for why
this is the sanctioned way to add an intrinsic, and the extension itself).

Author: B.G (08/2026)
"""

from ..core import GroupBuilder, RoutineBuilder, freeze_helper as _helper, freeze_kernel as _kernel

# ---------------------------------------------------------------------------
# bitpack: pack(f, i) -> i64, unpack_value(p) -> f32, unpack_index(p) -> i32
# ---------------------------------------------------------------------------


def _flip_float_bits_tmpl(ctx, f):
    u = ctx.bk.bit_cast(f, ctx.bk.u32)
    return ctx.bk.select(u & ctx.bk.u32(0x80000000) != 0, u ^ ctx.bk.u32(0x80000000), ~u)


def _unflip_float_bits_tmpl(ctx, u):
    restored = ctx.bk.select(u & ctx.bk.u32(0x80000000) != 0, ~u, u ^ ctx.bk.u32(0x80000000))
    return ctx.bk.bit_cast(restored, ctx.bk.f32)


def _pack_tmpl(ctx, f, i):
    f_enc = ctx._FLIP(f)
    i_enc = ctx.bk.bit_cast(i, ctx.bk.u32)
    packed = (ctx.bk.cast(f_enc, ctx.bk.i64) << 32) | ctx.bk.cast(i_enc, ctx.bk.i64)
    flipped_upper = (~packed) & (ctx.bk.i64(0xFFFFFFFF) << 32)
    unchanged_lower = packed & ctx.bk.i64(0xFFFFFFFF)
    return flipped_upper | unchanged_lower


def _unpack_raw_tmpl(ctx, packed):
    flipped_upper = (~packed) & (ctx.bk.i64(0xFFFFFFFF) << 32)
    unchanged_lower = packed & ctx.bk.i64(0xFFFFFFFF)
    return flipped_upper | unchanged_lower


def _unpack_value_tmpl(ctx, packed):
    u = ctx._UNPACKRAW(packed)
    f_enc = ctx.bk.cast(u >> 32, ctx.bk.u32)
    return ctx._UNFLIP(f_enc)


def _unpack_index_tmpl(ctx, packed):
    u = ctx._UNPACKRAW(packed)
    i_enc = ctx.bk.cast(u & ctx.bk.i64(0xFFFFFFFF), ctx.bk.u32)
    return ctx.bk.bit_cast(i_enc, ctx.bk.i32)


def build_bitpack_group() -> "FrozenGroup":
    """
    pack(f, i) -> i64, unpack_value(p) -> f32, unpack_index(p) -> i32: the
    IEEE-754 bit-flip trick that makes an i64 atomic_min double as a
    lexicographic argmin over (float, int), composed onto a fresh
    GroupBuilder under those three public names. No PARAM slots anywhere in
    this tree - see the module docstring.

    Author: B.G (08/2026)
    """
    flip = _helper(_flip_float_bits_tmpl)
    unflip = _helper(_unflip_float_bits_tmpl)
    pack = _helper(_pack_tmpl, helpers={"_FLIP": flip})
    unpack_raw = _helper(_unpack_raw_tmpl)
    unpack_value = _helper(_unpack_value_tmpl, helpers={"_UNPACKRAW": unpack_raw, "_UNFLIP": unflip})
    unpack_index = _helper(_unpack_index_tmpl, helpers={"_UNPACKRAW": unpack_raw})

    group = GroupBuilder()
    group.compose("pack", pack)
    group.compose("unpack_value", unpack_value)
    group.compose("unpack_index", unpack_index)
    return group.freeze()


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def _atan_tmpl(ctx, x):
    return ctx.bk.atan2(x, 1.0)


def _nextafter_tmpl(ctx, x, y):
    result = y
    if x != y:
        sign_mask = ctx.bk.bit_cast(ctx.bk.cast(-0.0, ctx.bk.f32), ctx.bk.u32)
        ix = ctx.bk.bit_cast(x, ctx.bk.u32)
        if x == 0.0:
            ix = (ctx.bk.bit_cast(y, ctx.bk.u32) & sign_mask) | ctx.bk.cast(1, ctx.bk.u32)
        elif (x > 0.0) == (y > x):
            ix += ctx.bk.cast(1, ctx.bk.u32)
        else:
            ix -= ctx.bk.cast(1, ctx.bk.u32)
        result = ctx.bk.bit_cast(ix, ctx.bk.f32)
    return result


def build_math_group() -> "FrozenGroup":
    """
    atan(x) via atan2(x, 1); nextafter(x, y), one ULP of f32 towards y via
    IEEE-754 bit-twiddling (no libm nextafter on GPU) - composed onto a fresh
    GroupBuilder under those two public names. No PARAM slots.

    Author: B.G (08/2026)
    """
    atan = _helper(_atan_tmpl)
    nextafter = _helper(_nextafter_tmpl)

    group = GroupBuilder()
    group.compose("atan", atan)
    group.compose("nextafter", nextafter)
    return group.freeze()


# ---------------------------------------------------------------------------
# elementwise (kernels, returned unbuilt)
# ---------------------------------------------------------------------------


def build_elementwise(backend: str, backend_mod) -> dict:
    """
    swap/add_B_to_A/add_B_to_weighted_A/weighted_mean_B_in_A/arange/
    multiply_by_scalar over a flat buffer, as unbuilt FrozenKernels - a
    caller `.build()`s the one it wants, binds data addresses, `.compile()`s.
    Buffers (array1/array2/A/scalar/weight/array) are DATA slots, not bound
    Parameters - see parameter.py, "Data at call time, configuration at
    compile time". `multiply_by_scalar`'s own `scalar` argument is annotated
    `T` (a compile-time template parameter), not `F` (a plain runtime f32) -
    ported unchanged from the pre-rewrite template; the apparent
    inconsistency with `add_B_to_weighted_A`'s `weight: F` already existed
    before this port and is not something this pass changes.

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    F = backend_mod.f32

    def swap_tmpl(ctx, array1: T, array2: T):
        for idx in ctx.bk.grouped(array1):
            temp = array1[idx]
            array1[idx] = array2[idx]
            array2[idx] = temp

    def add_B_to_A_tmpl(ctx, array1: T, array2: T):
        for i in array1:
            array1[i] += array2[i]

    def add_B_to_weighted_A_tmpl(ctx, array1: T, array2: T, weight: F):
        for i in array1:
            array1[i] += array2[i] * weight

    def weighted_mean_B_in_A_tmpl(ctx, array1: T, array2: T, weight: F):
        for i in array1:
            array1[i] = array2[i] * weight + array1[i] * (1 - weight)

    def arange_tmpl(ctx, array: T):
        for i in array:
            array[i] = i

    def multiply_by_scalar_tmpl(ctx, A: T, scalar: T):
        for i in A:
            A[i] *= scalar

    def _mk(template, names):
        return KernelBuilder(template).freeze()

    return {
        "swap": _mk(swap_tmpl, ["array1", "array2"]),
        "add_B_to_A": _mk(add_B_to_A_tmpl, ["array1", "array2"]),
        "add_B_to_weighted_A": _mk(add_B_to_weighted_A_tmpl, ["array1", "array2", "weight"]),
        "weighted_mean_B_in_A": _mk(weighted_mean_B_in_A_tmpl, ["array1", "array2", "weight"]),
        "arange": _mk(arange_tmpl, ["array"]),
        "multiply_by_scalar": _mk(multiply_by_scalar_tmpl, ["A", "scalar"]),
    }


# ---------------------------------------------------------------------------
# slope (grid-aware) - the first ops/ case of a nested FrozenGroup-in-Frozen
# Group child, mirroring visu/__init__.py's hillshade gradient blocks
# ---------------------------------------------------------------------------


def _sumslope_downstream_tmpl(ctx, z, i):
    sumslope = 0.0
    for k in range(ctx.grid.N_NEIGHBOURS.get(0)):
        j = ctx.grid.neighbour(i, k)
        if j > -1:
            if z[j] < z[i]:
                sumslope += (z[i] - z[j]) / ctx.grid.DX.get(0)
    return sumslope


def _slope_dir_tmpl(ctx, z, i, k):
    j = ctx.grid.neighbour(i, k)
    slope = 0.0
    if j > -1:
        slope = (z[i] - z[j]) / ctx.grid.DX.get(0)
    return slope


def build_slope_group(grid) -> "FrozenGroup":
    """
    sumslope_downstream(z, i): sum of (z[i]-z[j])/dx over every downstream
    neighbour of i. slope_dir(z, i, k): the signed slope towards neighbour k,
    0 where there is none. Both walk `grid`'s own neighbour/dx/n_neighbours
    surface, so they follow whatever topology/boundary/nodata `grid` was
    built with - `grid` (a FrozenGroup, ../grid's own make_grid_group result)
    is composed independently as each helper's own child (a device template
    can only reach what is composed directly onto its own scope - builder.py's
    module docstring), the same nested-FrozenGroup-in-FrozenGroup shape
    visu/__init__.py's hillshade gradient blocks establish. Every name in
    `grid`'s own top-level PARAM slots is wired again at this group's own top
    level and declared build-phase-shared with both nested occurrences (see
    the module docstring's build-phase-sharing section, or grid/__init__.py's
    own), so a caller binds e.g. `slope.DX` once rather than once per helper.

    Author: B.G (08/2026)
    """
    sumslope_downstream = HelperBuilder(_sumslope_downstream_tmpl).compose("grid", grid).freeze()
    slope_dir = HelperBuilder(_slope_dir_tmpl).compose("grid", grid).freeze()

    group = GroupBuilder()
    group.compose("sumslope_downstream", sumslope_downstream)
    group.compose("slope_dir", slope_dir)
    group.share_identical("sumslope_downstream.grid", as_="grid")

    return group.freeze()


# ---------------------------------------------------------------------------
# scan (Blelloch inclusive scan + flag compaction), i32 only
# ---------------------------------------------------------------------------


def next_pow2(n: int) -> int:
    """Smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def _tensor_annotation(backend_mod, backend: str):
    """
    The data-argument annotation a kernel template needs on this closure
    backend: `ti.template()` for Taichi, `qd.Tensor` for Quadrants.

    Author: B.G (08/2026)
    """
    return backend_mod.template() if backend == "taichi" else backend_mod.Tensor


def _make_copy_input_to_work_tmpl(T, n, work_size):
    def copy_input_to_work_tmpl(ctx, src: T, work: T):
        for i in range(work_size):
            if i < n:
                work[i] = src[i]
            else:
                work[i] = 0

    return copy_input_to_work_tmpl


def _make_upsweep_tmpl(T, work_size, stride):
    def upsweep_tmpl(ctx, work: T):
        for i in range(work_size):
            if (i + 1) % (stride * 2) == 0:
                work[i] += work[i - stride]

    return upsweep_tmpl


def _make_set_root_zero_tmpl(T, root_index):
    def set_root_zero_tmpl(ctx, work: T):
        work[root_index] = 0

    return set_root_zero_tmpl


def _make_downsweep_tmpl(T, work_size, stride):
    def downsweep_tmpl(ctx, work: T):
        for i in range(work_size):
            if (i + 1) % (stride * 2) == 0:
                temp = work[i - stride]
                work[i - stride] = work[i]
                work[i] += temp

    return downsweep_tmpl


def _make_inclusive_and_copy_tmpl(T, n):
    def inclusive_and_copy_tmpl(ctx, inp: T, work: T, out: T):
        for i in range(n):
            if i == 0:
                out[i] = inp[i]
            else:
                out[i] = work[i] + inp[i]

    return inclusive_and_copy_tmpl


def _make_scatter_tmpl(T, n):
    def scatter_tmpl(ctx, flags: T, scan_out: T, ids: T):
        for i in range(n):
            if flags[i] == 1:
                pos = scan_out[i] - 1
                ids[pos] = i

    return scatter_tmpl


def _make_read_count_tmpl(T, n):
    def read_count_tmpl(ctx, scan_out: T):
        ctx.COUNT.set_node(0, scan_out[n - 1])

    return read_count_tmpl


def build_scan_routine(backend: str, backend_mod, n: int, work_size: int):
    """
    The Blelloch up-sweep/down-sweep FrozenRoutine (routine.py) over a
    size-`work_size` work buffer: copy-in, log2(work_size) up-sweep steps,
    zero the root, log2(work_size) down-sweep steps, convert to inclusive and
    copy out. Every step is a separate composed FrozenKernel under its own
    name (a genuinely different template per step - each up/down-sweep
    closes over its own `stride` - so there is no same-kernel-twice
    instancing to exploit here), and every step is therefore its own kernel
    launch: each step's reads depend on every write the previous one made
    across the whole buffer, so the steps need a real global barrier between
    them, which CompiledRoutine.__call__'s plain launch-in-order already is.

    Returns the FrozenRoutine, unbuilt - the caller (make_scan) builds it
    once, binds "copy_in.src"/"inclusive_copy.inp"/"inclusive_copy.out" per
    call via swap() (the buffers that vary), and binds every "work" address
    once (the internal scratch, fixed for the life of this Scan).

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    rb = RoutineBuilder()
    rb.step("copy_in", _kernel(_make_copy_input_to_work_tmpl(T, n, work_size), data=["src", "work"]))

    stride = 1
    i = 0
    while stride < work_size:
        rb.step(f"upsweep{i}", _kernel(_make_upsweep_tmpl(T, work_size, stride), data=["work"]))
        stride *= 2
        i += 1

    rb.step("zero_root", _kernel(_make_set_root_zero_tmpl(T, work_size - 1), data=["work"]))

    stride = work_size // 2
    j = 0
    while stride > 0:
        rb.step(f"downsweep{j}", _kernel(_make_downsweep_tmpl(T, work_size, stride), data=["work"]))
        stride //= 2
        j += 1

    rb.step("inclusive_copy", _kernel(_make_inclusive_and_copy_tmpl(T, n), data=["inp", "work", "out"]))

    return rb.freeze()


def build_count_and_scatter_kernels(backend: str, backend_mod, n: int):
    """
    read_count(scan_out): writes scan_out[n-1] into a wired PARAM slot
    "COUNT" (a real Parameter - set_node is a plain, non-atomic single write
    from one thread, legal PARAM access). scatter(flags, scan_out, ids): for
    flags[i]==1, ids[scan_out[i]-1] = i - the scatter half of scan-based
    compaction. Returns the two FrozenKernels, unbuilt.

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    read_count = _kernel(_make_read_count_tmpl(T, n), data=["scan_out"], params=["COUNT"])
    scatter = _kernel(_make_scatter_tmpl(T, n), data=["flags", "scan_out", "ids"])
    return read_count, scatter


# ---------------------------------------------------------------------------
# reduce (sum/min/max/argmin), i32 count / f32 value
# ---------------------------------------------------------------------------


def _make_sum_tmpl(T, n):
    def sum_tmpl(ctx, x: T, acc: T):
        for i in range(n):
            acc[None] += x[i]

    return sum_tmpl


def _make_min_tmpl(T, n):
    def min_tmpl(ctx, x: T, acc: T):
        for i in range(n):
            ctx.bk.atomic_min(acc[None], x[i])

    return min_tmpl


def _make_max_tmpl(T, n):
    def max_tmpl(ctx, x: T, acc: T):
        for i in range(n):
            ctx.bk.atomic_max(acc[None], x[i])

    return max_tmpl


def _make_argmin_tmpl(T, n):
    def argmin_tmpl(ctx, x: T, acc: T):
        for i in range(n):
            packed = ctx.bitpack.pack(x[i], i)
            ctx.bk.atomic_min(acc[None], packed)

    return argmin_tmpl


def _make_argmin_unpack_tmpl(T):
    def argmin_unpack_tmpl(ctx, packed_acc: T, out: T):
        for _ in range(1):
            out[None] = ctx.bitpack.unpack_index(packed_acc[None])

    return argmin_unpack_tmpl


def build_reduce_kernels(backend: str, backend_mod, bitpack_group, n: int):
    """
    One FrozenKernel per op (sum/min/max), each accumulating atomically into
    its own "acc" DATA argument - see the module docstring for why this is a
    DATA slot, not a PARAM one - by Taichi/Quadrants' automatic atomic `+=`
    reduction (sum) or `ctx.bk.atomic_min`/`atomic_max` (min/max). argmin
    composes the whole `bitpack_group` (a FrozenGroup - see build_bitpack_
    group) under the name "bitpack" and accumulates the packed (value,
    index) pair; a second, single-thread FrozenKernel unpacks that pair's
    index into a plain "out" DATA argument, so what make_reduce hands back
    as `argmin_param` holds a plain index on every backend.

    Returns the five FrozenKernels, unbuilt.

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    sum_frozen = _kernel(_make_sum_tmpl(T, n), data=["x", "acc"])
    min_frozen = _kernel(_make_min_tmpl(T, n), data=["x", "acc"])
    max_frozen = _kernel(_make_max_tmpl(T, n), data=["x", "acc"])
    argmin_frozen = _kernel(_make_argmin_tmpl(T, n), data=["x", "acc"], helpers={"bitpack": bitpack_group})
    argmin_unpack_frozen = _kernel(_make_argmin_unpack_tmpl(T), data=["packed_acc", "out"], helpers={"bitpack": bitpack_group})
    return sum_frozen, min_frozen, max_frozen, argmin_frozen, argmin_unpack_frozen
