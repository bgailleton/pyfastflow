"""CUDA operation templates for CuPy."""

from ..core import GroupBuilder, RoutineBuilder, freeze_helper as _helper, freeze_kernel as _kernel, new_uid

# ---------------------------------------------------------------------------
# bitpack: pack(f, i) -> i64, unpack_value(p) -> f32, unpack_index(p) -> i32
# ---------------------------------------------------------------------------


def build_bitpack_group() -> "FrozenGroup":
    """
    pack(f, i) -> i64, unpack_value(p) -> f32, unpack_index(p) -> i32, same
    IEEE-754 bit-flip trick as _closure_blocks.build_bitpack_group, using
    CUDA's __float_as_uint/__uint_as_float. No PARAM slots anywhere in this
    tree.

    """
    t = f"pf{new_uid()}"
    flip = _helper(
        f"""
__device__ unsigned int {t}_flip(float f) {{
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? (u ^ 0x80000000u) : (~u);
}}
"""
    )
    unflip = _helper(
        f"""
__device__ float {t}_unflip(unsigned int u) {{
    unsigned int restored = (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
    return __uint_as_float(restored);
}}
"""
    )
    pack = _helper(
        f"""
__device__ long long {t}_pack(float f, int i) {{
    unsigned int f_enc = $ctx.flip(f)$;
    unsigned int i_enc = (unsigned int)i;
    long long packed = ((long long)f_enc << 32) | (long long)i_enc;
    long long flipped_upper = (~packed) & (0xFFFFFFFFLL << 32);
    long long unchanged_lower = packed & 0xFFFFFFFFLL;
    return flipped_upper | unchanged_lower;
}}
""",
        helpers={"flip": flip},
    )
    unpack_raw = _helper(
        f"""
__device__ long long {t}_unpack_raw(long long packed) {{
    long long flipped_upper = (~packed) & (0xFFFFFFFFLL << 32);
    long long unchanged_lower = packed & 0xFFFFFFFFLL;
    return flipped_upper | unchanged_lower;
}}
"""
    )
    unpack_value = _helper(
        f"""
__device__ float {t}_unpack_value(long long packed) {{
    long long u = $ctx.unpack_raw(packed)$;
    unsigned int f_enc = (unsigned int)(u >> 32);
    return $ctx.unflip(f_enc)$;
}}
""",
        helpers={"unpack_raw": unpack_raw, "unflip": unflip},
    )
    unpack_index = _helper(
        f"""
__device__ int {t}_unpack_index(long long packed) {{
    long long u = $ctx.unpack_raw(packed)$;
    unsigned int i_enc = (unsigned int)(u & 0xFFFFFFFFLL);
    return (int)i_enc;
}}
""",
        helpers={"unpack_raw": unpack_raw},
    )

    group = GroupBuilder()
    group.compose("pack", pack)
    group.compose("unpack_value", unpack_value)
    group.compose("unpack_index", unpack_index)
    return group.freeze()


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def build_math_group() -> "FrozenGroup":
    """
    atan(x) via atan2f(x, 1); nextafter(x, y), one ULP of f32 towards y via
    the same bit-twiddling as _closure_blocks.build_math_group - composed
    onto a fresh GroupBuilder under those two public names. No PARAM slots.

    """
    t = f"pf{new_uid()}"
    atan = _helper(f"__device__ float {t}_atan(float x) {{ return atan2f(x, 1.0f); }}")
    nextafter = _helper(
        f"""
__device__ float {t}_nextafter(float x, float y) {{
    float result = y;
    if (x != y) {{
        unsigned int sign_mask = 0x80000000u;
        unsigned int ix = __float_as_uint(x);
        if (x == 0.0f) {{
            ix = (__float_as_uint(y) & sign_mask) | 1u;
        }} else if ((x > 0.0f) == (y > x)) {{
            ix += 1u;
        }} else {{
            ix -= 1u;
        }}
        result = __uint_as_float(ix);
    }}
    return result;
}}
"""
    )

    group = GroupBuilder()
    group.compose("atan", atan)
    group.compose("nextafter", nextafter)
    return group.freeze()


# ---------------------------------------------------------------------------
# Elementwise kernels close over ``n`` as a build-time Python integer.
# ---------------------------------------------------------------------------


def build_elementwise(n: int) -> dict:
    """
    swap/add_B_to_A/add_B_to_weighted_A/weighted_mean_B_in_A/arange/
    multiply_by_scalar over a flat f32 buffer of length `n`, as unbuilt
    FrozenKernels.

    """
    t = f"pf{new_uid()}"

    def _k(template, names):
        return KernelBuilder(template, domain=n).freeze()

    return {
        "swap": _k(
            f"""
extern "C" __global__ void {t}_swap(float* array1, float* array2) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    float temp = array1[i];
    array1[i] = array2[i];
    array2[i] = temp;
}}
""",
            ["array1", "array2"],
        ),
        "add_B_to_A": _k(
            f"""
extern "C" __global__ void {t}_add_B_to_A(float* array1, const float* array2) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    array1[i] += array2[i];
}}
""",
            ["array1", "array2"],
        ),
        "add_B_to_weighted_A": _k(
            f"""
extern "C" __global__ void {t}_add_B_to_weighted_A(float* array1, const float* array2, float weight) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    array1[i] += array2[i] * weight;
}}
""",
            ["array1", "array2", "weight"],
        ),
        "weighted_mean_B_in_A": _k(
            f"""
extern "C" __global__ void {t}_weighted_mean_B_in_A(float* array1, const float* array2, float weight) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    array1[i] = array2[i] * weight + array1[i] * (1.0f - weight);
}}
""",
            ["array1", "array2", "weight"],
        ),
        "arange": _k(
            f"""
extern "C" __global__ void {t}_arange(float* array) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    array[i] = (float)i;
}}
""",
            ["array"],
        ),
        "multiply_by_scalar": _k(
            f"""
extern "C" __global__ void {t}_multiply_by_scalar(float* A, float scalar) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    A[i] *= scalar;
}}
""",
            ["A", "scalar"],
        ),
    }


# ---------------------------------------------------------------------------
# slope (grid-aware) - the first ops/ case of a nested FrozenGroup-in-Frozen
# Group child on cupy, mirroring _closure_blocks.build_slope_group
# ---------------------------------------------------------------------------


def build_slope_group(grid) -> "FrozenGroup":
    """
    sumslope_downstream(z, i) / slope_dir(z, i, k), same arithmetic as
    _closure_blocks.build_slope_group, walking `grid`'s neighbour/dx/
    n_neighbours surface through `$ctx.grid...$` spans - `grid` composed
    independently as each helper's own child, same nested-FrozenGroup shape
    as the closure port (see that module's own docstring).

    """
    t = f"pf{new_uid()}"
    sumslope_downstream = _helper(
        f"""
__device__ float {t}_sumslope_downstream(const float* z, int i) {{
    float sumslope = 0.0f;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(i, k)$;
        if (j > -1) {{
            if (z[j] < z[i]) {{
                sumslope += (z[i] - z[j]) / $ctx.grid.DX.get(0)$;
            }}
        }}
    }}
    return sumslope;
}}
""",
        helpers={"grid": grid},
    )
    slope_dir = _helper(
        f"""
__device__ float {t}_slope_dir(const float* z, int i, int k) {{
    int j = $ctx.grid.neighbour(i, k)$;
    float slope = 0.0f;
    if (j > -1) {{
        slope = (z[i] - z[j]) / $ctx.grid.DX.get(0)$;
    }}
    return slope;
}}
""",
        helpers={"grid": grid},
    )

    group = GroupBuilder()
    group.compose("sumslope_downstream", sumslope_downstream)
    group.compose("slope_dir", slope_dir)
    group.share_identical("sumslope_downstream.grid", as_="grid")

    return group.freeze()


# ---------------------------------------------------------------------------
# block_reduce (cub::BlockReduce wrapper) - cupy only
# ---------------------------------------------------------------------------


def build_block_reduce_group(block_size: int = 128) -> "FrozenGroup":
    """
    sum(val): one cub::BlockReduce<float, block_size>::Sum() per calling
    block, returning the block-wide total to thread 0 (undefined on other
    threads - cub's own contract), composed under the public name "sum". The
    first compile that reaches this triggers a one-time jitify header cache
    warm-up for <cub/block/block_reduce.cuh>, roughly two minutes; that is
    expected, not a hang.

    """
    t = f"pf{new_uid()}"
    sum_helper = _helper(
        f"""
#include <cub/block/block_reduce.cuh>
__device__ float {t}_block_reduce_sum(float val) {{
    typedef cub::BlockReduce<float, {int(block_size)}> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    return BlockReduceT(temp_storage).Sum(val);
}}
"""
    )
    group = GroupBuilder()
    group.compose("sum", sum_helper)
    return group.freeze()


# ---------------------------------------------------------------------------
# scan compaction: read_count + scatter, as a 2-step FrozenRoutine
# ---------------------------------------------------------------------------


def build_count_and_scatter_routine(n: int, *, block: int = 256) -> "FrozenRoutine":
    """
    A 2-step FrozenRoutine (routine.py): "read_count" (one thread, writes
    scan_out[n-1] into the wired PARAM slot "COUNT") then "scatter" (one
    thread per node, `ids[scan_out[i]-1] = i` wherever `flags[i] != 0`) - the
    compaction half of scan-based stream compaction. Each kernel declares its
    own domain: one thread for "read_count", and `n` threads for "scatter".

    """
    t = f"pf{new_uid()}"
    read_count = _kernel(
        f"""
extern "C" __global__ void {t}_read_count(const int* scan_out) {{
    $ctx.COUNT.set_node(0, scan_out[{int(n)} - 1])$;
}}
""",
        data=["scan_out"],
        domain=1,
        block=1,
    )
    scatter = _kernel(
        f"""
extern "C" __global__ void {t}_scatter(const int* flags, const int* scan_out, int* ids) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    if (flags[i] != 0) {{
        ids[scan_out[i] - 1] = i;
    }}
}}
""",
        data=["flags", "scan_out", "ids"],
        domain=int(n),
        block=block,
    )

    rb = RoutineBuilder()
    rb.step("read_count", read_count)
    rb.step("scatter", scatter)
    return rb.freeze()
