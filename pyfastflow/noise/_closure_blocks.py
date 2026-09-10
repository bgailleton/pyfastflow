"""Python white- and Perlin-noise templates for Taichi and Quadrants."""

from ..core import freeze_helper as _helper

# ---------------------------------------------------------------------------
# shared: flat index -> row / column
# ---------------------------------------------------------------------------


def _row_tmpl(ctx, i):
    return i // ctx.NX.get(0)


def _col_tmpl(ctx, i):
    return i % ctx.NX.get(0)


# ---------------------------------------------------------------------------
# white noise
# ---------------------------------------------------------------------------


def _hash_u32_tmpl(ctx, x):
    h = x
    h ^= h >> ctx.bk.u32(16)
    h *= ctx.bk.u32(0x7FEB352D)
    h ^= h >> ctx.bk.u32(15)
    h *= ctx.bk.u32(0x846CA68B)
    h ^= h >> ctx.bk.u32(16)
    return h


def _white_unit_tmpl(ctx, i):
    # column first, row second - the argument order white_noise.py hashes in
    col = ctx._COL(i)
    row = ctx._ROW(i)
    key = ctx.bk.u32(ctx.SEED.get(0))
    key ^= ctx.bk.u32(col) * ctx.bk.u32(374761393)
    key ^= ctx.bk.u32(row) * ctx.bk.u32(668265263)
    hashed = ctx._HASH(key)
    return float(hashed) / 4294967296.0


def _at_white_tmpl(ctx, i):
    return (ctx._WHITEUNIT(i) - 0.5) * 2.0 * ctx.AMPLITUDE.get(0)


# ---------------------------------------------------------------------------
# perlin noise
# ---------------------------------------------------------------------------


def _fade_tmpl(ctx, t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp_tmpl(ctx, t, a, b):
    return a + t * (b - a)


def _grad_tmpl(ctx, hash_val, dx, dy):
    idx = hash_val & 7
    gx = 0.0
    gy = 0.0

    if idx == 0:
        gx, gy = 1.0, 1.0
    elif idx == 1:
        gx, gy = -1.0, 1.0
    elif idx == 2:
        gx, gy = 1.0, -1.0
    elif idx == 3:
        gx, gy = -1.0, -1.0
    elif idx == 4:
        gx, gy = 1.0, 0.0
    elif idx == 5:
        gx, gy = -1.0, 0.0
    elif idx == 6:
        gx, gy = 0.0, 1.0
    else:
        gx, gy = 0.0, -1.0

    return gx * dx + gy * dy


def _perlin_at_tmpl(ctx, x, y):
    x_floor = ctx.bk.floor(x)
    y_floor = ctx.bk.floor(y)

    X = int(x_floor) & 255
    Y = int(y_floor) & 255

    x_local = x - x_floor
    y_local = y - y_floor

    u = ctx._FADE(x_local)
    v = ctx._FADE(y_local)

    A = ctx.PERM.get(X) + Y
    B = ctx.PERM.get((X + 1) & 255) + Y
    AA = ctx.PERM.get(A & 255)
    AB = ctx.PERM.get((A + 1) & 255)
    BA = ctx.PERM.get(B & 255)
    BB = ctx.PERM.get((B + 1) & 255)

    return ctx._LERP(
        v,
        ctx._LERP(u, ctx._GRAD(AA, x_local, y_local), ctx._GRAD(BA, x_local - 1.0, y_local)),
        ctx._LERP(u, ctx._GRAD(AB, x_local, y_local - 1.0), ctx._GRAD(BB, x_local - 1.0, y_local - 1.0)),
    )


def _at_perlin_tmpl(ctx, i):
    nx_f = float(ctx.NX.get(0))
    ny_f = float(ctx.NY.get(0))

    x = float(ctx._COL(i)) * ctx.FX.get(0) / nx_f
    y = float(ctx._ROW(i)) * ctx.FY.get(0) / ny_f

    total = 0.0
    max_value = 0.0
    current_amplitude = 1.0
    current_frequency = 1.0

    for _ in range(ctx.OCTAVES.get(0)):
        total += ctx._PERLINAT(x * current_frequency, y * current_frequency) * current_amplitude
        max_value += current_amplitude
        current_amplitude *= ctx.PERSISTENCE.get(0)
        current_frequency *= 2.0

    out = 0.0
    if max_value > 0.0:
        out = (total / max_value) * ctx.AMPLITUDE.get(0)
    return out


def build_hash_u32():
    """Build the standalone integer hash helper."""
    return _helper(_hash_u32_tmpl)


def build_group(group, *, kind):
    """Compose the selected closure-backend noise helpers onto ``group``."""
    row = _helper(_row_tmpl, params=["NX"])
    col = _helper(_col_tmpl, params=["NX"])
    hash_u32 = build_hash_u32()
    group.compose("hash_u32", hash_u32)

    if kind == "white":
        white_unit = _helper(
            _white_unit_tmpl,
            params=["SEED"],
            helpers={"_ROW": row, "_COL": col, "_HASH": hash_u32},
        )
        at = _helper(_at_white_tmpl, params=["AMPLITUDE"], helpers={"_WHITEUNIT": white_unit})
        group.compose("at", at)
        group.compose("white_unit", white_unit)
        return

    fade = _helper(_fade_tmpl)
    lerp = _helper(_lerp_tmpl)
    grad = _helper(_grad_tmpl)
    perlin_at = _helper(
        _perlin_at_tmpl, params=["PERM"], helpers={"_FADE": fade, "_LERP": lerp, "_GRAD": grad}
    )
    at = _helper(
        _at_perlin_tmpl,
        params=["NX", "NY", "FX", "FY", "OCTAVES", "PERSISTENCE", "AMPLITUDE"],
        helpers={"_ROW": row, "_COL": col, "_PERLINAT": perlin_at},
    )
    group.compose("at", at)
    group.compose("perlin_at", perlin_at)
