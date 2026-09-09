"""
cupy (CUDA source) block templates behind make_noise_group.

Mirrors _closure_blocks.py block for block - same private/public split, same
`kind` selector deciding which chain `at(i)` is wired to - written as CUDA
text instead of python defs. Every span reaching a PARAM is spelled
`$ctx.NAME.get(...)$` in full, exactly like grid/_cupy_blocks.py; every span
reaching a composed HELPER is spelled `$ctx.name(args)$`.

No `ctx.bk` here - cupy stays plain C, as grid/_cupy_blocks.py's own module
docstring already establishes for this backend: `floorf`, `(int)`/`(float)`
casts and a plain `0x846CA68Bu` literal are the native spelling, and the
python/cupy template surfaces are already a different grammar by design (see
core/context/bk.py's own module docstring for why `ctx.bk` is deliberately
excluded from cupy).

Every device function name is prefixed with this noise group's own tag (a
fresh new_uid()), so two make_noise_group() calls in one process never
collide inside a single compiled cupy module even if both are bound into the
same kernel - see grid/_cupy_blocks.py.

Author: B.G (08/2026)
"""

from ..core import freeze_helper as _helper, new_uid


def build_hash_u32():
    """
    The standalone hash_u32(x) FrozenHelper - no bound Parameters, so it can
    be built with nothing else in hand. See _closure_blocks.py's
    build_hash_u32 for why this exists.

    Author: B.G (08/2026)
    """
    t = f"pn{new_uid()}"
    return _helper(
        f"""
__device__ unsigned int {t}_hash_u32(unsigned int x) {{
    unsigned int h = x;
    h ^= h >> 16u;
    h *= 0x7FEB352Du;
    h ^= h >> 15u;
    h *= 0x846CA68Bu;
    h ^= h >> 16u;
    return h;
}}
"""
    )


def build_group(group, *, kind):
    """
    Compose every private block and public helper for the cupy backend onto
    `group` (a GroupBuilder), picking the white or Perlin chain from `kind`.

    Returns nothing - every public helper (`at`, `hash_u32`, and
    `white_unit`/`perlin_at`) is compose()d onto `group` itself, under its
    own public name, by this call.

    Author: B.G (08/2026)
    """
    t = f"pn{new_uid()}"

    row = _helper(f"__device__ int {t}_row(int i) {{ return i / $ctx.NX.get(0)$; }}")
    col = _helper(f"__device__ int {t}_col(int i) {{ return i % $ctx.NX.get(0)$; }}")
    hash_u32 = build_hash_u32()
    group.compose("hash_u32", hash_u32)

    if kind == "white":
        white_unit = _helper(
            f"""
__device__ float {t}_white_unit(int i) {{
    // column first, row second - the argument order white_noise.py hashes in
    int c = $ctx.col(i)$;
    int r = $ctx.row(i)$;
    unsigned int key = (unsigned int)$ctx.SEED.get(0)$;
    key ^= (unsigned int)c * 374761393u;
    key ^= (unsigned int)r * 668265263u;
    unsigned int hashed = $ctx.hash_u32(key)$;
    return (float)hashed / 4294967296.0f;
}}
""",
            helpers={"row": row, "col": col, "hash_u32": hash_u32},
        )
        at = _helper(
            f"""
__device__ float {t}_at(int i) {{
    return ($ctx.white_unit(i)$ - 0.5f) * 2.0f * $ctx.AMPLITUDE.get(0)$;
}}
""",
            helpers={"white_unit": white_unit},
        )
        group.compose("at", at)
        group.compose("white_unit", white_unit)
        return

    fade = _helper(f"__device__ float {t}_fade(float t) {{ return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }}")
    lerp = _helper(f"__device__ float {t}_lerp(float t, float a, float b) {{ return a + t * (b - a); }}")
    grad = _helper(
        f"""
__device__ float {t}_grad(int hash_val, float dx, float dy) {{
    int idx = hash_val & 7;
    float gx = 0.0f;
    float gy = 0.0f;
    if (idx == 0)      {{ gx =  1.0f; gy =  1.0f; }}
    else if (idx == 1) {{ gx = -1.0f; gy =  1.0f; }}
    else if (idx == 2) {{ gx =  1.0f; gy = -1.0f; }}
    else if (idx == 3) {{ gx = -1.0f; gy = -1.0f; }}
    else if (idx == 4) {{ gx =  1.0f; gy =  0.0f; }}
    else if (idx == 5) {{ gx = -1.0f; gy =  0.0f; }}
    else if (idx == 6) {{ gx =  0.0f; gy =  1.0f; }}
    else               {{ gx =  0.0f; gy = -1.0f; }}
    return gx * dx + gy * dy;
}}
"""
    )
    perlin_at = _helper(
        f"""
__device__ float {t}_perlin_at(float x, float y) {{
    float x_floor = floorf(x);
    float y_floor = floorf(y);

    int X = ((int)x_floor) & 255;
    int Y = ((int)y_floor) & 255;

    float x_local = x - x_floor;
    float y_local = y - y_floor;

    float u = $ctx.fade(x_local)$;
    float v = $ctx.fade(y_local)$;

    int iX1 = (X + 1) & 255;
    int A = $ctx.PERM.get(X)$ + Y;
    int B = $ctx.PERM.get(iX1)$ + Y;

    int iAA = A & 255;
    int iAB = (A + 1) & 255;
    int iBA = B & 255;
    int iBB = (B + 1) & 255;

    int AA = $ctx.PERM.get(iAA)$;
    int AB = $ctx.PERM.get(iAB)$;
    int BA = $ctx.PERM.get(iBA)$;
    int BB = $ctx.PERM.get(iBB)$;

    float gaa = $ctx.grad(AA, x_local, y_local)$;
    float gba = $ctx.grad(BA, x_local - 1.0f, y_local)$;
    float gab = $ctx.grad(AB, x_local, y_local - 1.0f)$;
    float gbb = $ctx.grad(BB, x_local - 1.0f, y_local - 1.0f)$;

    float lo = $ctx.lerp(u, gaa, gba)$;
    float hi = $ctx.lerp(u, gab, gbb)$;
    return $ctx.lerp(v, lo, hi)$;
}}
""",
        helpers={"fade": fade, "lerp": lerp, "grad": grad},
    )
    at = _helper(
        f"""
__device__ float {t}_at(int i) {{
    float nx_f = (float)$ctx.NX.get(0)$;
    float ny_f = (float)$ctx.NY.get(0)$;

    float x = (float)$ctx.col(i)$ * $ctx.FX.get(0)$ / nx_f;
    float y = (float)$ctx.row(i)$ * $ctx.FY.get(0)$ / ny_f;

    float total = 0.0f;
    float max_value = 0.0f;
    float current_amplitude = 1.0f;
    float current_frequency = 1.0f;

    int octaves = $ctx.OCTAVES.get(0)$;
    for (int o = 0; o < octaves; ++o) {{
        total += $ctx.perlin_at(x * current_frequency, y * current_frequency)$ * current_amplitude;
        max_value += current_amplitude;
        current_amplitude *= $ctx.PERSISTENCE.get(0)$;
        current_frequency *= 2.0f;
    }}

    if (max_value > 0.0f) return (total / max_value) * $ctx.AMPLITUDE.get(0)$;
    return 0.0f;
}}
""",
        helpers={"row": row, "col": col, "perlin_at": perlin_at},
    )
    group.compose("at", at)
    group.compose("perlin_at", perlin_at)
