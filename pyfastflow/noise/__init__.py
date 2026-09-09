"""
make_noise_group / make_noise_parameters: the noise factory pair, built on
the new builder/frozen/bound stack (core/context/builder.py, frozen.py,
bound.py; see parameter.py for Parameter, unchanged) - same split as
grid/__init__.py's make_grid_group/make_grid_parameters (see that module's
own docstring for why the split is forced by the architecture, not a style
choice).

`make_noise_group` returns structure: a FrozenGroup wiring `NX` (always) and
`NY` (Perlin only), plus whatever value params the chosen `kind` needs
(`AMPLITUDE`, `SEED` for white; `AMPLITUDE`, `PERM`, `FX`, `FY`, `OCTAVES`,
`PERSISTENCE` for Perlin) as its own top-level PARAM slots, and composes the
public `at` device helper (plus `hash_u32`, always, and `white_unit`/
`perlin_at`, whichever the chosen `kind` builds) - uniform by name whatever
the backend. `make_noise_parameters` returns data: the concrete owned
Parameters the group's own value PARAM slots need bound.

NX/NY are deliberately NOT among make_noise_parameters' own output. A
FrozenGroup carries no Parameter objects of its own (frozen.py) -
make_grid_group returns pure structure, so there is no live grid Parameter
for noise to read at build time even if it wanted to. Noise instead wires
its own `NX`/`NY` PARAM slots, generic exactly like every other PARAM slot
(slot.py: "deliberately generic ... nothing here constrains mode or
dtype"), and a caller binds them to the *same* Parameter objects
make_grid_parameters already produced -
`kbound.bind_leaf(grid_params, prefix=("noise",))` alongside
`kbound.bind_leaf(grid_params, prefix=("grid",))`. This is why noise's own
`at(i)`/`row`/`col` never structurally compose the grid FrozenGroup at all:
row/column math only ever needs `nx`/`ny` as plain values, never a grid
HELPER call (no neighbour lookup, no distance, nothing structural), so there
is nothing here that hits the nested-FrozenGroup-in-FrozenGroup case
visu/__init__.py's hillshade gradient does (see that module's own docstring)
- noise simply never composes another FrozenGroup as a child at all.

Build-phase sharing collapses the duplicate NX/NY addresses
-------------------------------------------------------------
`NX` is read by the private `row`/`col` blocks and, for Perlin, by `at`
itself (`nx_f`/`ny_f`); `NY` (Perlin only) likewise by `at` itself. Each of
these wires its own local `NX`/`NY` PARAM slot (a device template can only
call/read what is composed or wired directly onto its own scope - builder.py's
module docstring), which would otherwise mint one independent address per
occurrence at build() time. `share_leaf`/`find_param_paths`
(core/context/builder.py) are the same mechanism grid/__init__.py uses for
its own NX/NY/DX (see that module's own docstring) - sharing a canonical
name across two independently-authored composites is always this explicit,
itemized, per-factory `share()` call, never the implicit name-matching
bound.py's own module docstring explicitly warns against. Every value param
(`AMPLITUDE`, `SEED`/`PERM`/`FX`/`FY`/`OCTAVES`/
`PERSISTENCE`) is shared the same way even though each currently occurs only
once in the tree - mirroring grid's own NODATA_MASK/OUTLET_MASK precedent -
so every one of noise's PARAM addresses lives at the group's own top level
(`noise.AMPLITUDE`, not `noise.at.AMPLITUDE`), never buried at whatever depth
the block that happens to read it sits at.

Values match pyfastflow/noise/ for the same seed and settings - this is a
port of that arithmetic, not a reimplementation. White noise lands in
[-amplitude, amplitude]; Perlin is the octave-averaged lattice noise scaled
by amplitude.

`ctx.bk` (core/context/bk.py) is what makes this port possible on the
closure backends at all - noise's hash needs u32-typed arithmetic (including
an oversized-for-i32 literal, `0x846CA68B`) and Perlin needs `floor`/typed
casts, none of which a plain python builtin covers (unlike grid, which only
ever needed `abs`/`min`). See _closure_blocks.py and bk.py's own module
docstring for the mechanism; cupy needs none of this (_cupy_blocks.py stays
plain C, as it always was).

Author: B.G (08/2026)
"""

import numpy as np

from ..core import Backend, FrozenGroup, FrozenHelper, GroupBuilder, require_backend, share_leaf

_KINDS = frozenset({"white", "perlin"})
_MODES = ("const", "scalar")


def permutation_table(seed: int) -> np.ndarray:
    """
    The 512-entry Perlin permutation table for one seed: a Fisher-Yates
    shuffle of 0..255 concatenated with itself, so a lattice lookup can index
    past 255 without wrapping by hand.

    Same construction (and same numpy Generator) as
    pyfastflow/noise/noisecontext.py, so a given seed yields the same table
    and therefore the same noise.

    Parameters
    ----------
    seed : int

    Returns
    -------
    np.ndarray
        512 int32 entries.

    Author: B.G (07/2026)
    """
    rng = np.random.default_rng(seed)
    perm = np.arange(256, dtype=np.int32)
    for i in range(255, 0, -1):
        j = rng.integers(0, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return np.concatenate([perm, perm])


def _blocks_for(be: Backend):
    """
    The private block module implementing make_noise_group's device code for
    one backend name: the closure blocks (shared by Taichi and Quadrants) or
    the cupy blocks.

    Author: B.G (08/2026)
    """
    if be.family == "closure":
        from . import _closure_blocks as blocks
    elif be.family == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"make_noise_group: unsupported backend family {be.family!r}")
    return blocks


def _check_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError(f"make_noise_group: kind must be one of {sorted(_KINDS)}, got {kind!r}")


def make_noise_group(be: Backend, *, kind: str = "perlin") -> FrozenGroup:
    """
    Build one noise's structure: a FrozenGroup wiring `NX` (always) and `NY`
    (Perlin only), plus whatever value params `kind` needs, as its own
    top-level PARAM slots, composing `at`/`hash_u32` (and `white_unit` or
    `perlin_at`) - uniform by name regardless of backend - then declaring
    every private occurrence of each of those PARAM names build-phase-shared
    with the group's own top-level slot (see the module docstring). Returns
    structure only, no Parameter objects - make_noise_parameters is the
    companion that builds the value params (NX/NY are the caller's own, from
    make_grid_parameters - see the module docstring).

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    kind : str, optional
        "white" or "perlin" (default). Picks the block chain.

    Returns
    -------
    FrozenGroup

    Raises
    ------
    ValueError
        If `kind` is not "white" or "perlin".

    Author: B.G (08/2026)
    """
    be = require_backend(be)
    _check_kind(kind)
    blocks = _blocks_for(be)

    group = GroupBuilder()
    group.param("NX")
    group.param("AMPLITUDE")
    if kind == "white":
        group.param("SEED")
    else:
        group.param("NY")
        group.param("PERM")
        group.param("FX")
        group.param("FY")
        group.param("OCTAVES")
        group.param("PERSISTENCE")

    blocks.build_group(group, kind=kind)

    share_leaf(group, "NX")
    share_leaf(group, "AMPLITUDE")
    if kind == "white":
        share_leaf(group, "SEED")
    else:
        share_leaf(group, "NY")
        share_leaf(group, "PERM")
        share_leaf(group, "FX")
        share_leaf(group, "FY")
        share_leaf(group, "OCTAVES")
        share_leaf(group, "PERSISTENCE")

    return group.freeze()


def make_hash_u32(be: Backend) -> FrozenHelper:
    """
    The standalone hash_u32(x) FrozenHelper make_noise_group's white-noise
    chain is built on - no Parameters, no grid, no pool. A caller that only
    wants the same integer hash (e.g. flow's rand_unit) reaches it here
    rather than building a whole noise group just to pull hash_u32 back out
    of it.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".

    Returns
    -------
    FrozenHelper

    Author: B.G (08/2026)
    """
    blocks = _blocks_for(require_backend(be))
    return blocks.build_hash_u32()


def make_noise_parameters(
    be: Backend,
    pool,
    *,
    kind: str = "perlin",
    amplitude: float = 1.0,
    seed: int = 42,
    frequency: float = 8.0,
    frequency_x: float | None = None,
    frequency_y: float | None = None,
    octaves: int = 4,
    persistence: float = 0.5,
    amplitude_mode: str = "const",
    seed_mode: str = "scalar",
    frequency_mode: str = "const",
    octaves_mode: str = "const",
    persistence_mode: str = "const",
) -> dict:
    """
    Build the concrete, caller-owned Parameter objects one noise group's own
    value PARAM slots need bound: {"AMPLITUDE": ..., "SEED": ...} for
    kind="white", {"AMPLITUDE": ..., "PERM": ..., "FX": ..., "FY": ...,
    "OCTAVES": ..., "PERSISTENCE": ...} for kind="perlin". Keys match exactly
    the value-param top-level PARAM slot names make_noise_group()'s
    FrozenGroup wires. `NX`/`NY` are not among these - see the module
    docstring: bind the same Parameter objects make_grid_parameters already
    produced into `noise.NX`/`noise.NY` as well as `grid.NX`/`grid.NY`.

    `kind` must match whatever was passed to make_noise_group() for the
    noise group this backs.

    The `*_mode` arguments are "const" or "scalar" and decide whether a
    value is folded in at compile time or lives in a one-cell device field
    the host can retune. Defaults are "const" except `seed_mode`, which is
    "scalar" - reseeding white noise is a host write rather than a rebuild;
    Perlin's seed is not a device parameter at all, it drives the
    permutation table, built on the host and uploaded as a field (reseed by
    refilling it: `params["PERM"].set(permutation_table(7))`).

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    pool : Pool
        Device-buffer pool backing scalar/field-mode Parameters.
    kind : str, optional
        Must match make_noise_group()'s `kind`.
    amplitude, seed, frequency, frequency_x, frequency_y, octaves,
    persistence : optional
        Initial values.
    amplitude_mode, seed_mode, frequency_mode, octaves_mode,
    persistence_mode : str, optional
        "const" or "scalar"; see above for the `seed_mode` default.

    Returns
    -------
    dict
        {"AMPLITUDE": ..., "SEED": ...} for kind="white";
        {"AMPLITUDE": ..., "PERM": ..., "FX": ..., "FY": ..., "OCTAVES": ...,
        "PERSISTENCE": ...} for kind="perlin".

    Raises
    ------
    ValueError
        If `kind` or any `*_mode` is not a recognised value.

    Author: B.G (08/2026)
    """
    _check_kind(kind)
    for label, mode in (
        ("amplitude_mode", amplitude_mode),
        ("seed_mode", seed_mode),
        ("frequency_mode", frequency_mode),
        ("octaves_mode", octaves_mode),
        ("persistence_mode", persistence_mode),
    ):
        if mode not in _MODES:
            raise ValueError(f"make_noise_parameters: {label} must be 'const' or 'scalar', got {mode!r}")

    be = require_backend(be)
    ParamCls = be.ParameterCls

    amplitude_p = ParamCls(
        "NOISE_AMPLITUDE", dtype="f32", mode=amplitude_mode, value=float(amplitude), pool=pool
    )

    if kind == "white":
        seed_p = ParamCls("NOISE_SEED", dtype="u32", mode=seed_mode, value=int(seed), pool=pool)
        return {"AMPLITUDE": amplitude_p, "SEED": seed_p}

    perm_p = ParamCls(
        "NOISE_PERM", dtype="i32", mode="field", value=permutation_table(seed), pool=pool, shape=(512,)
    )
    fx = float(frequency_x if frequency_x is not None else frequency)
    fy = float(frequency_y if frequency_y is not None else frequency)
    frequency_x_p = ParamCls("NOISE_FX", dtype="f32", mode=frequency_mode, value=fx, pool=pool)
    frequency_y_p = ParamCls("NOISE_FY", dtype="f32", mode=frequency_mode, value=fy, pool=pool)
    octaves_p = ParamCls("NOISE_OCTAVES", dtype="i32", mode=octaves_mode, value=int(octaves), pool=pool)
    persistence_p = ParamCls(
        "NOISE_PERSISTENCE", dtype="f32", mode=persistence_mode, value=float(persistence), pool=pool
    )
    return {
        "AMPLITUDE": amplitude_p,
        "PERM": perm_p,
        "FX": frequency_x_p,
        "FY": frequency_y_p,
        "OCTAVES": octaves_p,
        "PERSISTENCE": persistence_p,
    }
