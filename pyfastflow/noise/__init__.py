"""Procedural white and Perlin noise helpers.

``make_noise_group`` returns the reusable device helper structure;
``make_noise_parameters`` returns its amplitude, seed, and Perlin settings.
Bind the grid dimensions separately when the helper is used in a model.
"""

import numpy as np

from ..core import Backend, FrozenGroup, FrozenHelper, GroupBuilder, require_backend, share_leaf

_KINDS = frozenset({"white", "perlin"})
_MODES = ("const", "scalar")


def permutation_table(seed: int) -> np.ndarray:
    """Return the duplicated 256-entry Perlin permutation table for ``seed``."""
    rng = np.random.default_rng(seed)
    perm = np.arange(256, dtype=np.int32)
    for i in range(255, 0, -1):
        j = rng.integers(0, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return np.concatenate([perm, perm])


def _blocks_for(be: Backend):
    """Return the implementation module for one backend family."""
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
    """Return reusable white- or Perlin-noise helper structure.

    Bind its value parameters from ``make_noise_parameters`` and bind ``NX``
    (and ``NY`` for Perlin) from the matching grid.
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
    """Return the standalone integer hash helper used by white noise."""
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
    """Return parameters for white or Perlin noise.

    Use the same ``kind`` as the noise group. White noise uses amplitude and
    seed; Perlin additionally uses its permutation table and frequency settings.
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
