"""
make_hillshade_group / make_hillshade_parameters / make_hillshade_kernel: the
hillshading factory set, built on the new builder/frozen/bound stack
(core/context/builder.py, frozen.py, bound.py; see parameter.py for
Parameter, unchanged) and on a caller-supplied grid FrozenGroup
(../grid/make_grid_group).

Three functions, not two
--------------------------
grid's/noise's own factory shape is two calls, structure vs data
(make_X_group -> FrozenGroup, make_X_parameters -> dict[str, Parameter] -
see grid/__init__.py's own docstring). visu needs a third: `at(z, i)` (a
device HELPER, shade value at one node, for a caller wanting hillshade
inline in its own kernel) fits `make_hillshade_group`'s FrozenGroup exactly
like grid's public helpers do, but the standalone `hillshade` pass (writes
shade for every node) is a KERNEL, a host entry point, and `compose()`
(builder.py) raises outright on a FrozenKernel - "a kernel is a host entry
point, not something device code can call". There is structurally nowhere
inside a FrozenGroup a compiled kernel could ever live, so it cannot be a
FrozenGroup member the way `at` is. `make_hillshade_kernel` is the third
function this forces: it takes the already-built `make_hillshade_group`
result and composes the *whole group* onto a fresh KernelBuilder, returning
a FrozenKernel a caller `.build()`s, binds data (`z`, `out`) and Parameters
into, and `.compile()`s independently - exactly the same build->bind->compile
lifecycle as any other FrozenKernel, just produced by this module instead of
written by hand.

Nested FrozenGroup-in-FrozenGroup: the first real case
----------------------------------------------------------
grid has no nested groups (its own composed children are all FrozenHelpers);
noise never composes the grid FrozenGroup at all (see noise/__init__.py's
own docstring - it only ever needs nx/ny as plain values). visu's gradient
blocks (`_gradient_x`/`_gradient_y`, in _closure_blocks.py/_cupy_blocks.py)
are the first thing in this rewrite that actually calls a grid HELPER
(`grid.neighbour(i, k)`) from inside a private block of its own - and a
device template can only call/read what is composed directly onto its own
scope, never a sibling's or a parent's (builder.py's module docstring), so
each of `_gradient_x`/`_gradient_y` must independently compose the caller's
grid FrozenGroup as its own child. That is a FrozenGroup (grid, which
already carries `.shared` entries for its own NX/NY/DX/... - see grid's own
docstring) composed as a child two levels down inside another FrozenGroup
(this module's own hillshade group) - bound.py's `_walk_group`/
`_walk_group_subtree` describe this as starting "a fresh sharing scope at
each group boundary" but nothing in the tree so far had exercised it. It
works, verified end to end (build(), inspect(), and a real hillshade number
matching a numpy reference on all three backends - see this module's own
verification run): `_walk_group_subtree`, on reaching a composed child that
is itself a FrozenGroup with `.shared` entries, correctly recurses via a
fresh `_walk_group` call rather than treating it as an ordinary FrozenHelper
child.

What is NOT free: composing the same `grid` FrozenGroup independently under
`_gradient_x` and `_gradient_y` mints TWO independent copies of every one of
grid's own top-level PARAM names (NX, NY, DX, N_NEIGHBOURS, ...) - `_walk_
group` mints every name in `group.slots.names(PARAM)` unconditionally,
regardless of whether the composing template happens to read it, so even
though `_gradient_x`/`_gradient_y` only ever read `GRID.neighbour(...)` (a
HELPER, shared by object identity, no address needed) and `GRID.DX.get(0)`,
composing `grid` still mints `NX`/`NY`/`N_NEIGHBOURS` addresses nobody reads,
twice over. This is exactly the case `share()` exists for: `make_hillshade_
group` wires every name in `grid.slots.names(PARAM)` at its own top level as
a canonical and `share_leaf`s (core/context/builder.py) every occurrence
found anywhere in its own composed subtree - the identical mechanism
grid/__init__.py and
noise/__init__.py already use, `share()`'s own path-walk resolving through a
nested FrozenGroup exactly as it would through a FrozenHelper (see
builder.py's `GroupBuilder.share()` - the walk checks `node.children`/`node.
slots.names(PARAM)` generically, indifferent to which kind of node it is
looking at). The result: a caller binds `hillshade.NX`/`hillshade.NY`/
`hillshade.DX`/`hillshade.N_NEIGHBOURS` once (to the same Parameter objects
already bound to `grid.NX`/...), not once per gradient block.

Structure needs `topology`, data does not, and the grid FrozenGroup itself
carries neither
------------------------------------------------------------------------------
Deciding which neighbour index `k` means "left"/"right"/"top"/"bottom" needs
the grid's own n_neighbours count. A FrozenGroup carries no Parameter
objects (frozen.py: pure structure), so there is nothing here to read a
concrete n_neighbours value off even in principle. `make_hillshade_group`
therefore takes an explicit `topology`
("D4"|"D8") argument instead, exactly mirroring make_grid_group's own - and,
exactly like the grid_group/grid_parameters pair, it is the caller's job to
pass the same topology the grid FrozenGroup composed here was itself built
with; nothing here cross-checks the two.

`k_left`/`k_right`/`k_top`/`k_bottom`, once picked from `topology`, are
per-call integers, not template-global constants the way e.g. grid's own
`_SQRT2` is - `_closure_blocks.py`'s `_make_gradient_x_tmpl`/`_make_gradient_
y_tmpl` close over them as ordinary python closure variables from a nested
def; `compile_closure.py`'s `_compile_dropping_ctx` carries a template's own
closure cells forward into its rebuilt globals (see that module's own
docstring). `_make_hillshade_kernel_tmpl`'s `z`/`out` annotations use the
`ti.template()`/`qd.template()` type marker the same nested-def way, though
that is a distinct case under the hood - an annotation-only name is not a
closure cell at all (evaluated eagerly by the enclosing frame at `def` time,
never read by the annotated function's own bytecode), so `_compile_dropping_
ctx` carries it forward via the original template's own `__annotations__`
instead - see that module's own docstring for why the two need separate
handling. `_cupy_blocks.py` needs no equivalent - its templates are already
f-strings with values substituted directly into CUDA text, the same
`new_uid()`-tagging grid/_cupy_blocks.py and noise/_cupy_blocks.py use.

`ctx.bk` (core/context/bk.py) supplies `sqrt`/`atan2`/`cos`/`sin` for the
hillshade formula itself - `max`/`min` stay plain python builtins, as grid
already established.

Author: B.G (08/2026)
"""

from ..core import (
    Backend, FrozenGroup, FrozenKernel, GroupBuilder, SlotKind, require_backend,
    share_leaf,
)

_TOPOLOGIES = {"D4": 4, "D8": 8}
_MODES = ("const", "scalar")


def _blocks_for(be: Backend):
    """
    The private block module implementing make_hillshade_group's device code
    for one backend name: the closure blocks (shared by Taichi and
    Quadrants) or the cupy blocks.

    Author: B.G (08/2026)
    """
    if be.family == "closure":
        from . import _closure_blocks as blocks
    elif be.family == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"make_hillshade_group: unsupported backend family {be.family!r}")
    return blocks


def _k_indices(topology: str):
    if topology == "D4":
        return {"k_top": 0, "k_left": 1, "k_right": 2, "k_bottom": 3}
    if topology == "D8":
        return {"k_top": 1, "k_left": 3, "k_right": 4, "k_bottom": 6}
    raise ValueError(f"make_hillshade_group: topology must be one of {sorted(_TOPOLOGIES)}, got {topology!r}")


def make_hillshade_group(be: Backend, grid: FrozenGroup, *, topology: str = "D8") -> FrozenGroup:
    """
    Build one hillshade's structure: a FrozenGroup wiring `AZIMUTH`/
    `ALTITUDE`/`ZFACTOR` (its own value params) plus every name in `grid`'s
    own top-level PARAM slots (NX/NY/DX/N_NEIGHBOURS, and NODATA_MASK/
    OUTLET_MASK if `grid` carries them) as its own top-level PARAM slots too
    - the latter purely as build-phase-sharing canonicals, see the module
    docstring's "Nested FrozenGroup-in-FrozenGroup" section - and composing
    the public `at(z, i)` device helper, uniform by name regardless of
    backend. Returns structure only, no Parameter objects -
    make_hillshade_parameters is the companion that builds AZIMUTH/ALTITUDE/
    ZFACTOR's own values (NX/NY/DX/N_NEIGHBOURS are the caller's own, from
    make_grid_parameters - bind the same objects into `hillshade.NX`/... as
    well as `grid.NX`/...).

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    grid : FrozenGroup
        The grid structure this hillshade reads neighbours/distances from.
    topology : str, optional
        "D4" or "D8" (default). Must match whatever `grid` was itself built
        with - see the module docstring's "Structure needs topology" section
        for why this module cannot read that off `grid` itself.

    Returns
    -------
    FrozenGroup

    Raises
    ------
    ValueError
        If `topology` is not "D4" or "D8".

    Author: B.G (08/2026)
    """
    if topology not in _TOPOLOGIES:
        raise ValueError(f"make_hillshade_group: topology must be one of {sorted(_TOPOLOGIES)}, got {topology!r}")
    be = require_backend(be)
    blocks = _blocks_for(be)
    k = _k_indices(topology)

    group = GroupBuilder()
    group.param("AZIMUTH")
    group.param("ALTITUDE")
    group.param("ZFACTOR")
    grid_param_names = grid.slots.names(SlotKind.PARAM)
    for name in grid_param_names:
        group.param(name)

    blocks.build_group(group, grid=grid, **k)

    share_leaf(group, "AZIMUTH")
    share_leaf(group, "ALTITUDE")
    share_leaf(group, "ZFACTOR")
    for name in grid_param_names:
        share_leaf(group, name)

    return group.freeze()


def make_hillshade_parameters(
    be: Backend,
    pool,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    azimuth_mode: str = "const",
    altitude_mode: str = "const",
    z_factor_mode: str = "const",
) -> dict:
    """
    Build the concrete, caller-owned Parameter objects one hillshade group's
    own value PARAM slots need bound: {"AZIMUTH": ..., "ALTITUDE": ...,
    "ZFACTOR": ...}. NX/NY/DX/N_NEIGHBOURS (and NODATA_MASK/OUTLET_MASK, if
    present) are not among these - see the module docstring: bind the same
    Parameter objects make_grid_parameters already produced into
    `hillshade.NX`/... as well as `grid.NX`/....

    `azimuth`/`altitude` are light-source angles in degrees (315/45 is the
    classic NW-lit default); `z_factor` scales the gradient before it enters
    the slope/aspect computation.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    pool : Pool
        Device-buffer pool backing scalar-mode Parameters.
    azimuth, altitude : float, optional
        Light-source angles in degrees.
    z_factor : float, optional
        Gradient scale factor.
    azimuth_mode, altitude_mode, z_factor_mode : str, optional
        "const" (default) or "scalar" - same convention as
        make_grid_parameters/make_noise_parameters.

    Returns
    -------
    dict
        {"AZIMUTH": ..., "ALTITUDE": ..., "ZFACTOR": ...}.

    Raises
    ------
    ValueError
        If any `*_mode` is not "const" or "scalar".

    Author: B.G (08/2026)
    """
    for label, mode in (
        ("azimuth_mode", azimuth_mode),
        ("altitude_mode", altitude_mode),
        ("z_factor_mode", z_factor_mode),
    ):
        if mode not in _MODES:
            raise ValueError(f"make_hillshade_parameters: {label} must be 'const' or 'scalar', got {mode!r}")

    ParamCls = require_backend(be).ParameterCls

    azimuth_p = ParamCls("HS_AZIMUTH", dtype="f32", mode=azimuth_mode, value=float(azimuth), pool=pool)
    altitude_p = ParamCls("HS_ALTITUDE", dtype="f32", mode=altitude_mode, value=float(altitude), pool=pool)
    z_factor_p = ParamCls("HS_ZFACTOR", dtype="f32", mode=z_factor_mode, value=float(z_factor), pool=pool)

    return {"AZIMUTH": azimuth_p, "ALTITUDE": altitude_p, "ZFACTOR": z_factor_p}


def make_hillshade_kernel(be: Backend, hillshade_group: FrozenGroup) -> FrozenKernel:
    """
    The standalone `hillshade` pass: a FrozenKernel composing the *whole*
    `hillshade_group` under the name `hillshade` and writing `out[i] =
    hillshade.at(z, i)` for every node - see the module docstring's "Three
    functions, not two" section for why this cannot be a FrozenGroup member
    the way `at` is. `z`/`out` are DATA slots (cupy additionally takes `n`,
    the node count, as a DATA slot too - a `cp.RawModule` kernel has no
    auto-ranging equivalent to Taichi/Quadrants' `for i in range(n)`).

    A caller `.build()`s the result, binds `z`/`out` (and, on cupy, `n`) plus
    every PARAM address `hillshade_group` itself carries (`hillshade.NX`,
    `hillshade.AZIMUTH`, ...), then `.compile()`s.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    hillshade_group : FrozenGroup
        The group built by make_hillshade_group(), composed whole under
        `hillshade`.

    Returns
    -------
    FrozenKernel

    Author: B.G (08/2026)
    """
    be = require_backend(be)
    blocks = _blocks_for(be)
    if be.family == "cupy":
        return blocks.build_kernel(hillshade_group)
    return blocks.build_kernel(hillshade_group, backend=be.name)
