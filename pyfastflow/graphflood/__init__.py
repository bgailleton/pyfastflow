"""
make_graphflood: the GraphFlood vanilla-SFD solver, built on top of the
already-ported ../flow (receivers, accumulation, depressions, fill-by-
reconstruction) and ../grid, plus this package's own friction-law helper
(_closure_friction.py/_cupy_friction.py) and per-step core kernels
(_closure_core.py/_cupy_core.py) - see ../../flood/flood_graphflood_kernels.py
for the legacy kernels this ports the physics from.

One GraphFlood timestep, in order:

    1. (h-aware) receivers on the current (z, h) -> rec, then a prior
       local-minima resolution over rec/h - `fill_method` picks which:
         "jump": depression routing (../flow's make_depressions/
           make_depression_solver, reroute="carve" - always carve, never
           jump-reroute, despite this value's own name; see below) -
           redirects `rec` around pits without touching z or h.
         "reconstruct": grayscale morphological reconstruction (../flow's
           make_fill_reconstruct/make_fill_reconstruct_solver) run against
           the z+h surface, not bare z - see "Filling on the surface" below.
    2. full downstream accumulation over the resolved graph (../flow's
       make_accumulation) with a caller-bound rain/inflow SOURCE Parameter
       -> Q_in.
    3. this package's own "core" step: compute_qo (friction-law outflow
       capacity from the steepest h-aware slope) then apply_divergence
       (h += DT*(Q_in - Qo)/DX**2, clamped) - see _closure_core.py's module
       docstring for why splitting the legacy single-kernel
       graphflood_core_kernel into these two already avoids the race its
       own two-pass dh buffer existed for, with no dh buffer needed here.
       What a can_out node does in this step is a build-time choice,
       `outlet_behavior` - see "Outlet behaviour" below.

Outlet behaviour
------------------
The grid's own `can_out(i)` helper only answers "is this node allowed to
drain out of the domain" - it carries no opinion on what that drainage
looks like, and until now this factory hardcoded exactly one answer.
`outlet_behavior` (build-time, picked - never branched on inside one
kernel body, same as `friction_law`) now selects it explicitly, in both
compute_qo and apply_divergence (_closure_core.py/_cupy_core.py's own
docstrings on each function have the per-behavior kernel-level detail):

  "fixed_h" (default): a Dirichlet stage boundary - apply_divergence pins
    h[i] = BOUNDARY_H at every can_out node regardless of Q_in/Qo;
    compute_qo correspondingly never computes a real Qo there (Qo[i] = 0,
    unused). Requires `boundary_h_p`.
  "free": an open/free-outfall boundary - can_out nodes are not special-
    cased at all. compute_qo gives them a real friction-law Qo from their
    own steepest h-aware slope exactly like an interior node, and
    apply_divergence runs the same Q_in/Qo divergence update on them -
    water actually drains out at the rate the friction law and whatever
    water is present there give it, rather than being pinned to a fixed
    stage. Wires no `BOUNDARY_H` at all; `boundary_h_p` is not required
    (and unused if given).
  "fixed_s": a normal/uniform-flow outfall - like "free" (apply_divergence
    is identical, no special case, no `BOUNDARY_H`), but compute_qo gives a
    can_out node's Qo from a caller-supplied constant hydraulic slope
    (`boundary_slope_p`) instead of its own locally-computed steepest
    slope, through the exact same friction-law call every other node uses
    (see _closure_core.py's build_compute_qo docstring - the friction
    helper already takes h/slope as plain arguments, so this needs no
    second copy of it). Useful when the real exit slope is known (e.g. a
    surveyed channel grade) and better trusted than this cell's own local
    geometry. Requires `boundary_slope_p`; does not require `boundary_h_p`.

Filling on the surface
------------------------
`fill_method="reconstruct"` does not modify ../flow's fill_reconstruct
kernels at all: this module's own `make_surface` kernel builds `surface =
z + h`, that gets bound as fill_reconstruct's own "z" data argument (the
factory is generic over what "elevation" means), and this module's own
`h_from_filled` kernel (`h = max(0, filled - z)`, against the REAL z) pulls
the result back into a depth field afterwards - two small dedicated kernels,
no changes to _closure_reconstruct.py/_cupy_reconstruct.py.
make_fill_reconstruct_solver's own "counters"/"queued_gen" caller-side init
is a single-solve contract ("zeroed/filled once, before the first call" -
../flow/__init__.py); since GraphFlood calls it once per timestep on a
surface that changes every timestep, this module's own
`build_reset_reconstruct_scratch` (_closure_core.py/_cupy_core.py) re-zeros/
re-fills both immediately before every call - see that function's own
docstring for what reusing stale state without this would corrupt.
`parent` (the
receiver graph fill_reconstruct converges directly, no separate depression
pass, no basin ids) stands in for `rec` in this path - `make_receivers` is
not called at all, since fill_reconstruct's own descent already resolves
routing and pits together. `h_from_filled` runs before accumulation/core,
so a depression's rise is already reflected in h before Qo/divergence read
it. Nothing here enforces a nonzero gradient across a filled plateau -
`filled`/`parent` come out of ../flow's fill_reconstruct exactly as
documented there (an exact grayscale reconstruction, flat where a basin is
genuinely flat); the only floor on slope is compute_qo's own friction-law
epsilon clamp (`_closure_friction.py`'s `_MIN_SLOPE`), the same guard
legacy's compute_qo_from_h_slope used, there to keep Qo's division finite,
not to fabricate a physical gradient a real flat lake surface would not
have.

`fill_method="jump"` always resolves depressions with `reroute="carve"` -
there is no `reroute` parameter on this factory, and carve is the only
option it ever drives (jump-reroute is cheaper per pass but produces a
worse - less locally coherent - reroute; carve is the better default and
the only one wired here). "jump" is this fill_method value's own name
(distinguishing it from "reconstruct" - no surface fill, `rec` is
redirected around each pit instead), unrelated to which reroute technique
runs underneath. It needs make_receivers's `rec` plus every buffer
make_depression_solver(reroute="carve") itself needs: `bid`, `rec_jump`,
`z_prime`, `is_border`, `basin_saddle`, `basin_saddlenode`, `outlet`,
`ndep_p` always; `depression_method="vanilla"` additionally needs `tag`,
`tag_alt`, `rec_scratch` and `rerouted` (the carve-vanilla combination's own
extra scratch, per make_depression_solver's own docstring, ../flow/
__init__.py); the default `depression_method="optimized"` needs none of
those four.

kind="unstable"
-----------------
Bypasses routing, local-minima resolution and accumulation entirely -
../../flood/flood_graphflood_kernels.py's distribute_flow_local_kernel,
ported as this package's own build_distribute (_closure_core.py/
_cupy_core.py), followed by build_copy_q then the same compute_qo/
apply_divergence core every kind uses. Every node redistributes its own
current inflow Q_in[i] to its immediate downslope (h-aware) neighbours in
proportion to their slope share, every step - no receiver graph, no
depression solve, no fill. A local pit (no downslope neighbour) keeps its
own inflow in place and nudges h up by GF_MIN_INCREMENT, so it drains out
gradually as the outer step loop runs rather than being resolved exactly -
this replaces an acyclic-graph guarantee with a graduated approximation,
which is what makes the method "unstable" relative to kind="vanilla_sfd".
One compiled RoutineBuilder covers the whole step (closure backends: one
"distribute" kernel, two top-level loops, already barrier-separated by the
backend's own launch model; cupy: "distribute_zero"/"distribute_route", two
real launches, same reasoning as make_accumulation's method="atomic"
q_init/accum split) - no host decision anywhere in it, unlike
kind="vanilla_sfd"'s per-step orchestration (GraphfloodUnstable.step() is
one call, not several).

kind="vanilla_mfd"
--------------------
cupy-only (raises otherwise). Always fills by reconstruction (never
`fill_method="jump"` - there is no `fill_method` parameter under this
kind), for the same reason ../../CLAUDE.md's own state notes give for why
`persistent_mfd` accumulation itself is reconstruct-only: MFD needs a fully
resolved, monotonic surface to split weights over, which reroute-only
depression handling does not produce. Per step: make_surface -> reset
counters/queued_gen -> fill_reconstruct_solver -> h_from_filled (all
identical to kind="vanilla_sfd"'s fill_method="reconstruct" path - see
"Filling on the surface" above), then "reconstruct_epsilon"'s own
self-scaling perturbation pass (_cupy_reconstruct_epsilon.py - a per-cell
`dist`, a per-cell-ULP cumulative sum along `parent`, not a fixed constant
- see that module's own docstring for why plain `filled` gives every cell
inside a resolved, genuinely-flat depression zero outgoing MFD edges,
stalling accumulation at the flat's boundary, why a *fixed* epsilon
constant is itself wrong on real DEM data (too small relative to float32's
ULP at real elevation magnitudes, silently swallowed by rounding), and why
`dist` is passed to dirs_weights as a separate tie-break carrier rather
than added into `filled` - which would round the per-hop signal away),
then this package's
own MFD topology construction (_cupy_mfd_topology.py's build_mfd_topology -
`dirs`/`mfd_w`/`indegree`, built from `filled` with `dist` as the flat-
tie-break carrier, every step) feeding ../flow's `persistent_mfd`
accumulation (_cupy_mfd_accum.py's build_persistent_mfd/persistent_grid_block/
init_frontier_mfd) instead of the SFD accumulation kind="vanilla_sfd"
uses, then the same compute_qo/apply_divergence core every kind uses (fed
the real, un-perturbed `h`/`filled`; `dist` only ever reaches
`dirs_weights`). See GraphfloodVanillaMFD's own
docstring for why `count`/`barrier` are re-seeded via a direct cupy host
write every step rather than a compiled kernel.

Scope of this cut
--------------------
Only `accum_method="atomic"` is implemented for kind="vanilla_sfd" -
"rake_compress"/"pointer_jump_push" need their own extra scratch buffers
threaded through this factory's own signature, not done in this pass.

Author: B.G (08/2026)
"""

import math
from dataclasses import dataclass
from types import MappingProxyType

from ..core import Backend, RoutineBuilder, require_backend
from ..flow import (
    bind_depression_solver,
    bind_fill_reconstruct_solver,
    make_accumulation,
    make_depression_solver,
    make_depressions,
    make_fill_reconstruct,
    make_fill_reconstruct_solver,
    make_receivers,
)

_KINDS = frozenset({"vanilla_sfd", "unstable", "vanilla_mfd"})
_TOPOLOGY_NN = {"D4": 4, "D8": 8}
_FILL_METHODS = frozenset({"jump", "reconstruct"})
_ACCUM_METHODS = frozenset({"atomic"})
_DEP_METHODS = frozenset({"vanilla", "optimized"})
_OUTLET_BEHAVIORS = frozenset({"fixed_h", "free", "fixed_s"})


def _core_blocks_for(be: Backend):
    if be.family == "closure":
        from . import _closure_core as blocks
    elif be.family == "cupy":
        from . import _cupy_core as blocks
    else:
        raise ValueError(f"make_graphflood: unsupported backend family {be.family!r}")
    return blocks


def _require(label: str, **buffers) -> None:
    missing = sorted(name for name, buf in buffers.items() if buf is None)
    if missing:
        raise ValueError(f"make_graphflood: {label} requires {missing}")


def _compile_bound(bound, be: Backend, **launch):
    """Compile one bound structure and release its top-level binding hold.

    Author: B.G (09/2026)
    """
    compiled = bound.compile(be, **launch)
    bound.close()
    return compiled


class GraphfloodVanillaSFD:
    """
    Host-orchestrated wrapper around the compiled pieces one GraphFlood
    vanilla-SFD timestep needs - mirrors ../ops/__init__.py's Scan/Reduce
    shape (a plain python object over already-compiled kernels/sequences),
    not a stateful context class: every compiled member here came from an
    established factory (make_receivers/make_accumulation/
    make_depression_solver/make_fill_reconstruct_solver) or this package's
    own small core kernels, already built/bound/compiled by make_graphflood
    before construction. `.step()` runs one timestep; call it in a python
    loop.

    Author: B.G (08/2026)
    """

    def __init__(self, *, fill_method: str, receivers=None, minima_solver, make_surface=None,
                 h_from_filled=None, reset_counters=None, reset_queued_gen=None, q_init=None, accum, core):
        self.fill_method = fill_method
        self._receivers = receivers
        self._minima_solver = minima_solver
        self._make_surface = make_surface
        self._h_from_filled = h_from_filled
        self._reset_counters = reset_counters
        self._reset_queued_gen = reset_queued_gen
        self._q_init = q_init
        self._accum = accum
        self._core = core

    def step(self) -> None:
        """
        Run one GraphFlood timestep in place - see the module docstring for
        the exact kernel order per `fill_method`.

        Author: B.G (08/2026)
        """
        if self.fill_method == "jump":
            self._receivers()
            self._minima_solver()
        else:
            self._make_surface()
            self._reset_counters()
            self._reset_queued_gen()
            self._minima_solver()
            self._h_from_filled()
        if self._q_init is not None:
            self._q_init()
        self._accum()
        self._core()

    def __call__(self) -> None:
        """Alias for step()."""
        self.step()

    def close(self) -> None:
        """Release every compiled step's bindings. Author: B.G (09/2026)"""
        for step in (self._receivers, self._minima_solver, self._make_surface,
                     self._h_from_filled, self._reset_counters, self._reset_queued_gen,
                     self._q_init, self._accum, self._core):
            if step is not None:
                step.close()


class GraphfloodUnstable:
    """
    Host-orchestrated wrapper around kind="unstable"'s single compiled
    routine (distribute[/distribute_zero+distribute_route] -> copy_q ->
    compute_qo -> apply_divergence, one RoutineBuilder, no host decision
    between steps - see make_graphflood's own module docstring). `.step()`
    runs one timestep; call it in a python loop.

    Author: B.G (08/2026)
    """

    def __init__(self, routine):
        self._routine = routine

    def step(self) -> None:
        """Run one GraphFlood timestep in place. Author: B.G (08/2026)"""
        self._routine()

    def __call__(self) -> None:
        """Alias for step()."""
        self.step()

    def close(self) -> None:
        """Release the compiled routine's bindings. Author: B.G (09/2026)"""
        self._routine.close()


class GraphfloodVanillaMFD:
    """
    Host-orchestrated wrapper around kind="vanilla_mfd"'s per-step pieces -
    fill-by-reconstruction (shared with kind="vanilla_sfd"'s
    fill_method="reconstruct"), "reconstruct_epsilon"'s hop-distance
    perturbation pass (_cupy_reconstruct_epsilon.py - see its own module
    docstring for why plain `filled` breaks MFD topology across a flat
    resolved depression, and why the `dist` it builds is passed to
    dirs_weights as a separate tie-break carrier), this package's own MFD topology
    construction (_cupy_mfd_topology.py), ../flow's persistent_mfd
    accumulation (_cupy_mfd_accum.py) and the same compute_qo/
    apply_divergence core every kind uses - see make_graphflood's own
    module docstring for the full per-step order. cupy-only: `count`/
    `barrier` are re-seeded from `init_frontier_mfd`'s own host-side cupy
    indexing every step (indegree/dirs/mfd_w are rebuilt from scratch every
    step too, since the surface changes every step), which only exists as
    a raw cupy operation, not a compiled kernel this wrapper could run on
    another backend. `.step()` runs one timestep; call it in a python loop.

    Author: B.G (08/2026)
    """

    def __init__(
        self, *, make_surface, reset_counters, reset_queued_gen, minima_solver, h_from_filled,
        hops_init, hops_jump_fwd, hops_jump_bwd, hops_rounds,
        indegree_reset, dirs_weights, indegree_count, indegree, frontier0, count, barrier,
        init_frontier_mfd, q_init, accum, core,
    ):
        self._make_surface = make_surface
        self._reset_counters = reset_counters
        self._reset_queued_gen = reset_queued_gen
        self._minima_solver = minima_solver
        self._h_from_filled = h_from_filled
        self._hops_init = hops_init
        self._hops_jump_fwd = hops_jump_fwd
        self._hops_jump_bwd = hops_jump_bwd
        self._hops_rounds = hops_rounds
        self._indegree_reset = indegree_reset
        self._dirs_weights = dirs_weights
        self._indegree_count = indegree_count
        self._indegree = indegree
        self._frontier0 = frontier0
        self._count = count
        self._barrier = barrier
        self._init_frontier_mfd = init_frontier_mfd
        self._q_init = q_init
        self._accum = accum
        self._core = core

    def step(self) -> None:
        """Run one GraphFlood timestep in place. Author: B.G (08/2026)"""
        self._make_surface()
        self._reset_counters()
        self._reset_queued_gen()
        self._minima_solver()
        self._h_from_filled()
        self._hops_init()
        # hops_rounds is always even (see make_graphflood's own computation),
        # so alternating fwd/bwd this many times always ends back in the
        # primary dist/anc buffers - see build_hops_jump's own docstring for
        # why this ping-pong exists (in-place pointer-jumping races on `+=`).
        for _ in range(self._hops_rounds // 2):
            self._hops_jump_fwd()
            self._hops_jump_bwd()
        self._indegree_reset()
        self._dirs_weights()
        self._indegree_count()
        n0 = self._init_frontier_mfd(self._indegree, self._frontier0)
        self._count[0:1] = n0
        self._count[1:2] = 0
        self._barrier[0:1] = 0
        self._q_init()
        self._accum()
        self._core()

    def __call__(self) -> None:
        """Alias for step()."""
        self.step()

    def close(self) -> None:
        """Release every compiled step's bindings. Author: B.G (09/2026)"""
        for step in (self._make_surface, self._reset_counters, self._reset_queued_gen,
                     self._minima_solver, self._h_from_filled, self._hops_init,
                     self._hops_jump_fwd, self._hops_jump_bwd, self._indegree_reset,
                     self._dirs_weights, self._indegree_count, self._q_init,
                     self._accum, self._core):
            step.close()


def _compile_graphflood(
    be: Backend,
    grid,
    grid_params: dict,
    *,
    kind: str = "vanilla_sfd",
    n_flat: int,
    nx: int,
    ny: int,
    z,
    h,
    Q_in,
    Qo,
    source_p,
    manning_p,
    friction_exponent_p,
    dt_p,
    gf_min_increment_p,
    boundary_h_p=None,
    boundary_slope_p=None,
    fill_method: str = "jump",
    accum_method: str = "atomic",
    depression_method: str = "optimized",
    topology: str = "D8",
    diagonal_partition_correction: bool = True,
    friction_law: str = "manning",
    outlet_behavior: str = "fixed_h",
    block_size: int = 256,
    rec=None,
    ndep_p=None,
    bid=None,
    rec_jump=None,
    z_prime=None,
    is_border=None,
    basin_saddle=None,
    basin_saddlenode=None,
    outlet=None,
    rerouted=None,
    tag=None,
    tag_alt=None,
    rec_scratch=None,
    basin_route=None,
    b_rcv=None,
    surface=None,
    filled=None,
    parent=None,
    frontier=None,
    counters=None,
    queued_gen=None,
    pass_p=None,
    active_p=None,
    max_passes=None,
    Q_next=None,
    dirs=None,
    mfd_w=None,
    indegree=None,
    frontier0=None,
    frontier1=None,
    count=None,
    barrier=None,
    dist=None,
    anc=None,
    dist2=None,
    anc2=None,
):
    """
    Build, bind and compile one GraphFlood timestep, returning a ready-to-
    call GraphfloodVanillaSFD (kind="vanilla_sfd") or GraphfloodUnstable
    (kind="unstable"). See the module docstring for the per-step order and
    for what `fill_method="jump"` vs `"reconstruct"` each need under
    kind="vanilla_sfd".

    kind="unstable" bypasses routing/local-minima resolution/accumulation
    entirely - `fill_method`/`accum_method`/`depression_method` and every
    jump/reconstruct-only buffer below are ignored. It only needs `z`, `h`,
    `Q_in`, `Qo`, `Q_next` (all n_flat-sized), `source_p`/`manning_p`/
    `friction_exponent_p`/`dt_p`/`gf_min_increment_p` (and `boundary_h_p`/
    `boundary_slope_p` per `outlet_behavior`, same as kind="vanilla_sfd")
    plus `topology`/`diagonal_partition_correction`/`friction_law`. `Q_next`
    is this kind's own scratch buffer (build_distribute's output before
    build_copy_q folds it back into `Q_in`) - required, and only used, here.

    Every array argument (`z`, `h`, `Q_in`, `Qo`, `Q_next`, `rec`, ...) is a
    raw device buffer (a DataHandle's `.array`), n_flat-sized, caller-
    allocated - this factory allocates nothing, matching
    make_depression_solver/make_fill_reconstruct_solver. `z`/`h` are read/
    written in place; `Q_in`/`Qo`/`Q_next` are scratch this factory owns the
    meaning of but not the storage.

    `source_p`/`manning_p`/`friction_exponent_p`/`dt_p`/`gf_min_increment_p`
    (and `boundary_h_p` when `outlet_behavior="fixed_h"`) are caller-
    allocated Parameters (any mode - const, scalar or field all work) bound
    into SOURCE/MANNING/EXPO/DT/GF_MIN_INCREMENT (and BOUNDARY_H)
    respectively - rain, friction and timestep are all params, per the
    module docstring; nothing here hardcodes a value or a unit conversion
    the caller cannot override. `source_p` is Q **per cell** (m^3/s), not a
    bare rate - apply_divergence's `(Q_in - Qo)/area*dt` compares it
    directly against `Qo`, which compute_qo's friction law already produces
    in m^3/s (build_friction_qo's own `* DX` term). A caller converting from
    a rain rate (m/s or mm/h) must multiply by cell area (`DX**2`) before
    binding it here - forgetting this under-scales `Q_in` by a factor of
    `DX**2` (invisible on any grid built with `DX=1`, silently wrong on a
    real DEM's actual cell size).

    fill_method="jump" additionally requires `rec`, `ndep_p`, `bid`,
    `rec_jump`, `z_prime`, `is_border`, `basin_saddle`, `basin_saddlenode`,
    `outlet`, `rerouted` (all n_flat-sized, i32 unless noted - see
    make_depression_solver's own docstring for exact dtypes/caller-side init
    needs). fill_method="reconstruct" additionally requires `surface`,
    `filled`, `parent`, `frontier` (2*n_flat,), `counters`, `queued_gen`,
    `pass_p`, `active_p` (see make_fill_reconstruct_solver's own docstring
    for shapes/caller-side init needs); `rec` is not used in this path.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".
    grid : FrozenGroup
        ../grid's make_grid_group result.
    grid_params : dict
        ../grid's make_grid_parameters result, backing the same grid.
    kind : str, optional
        "vanilla_sfd" (default), "unstable" or "vanilla_mfd" (cupy-only) -
        see above.
    n_flat, nx, ny : int
        `nx`/`ny` are unused under kind="unstable".
    z, h, Q_in, Qo : DataHandle
    Q_next : DataHandle, optional
        Required iff kind="unstable" - see above.
    source_p, manning_p, friction_exponent_p, dt_p, gf_min_increment_p : Parameter
    boundary_h_p : Parameter, optional
        Required iff `outlet_behavior="fixed_h"` (the default) - see
        "Outlet behaviour" in the module docstring.
    boundary_slope_p : Parameter, optional
        Required iff `outlet_behavior="fixed_s"` - see "Outlet behaviour"
        in the module docstring.
    fill_method : str, optional
        "jump" (default) or "reconstruct".
    accum_method : str, optional
        "atomic" (only value implemented so far).
    depression_method : str, optional
        "vanilla" or "optimized" (default) - only used when
        fill_method="jump".
    topology : str, optional
        "D4" or "D8" (default) - must match `grid`.
    diagonal_partition_correction : bool, optional
        Default True.
    friction_law : str, optional
        Passed to build_friction_qo (default "manning").
    outlet_behavior : str, optional
        "fixed_h" (default), "free" or "fixed_s" - see "Outlet behaviour"
        in the module docstring.
    block_size : int, optional
        cupy CUDA block size (default 256); unused on taichi/quadrants.
    rec, ndep_p, bid, rec_jump, z_prime, is_border, basin_saddle,
    basin_saddlenode, outlet : DataHandle/Parameter, optional
        Required when fill_method="jump" (always used with reroute="carve"
        internally - see above).
    tag, tag_alt, rec_scratch, rerouted : DataHandle, optional
        Required when fill_method="jump" and depression_method="vanilla" -
        see above; unused (and not required) under the default
        depression_method="optimized".
    surface, filled, parent, frontier, counters, queued_gen, pass_p,
    active_p : DataHandle/Parameter, optional
        Required when fill_method="reconstruct", or when kind="vanilla_mfd"
        (always fills by reconstruction there - see above).
    max_passes : int, optional
        Forwarded to make_fill_reconstruct_solver.
    dirs, mfd_w, indegree, frontier0, frontier1, count, barrier : DataHandle, optional
        Required iff kind="vanilla_mfd" - `dirs` u8 (n_flat,), `mfd_w` f32
        (n_flat * n_neighbours,) where n_neighbours is 4/8 for D4/D8,
        `indegree` i32 (n_flat,), `frontier0`/`frontier1` i32 (n_flat,)
        each, `count` i32 (2,), `barrier` u32 (1,) - all rebuilt from
        scratch every step, no caller-side init needed beyond allocation.
    dist, anc, dist2, anc2 : DataHandle, optional
        Required iff kind="vanilla_mfd" - "reconstruct_epsilon"'s own
        scratch (_cupy_reconstruct_epsilon.py): `dist`/`dist2` f32 (n_flat,)
        (a self-scaling, per-cell-ULP cumulative perturbation - see
        build_hops_init's own docstring for why this is float, not a hop
        count), `anc`/`anc2` i32 (n_flat,) - `dist2`/`anc2` are the
        double-buffering partner build_hops_jump's ping-pong needs (see its
        own docstring for why in-place pointer-jumping isn't safe here).
        `dist` is handed straight to dirs_weights as its flat-tie-break
        carrier; there is no separate perturbed-elevation array. All rebuilt
        from scratch every step, no caller-side init needed beyond allocation.

    Returns
    -------
    GraphfloodVanillaSFD, GraphfloodUnstable or GraphfloodVanillaMFD
        Per `kind`.

    Raises
    ------
    ValueError
        Unrecognised `kind`/`fill_method`/`accum_method`/`depression_method`/
        `outlet_behavior`; `kind="vanilla_mfd"` on a non-cupy `backend`;
        `boundary_h_p` missing under `outlet_behavior="fixed_h"`;
        `boundary_slope_p` missing under `outlet_behavior="fixed_s"`; or a
        buffer required by the selected `kind`/`fill_method` is missing.

    Author: B.G (08/2026)
    """
    be = require_backend(be)
    backend = be.name
    if kind not in _KINDS:
        raise ValueError(f"make_graphflood: kind must be one of {sorted(_KINDS)}, got {kind!r}")
    if outlet_behavior not in _OUTLET_BEHAVIORS:
        raise ValueError(
            f"make_graphflood: outlet_behavior must be one of {sorted(_OUTLET_BEHAVIORS)}, got {outlet_behavior!r}"
        )
    if outlet_behavior == "fixed_h":
        _require("outlet_behavior='fixed_h'", boundary_h_p=boundary_h_p)
    if outlet_behavior == "fixed_s":
        _require("outlet_behavior='fixed_s'", boundary_slope_p=boundary_slope_p)

    closure = be.family == "closure"
    launch = {}
    core_blocks = _core_blocks_for(be)
    if closure:
        backend_mod = be.module

    if kind == "unstable":
        _require("kind='unstable'", Q_next=Q_next)
        if closure:
            distribute_fk = core_blocks.build_distribute(
                backend=backend, backend_mod=backend_mod, grid=grid, topology=topology,
                diagonal_partition_correction=diagonal_partition_correction,
            )
            copy_q_fk = core_blocks.build_copy_q(backend=backend, backend_mod=backend_mod)
            compute_qo_fk = core_blocks.build_compute_qo(
                backend=backend, backend_mod=backend_mod, grid=grid, topology=topology,
                diagonal_partition_correction=diagonal_partition_correction, law=friction_law,
                outlet_behavior=outlet_behavior,
            )
            apply_div_fk = core_blocks.build_apply_divergence(
                backend=backend, backend_mod=backend_mod, grid=grid, outlet_behavior=outlet_behavior,
            )
            rb = RoutineBuilder()
            rb.step("distribute", distribute_fk)
            rb.step("copy_q", copy_q_fk)
            distribute_steps = ("distribute",)
        else:
            distribute_fks = core_blocks.build_distribute(
                grid=grid, n_flat=n_flat, topology=topology,
                diagonal_partition_correction=diagonal_partition_correction,
            )
            copy_q_fk = core_blocks.build_copy_q(n_flat=n_flat)
            compute_qo_fk = core_blocks.build_compute_qo(
                grid=grid, n_flat=n_flat, topology=topology,
                diagonal_partition_correction=diagonal_partition_correction, law=friction_law,
                outlet_behavior=outlet_behavior,
            )
            apply_div_fk = core_blocks.build_apply_divergence(grid=grid, n_flat=n_flat, outlet_behavior=outlet_behavior)
            rb = RoutineBuilder()
            rb.step("distribute_zero", distribute_fks["zero"])
            rb.step("distribute_route", distribute_fks["route"])
            rb.step("copy_q", copy_q_fk)
            distribute_steps = ("distribute_zero", "distribute_route")

        rb.step("compute_qo", compute_qo_fk)
        rb.step("apply_divergence", apply_div_fk)
        frozen = rb.freeze()
        bound = frozen.build()

        for step in distribute_steps:
            bound.bind_leaf({"z": z, "h": h, "Q_in": Q_in, "Q_next": Q_next}, prefix=(step,))
            bound.bind_leaf({"SOURCE": source_p, "GF_MIN_INCREMENT": gf_min_increment_p}, prefix=(step,))
            bound.bind_leaf(grid_params, prefix=(step,))
        bound.bind(("copy_q", "Q_next"), Q_next)
        bound.bind(("copy_q", "Q_in"), Q_in)
        bound.bind(("compute_qo", "z"), z)
        bound.bind(("compute_qo", "h"), h)
        bound.bind(("compute_qo", "Qo"), Qo)
        bound.bind(("compute_qo", "friction", "MANNING"), manning_p)
        bound.bind(("compute_qo", "friction", "EXPO"), friction_exponent_p)
        if outlet_behavior == "fixed_s":
            bound.bind(("compute_qo", "BOUNDARY_SLOPE"), boundary_slope_p)
        bound.bind_leaf(grid_params, prefix=("compute_qo",))
        bound.bind(("apply_divergence", "h"), h)
        bound.bind(("apply_divergence", "Q_in"), Q_in)
        bound.bind(("apply_divergence", "Qo"), Qo)
        bound.bind(("apply_divergence", "DT"), dt_p)
        bound.bind(("apply_divergence", "GF_MIN_INCREMENT"), gf_min_increment_p)
        if outlet_behavior == "fixed_h":
            bound.bind(("apply_divergence", "BOUNDARY_H"), boundary_h_p)
        bound.bind_leaf(grid_params, prefix=("apply_divergence",))
        routine = _compile_bound(bound, be, **launch)
        return GraphfloodUnstable(routine)

    if kind == "vanilla_mfd":
        if backend != "cupy":
            raise ValueError("make_graphflood: kind='vanilla_mfd' is cupy-only")
        _require(
            "kind='vanilla_mfd'", surface=surface, filled=filled, parent=parent, frontier=frontier,
            counters=counters, queued_gen=queued_gen, pass_p=pass_p, active_p=active_p,
            dirs=dirs, mfd_w=mfd_w, indegree=indegree, frontier0=frontier0, frontier1=frontier1,
            count=count, barrier=barrier, dist=dist, anc=anc, dist2=dist2, anc2=anc2,
        )
        from ..flow._cupy_mfd_accum import build_persistent_mfd, init_frontier_mfd, persistent_grid_block
        from . import _cupy_mfd_topology, _cupy_reconstruct_epsilon

        make_surface_fk = core_blocks.build_make_surface(n_flat=n_flat)
        ms_bound = make_surface_fk.build()
        ms_bound.bind("z", z)
        ms_bound.bind("h", h)
        ms_bound.bind("surface", surface)
        make_surface_kernel = _compile_bound(ms_bound, be, **launch)

        h_from_filled_fk = core_blocks.build_h_from_filled(n_flat=n_flat)
        hf_bound = h_from_filled_fk.build()
        hf_bound.bind("z", z)
        hf_bound.bind("filled", filled)
        hf_bound.bind("h", h)
        h_from_filled_kernel = _compile_bound(hf_bound, be, **launch)

        resolved_max_passes = max_passes if max_passes is not None else 4 * max(int(nx), int(ny))
        reset_fks = core_blocks.build_reset_reconstruct_scratch(n_flat=n_flat, counters_size=resolved_max_passes + 2)
        rc_bound = reset_fks["counters"].build()
        rc_bound.bind("counters", counters)
        reset_counters_kernel = _compile_bound(rc_bound, be, **launch)
        rq_bound = reset_fks["queued_gen"].build()
        rq_bound.bind("queued_gen", queued_gen)
        reset_queued_gen_kernel = _compile_bound(rq_bound, be, **launch)

        recon = make_fill_reconstruct(be, grid, nx=nx, ny=ny)
        recon_frozen, _ = make_fill_reconstruct_solver(
            be, recon, grid_params, pass_p=pass_p, active_p=active_p,
            n_flat=n_flat, nx=nx, ny=ny, block_size=block_size, max_passes=max_passes,
        )
        recon_bound = bind_fill_reconstruct_solver(
            recon_frozen, grid_params, z=surface, filled=filled, parent=parent,
            frontier=frontier, counters=counters, queued_gen=queued_gen,
            pass_p=pass_p, active_p=active_p,
        )
        minima_solver = _compile_bound(recon_bound, be, **launch)

        # "reconstruct_epsilon": accumulate a per-cell `dist` perturbation
        # along the `parent` chain (hops-to-outlet, ULP-scaled), then feed it
        # alongside real `filled` to dirs_weights, which breaks the flat-lake
        # tie inside the slope helper's own additive `h` term - see
        # _cupy_reconstruct_epsilon.py's own module docstring for why plain
        # `filled` dead-ends MFD topology across a flat resolved depression
        # and why `dist` must never be folded up into `filled` to fix it.
        hops_init_fk = _cupy_reconstruct_epsilon.build_hops_init(n_flat=n_flat)
        hi_bound = hops_init_fk.build()
        hi_bound.bind("parent", parent)
        hi_bound.bind("filled", filled)
        hi_bound.bind("dist", dist)
        hi_bound.bind("anc", anc)
        hops_init_kernel = _compile_bound(hi_bound, be, **launch)

        hops_jump_fk = _cupy_reconstruct_epsilon.build_hops_jump(n_flat=n_flat)
        hj_fwd_bound = hops_jump_fk.build()
        hj_fwd_bound.bind("dist_in", dist)
        hj_fwd_bound.bind("anc_in", anc)
        hj_fwd_bound.bind("dist_out", dist2)
        hj_fwd_bound.bind("anc_out", anc2)
        hops_jump_fwd_kernel = _compile_bound(hj_fwd_bound, be, **launch)

        hj_bwd_bound = hops_jump_fk.build()
        hj_bwd_bound.bind("dist_in", dist2)
        hj_bwd_bound.bind("anc_in", anc2)
        hj_bwd_bound.bind("dist_out", dist)
        hj_bwd_bound.bind("anc_out", anc)
        hops_jump_bwd_kernel = _compile_bound(hj_bwd_bound, be, **launch)

        # rounded up to even so alternating fwd/bwd always ends back in the
        # primary dist/anc buffers - see build_hops_jump's own docstring.
        hops_rounds = math.ceil(math.log2(max(2, n_flat))) + 1
        if hops_rounds % 2 != 0:
            hops_rounds += 1

        topo = _cupy_mfd_topology.build_mfd_topology(
            grid=grid, n_flat=n_flat, topology=topology, diagonal_partition_correction=diagonal_partition_correction,
        )
        dw_bound = topo["dirs_weights"].build()
        dw_bound.bind("filled", filled)
        dw_bound.bind("dist", dist)
        dw_bound.bind("dirs", dirs)
        dw_bound.bind("mfd_w", mfd_w)
        dw_bound.bind_leaf(grid_params)
        dirs_weights_kernel = _compile_bound(dw_bound, be, **launch)

        ir_bound = topo["indegree_reset"].build()
        ir_bound.bind("indegree", indegree)
        indegree_reset_kernel = _compile_bound(ir_bound, be, **launch)

        ic_bound = topo["indegree_count"].build()
        ic_bound.bind("dirs", dirs)
        ic_bound.bind("indegree", indegree)
        ic_bound.bind_leaf(grid_params)
        indegree_count_kernel = _compile_bound(ic_bound, be, **launch)

        nn = _TOPOLOGY_NN[topology]
        persistent = build_persistent_mfd(grid=grid, n_flat=n_flat, n_neighbours=nn)
        qi_bound = persistent["q_init"].build()
        qi_bound.bind("SOURCE", source_p)
        qi_bound.bind("accum", Q_in)
        qi_bound.bind_leaf(grid_params, prefix=("grid",))
        persistent_q_init_kernel = _compile_bound(qi_bound, be, **launch)

        pa_bound = persistent["accum"].build()
        pa_bound.bind("frontier0", frontier0)
        pa_bound.bind("frontier1", frontier1)
        pa_bound.bind("count", count)
        pa_bound.bind("barrier", barrier)
        pa_bound.bind("dirs", dirs)
        pa_bound.bind("mfd_w", mfd_w)
        pa_bound.bind("accum", Q_in)
        pa_bound.bind("indegree", indegree)
        pa_bound.bind_leaf(grid_params)
        pgrid, pblock = persistent_grid_block()
        persistent_accum_kernel = _compile_bound(pa_bound, be, grid=pgrid, block=pblock)

        compute_qo_fk = core_blocks.build_compute_qo(
            grid=grid, n_flat=n_flat, topology=topology,
            diagonal_partition_correction=diagonal_partition_correction, law=friction_law,
            outlet_behavior=outlet_behavior,
        )
        apply_div_fk = core_blocks.build_apply_divergence(grid=grid, n_flat=n_flat, outlet_behavior=outlet_behavior)
        core_frozen = RoutineBuilder().step("compute_qo", compute_qo_fk).step("apply_divergence", apply_div_fk).freeze()
        core_bound = core_frozen.build()
        core_bound.bind(("compute_qo", "z"), z)
        core_bound.bind(("compute_qo", "h"), h)
        core_bound.bind(("compute_qo", "Qo"), Qo)
        core_bound.bind(("compute_qo", "friction", "MANNING"), manning_p)
        core_bound.bind(("compute_qo", "friction", "EXPO"), friction_exponent_p)
        if outlet_behavior == "fixed_s":
            core_bound.bind(("compute_qo", "BOUNDARY_SLOPE"), boundary_slope_p)
        core_bound.bind_leaf(grid_params, prefix=("compute_qo",))
        core_bound.bind(("apply_divergence", "h"), h)
        core_bound.bind(("apply_divergence", "Q_in"), Q_in)
        core_bound.bind(("apply_divergence", "Qo"), Qo)
        core_bound.bind(("apply_divergence", "DT"), dt_p)
        core_bound.bind(("apply_divergence", "GF_MIN_INCREMENT"), gf_min_increment_p)
        if outlet_behavior == "fixed_h":
            core_bound.bind(("apply_divergence", "BOUNDARY_H"), boundary_h_p)
        core_bound.bind_leaf(grid_params, prefix=("apply_divergence",))
        core_kernel = _compile_bound(core_bound, be, **launch)

        return GraphfloodVanillaMFD(
            make_surface=make_surface_kernel, reset_counters=reset_counters_kernel,
            reset_queued_gen=reset_queued_gen_kernel, minima_solver=minima_solver,
            h_from_filled=h_from_filled_kernel,
            hops_init=hops_init_kernel, hops_jump_fwd=hops_jump_fwd_kernel, hops_jump_bwd=hops_jump_bwd_kernel,
            hops_rounds=hops_rounds,
            indegree_reset=indegree_reset_kernel,
            dirs_weights=dirs_weights_kernel, indegree_count=indegree_count_kernel,
            indegree=indegree, frontier0=frontier0, count=count, barrier=barrier,
            init_frontier_mfd=init_frontier_mfd, q_init=persistent_q_init_kernel,
            accum=persistent_accum_kernel, core=core_kernel,
        )

    if fill_method not in _FILL_METHODS:
        raise ValueError(f"make_graphflood: fill_method must be one of {sorted(_FILL_METHODS)}, got {fill_method!r}")
    if accum_method not in _ACCUM_METHODS:
        raise ValueError(f"make_graphflood: accum_method must be one of {sorted(_ACCUM_METHODS)}, got {accum_method!r}")
    if depression_method not in _DEP_METHODS:
        raise ValueError(
            f"make_graphflood: depression_method must be one of {sorted(_DEP_METHODS)}, got {depression_method!r}"
        )

    closure = be.family == "closure"
    launch = {}
    core_blocks = _core_blocks_for(be)
    if closure:
        backend_mod = be.module

    # ------------------------------------------------------------------
    # 1. routing + local-minima resolution
    # ------------------------------------------------------------------
    receivers_kernel = None
    make_surface_kernel = None
    h_from_filled_kernel = None
    reset_counters_kernel = None
    reset_queued_gen_kernel = None
    q_init_kernel = None
    rec_for_accum = rec

    if fill_method == "jump":
        _require(
            "fill_method='jump'", rec=rec, ndep_p=ndep_p, bid=bid, rec_jump=rec_jump, z_prime=z_prime,
            is_border=is_border, basin_saddle=basin_saddle, basin_saddlenode=basin_saddlenode,
            outlet=outlet, b_rcv=b_rcv, basin_route=basin_route,
        )
        if depression_method == "vanilla":
            _require(
                "fill_method='jump', depression_method='vanilla'",
                tag=tag, tag_alt=tag_alt, rec_scratch=rec_scratch, rerouted=rerouted,
            )
        recv = make_receivers(
            be, grid, topology=topology, mode="steepest",
            diagonal_partition_correction=diagonal_partition_correction, h_aware=True,
        )
        recv_bound = recv["receivers"].build()
        recv_bound.bind_leaf(grid_params)
        recv_bound.bind("z", z)
        recv_bound.bind("h", h)
        recv_bound.bind("rec", rec)
        receivers_kernel = _compile_bound(recv_bound, be, **launch)

        deps = make_depressions(be, grid, ndep_p, method=depression_method, reroute="carve", n_flat=n_flat)
        deps_frozen, _ = make_depression_solver(
            be, deps, grid_params, method=depression_method, reroute="carve",
            n_flat=n_flat, block_size=block_size,
        )
        deps_bound = bind_depression_solver(
            deps_frozen, grid_params, ndep_p=ndep_p, method=depression_method, reroute="carve",
            rec=rec, z=z, bid=bid, rec_jump=rec_jump, z_prime=z_prime, is_border=is_border,
            basin_saddle=basin_saddle, basin_saddlenode=basin_saddlenode, outlet=outlet,
            rerouted=rerouted, tag=tag, tag_alt=tag_alt, rec_scratch=rec_scratch,
            basin_route=basin_route, b_rcv=b_rcv,
        )
        minima_solver = _compile_bound(deps_bound, be, **launch)
        rec_for_accum = rec
    else:
        _require(
            "fill_method='reconstruct'", surface=surface, filled=filled, parent=parent, frontier=frontier,
            counters=counters, queued_gen=queued_gen, pass_p=pass_p, active_p=active_p,
        )
        if closure:
            make_surface_fk = core_blocks.build_make_surface(backend=backend, backend_mod=backend_mod)
            h_from_filled_fk = core_blocks.build_h_from_filled(backend=backend, backend_mod=backend_mod)
        else:
            make_surface_fk = core_blocks.build_make_surface(n_flat=n_flat)
            h_from_filled_fk = core_blocks.build_h_from_filled(n_flat=n_flat)

        ms_bound = make_surface_fk.build()
        ms_bound.bind("z", z)
        ms_bound.bind("h", h)
        ms_bound.bind("surface", surface)
        make_surface_kernel = _compile_bound(ms_bound, be, **launch)

        hf_bound = h_from_filled_fk.build()
        hf_bound.bind("z", z)
        hf_bound.bind("filled", filled)
        hf_bound.bind("h", h)
        h_from_filled_kernel = _compile_bound(hf_bound, be, **launch)

        resolved_max_passes = max_passes if max_passes is not None else 4 * max(int(nx), int(ny))
        if closure:
            reset_fks = core_blocks.build_reset_reconstruct_scratch(backend=backend, backend_mod=backend_mod)
        else:
            reset_fks = core_blocks.build_reset_reconstruct_scratch(
                n_flat=n_flat, counters_size=resolved_max_passes + 2
            )
        rc_bound = reset_fks["counters"].build()
        rc_bound.bind("counters", counters)
        reset_counters_kernel = _compile_bound(rc_bound, be, **launch)
        rq_bound = reset_fks["queued_gen"].build()
        rq_bound.bind("queued_gen", queued_gen)
        reset_queued_gen_kernel = _compile_bound(rq_bound, be, **launch)

        recon = make_fill_reconstruct(be, grid, nx=nx, ny=ny)
        recon_frozen, _ = make_fill_reconstruct_solver(
            be, recon, grid_params, pass_p=pass_p, active_p=active_p,
            n_flat=n_flat, nx=nx, ny=ny, block_size=block_size, max_passes=max_passes,
        )
        recon_bound = bind_fill_reconstruct_solver(
            recon_frozen, grid_params, z=surface, filled=filled, parent=parent,
            frontier=frontier, counters=counters, queued_gen=queued_gen,
            pass_p=pass_p, active_p=active_p,
        )
        minima_solver = _compile_bound(recon_bound, be, **launch)
        rec_for_accum = parent

    # ------------------------------------------------------------------
    # 2. full downstream accumulation
    # ------------------------------------------------------------------
    accum = make_accumulation(be, grid, method="atomic", n_flat=n_flat)
    if "q_init" in accum:
        qi_bound = accum["q_init"].build()
        qi_bound.bind("SOURCE", source_p)
        qi_bound.bind("q", Q_in)
        q_init_kernel = _compile_bound(qi_bound, be, **launch)
    a_bound = accum["accum"].build()
    a_bound.bind("SOURCE", source_p)
    a_bound.bind("rec", rec_for_accum)
    a_bound.bind("q", Q_in)
    accum_kernel = _compile_bound(a_bound, be, **launch)

    # ------------------------------------------------------------------
    # 3. core: compute_qo then apply_divergence
    # ------------------------------------------------------------------
    if closure:
        compute_qo_fk = core_blocks.build_compute_qo(
            backend=backend, backend_mod=backend_mod, grid=grid, topology=topology,
            diagonal_partition_correction=diagonal_partition_correction, law=friction_law,
            outlet_behavior=outlet_behavior,
        )
        apply_div_fk = core_blocks.build_apply_divergence(
            backend=backend, backend_mod=backend_mod, grid=grid, outlet_behavior=outlet_behavior,
        )
    else:
        compute_qo_fk = core_blocks.build_compute_qo(
            grid=grid, n_flat=n_flat, topology=topology,
            diagonal_partition_correction=diagonal_partition_correction, law=friction_law,
            outlet_behavior=outlet_behavior,
        )
        apply_div_fk = core_blocks.build_apply_divergence(grid=grid, n_flat=n_flat, outlet_behavior=outlet_behavior)

    core_frozen = RoutineBuilder().step("compute_qo", compute_qo_fk).step("apply_divergence", apply_div_fk).freeze()
    core_bound = core_frozen.build()
    core_bound.bind(("compute_qo", "z"), z)
    core_bound.bind(("compute_qo", "h"), h)
    core_bound.bind(("compute_qo", "Qo"), Qo)
    core_bound.bind(("compute_qo", "friction", "MANNING"), manning_p)
    core_bound.bind(("compute_qo", "friction", "EXPO"), friction_exponent_p)
    if outlet_behavior == "fixed_s":
        core_bound.bind(("compute_qo", "BOUNDARY_SLOPE"), boundary_slope_p)
    core_bound.bind(("apply_divergence", "h"), h)
    core_bound.bind(("apply_divergence", "Q_in"), Q_in)
    core_bound.bind(("apply_divergence", "Qo"), Qo)
    core_bound.bind(("apply_divergence", "DT"), dt_p)
    core_bound.bind(("apply_divergence", "GF_MIN_INCREMENT"), gf_min_increment_p)
    if outlet_behavior == "fixed_h":
        core_bound.bind(("apply_divergence", "BOUNDARY_H"), boundary_h_p)
    core_bound.bind_leaf(grid_params, prefix=("compute_qo",))
    core_bound.bind_leaf(grid_params, prefix=("apply_divergence",))
    core_kernel = _compile_bound(core_bound, be, **launch)

    return GraphfloodVanillaSFD(
        fill_method=fill_method,
        receivers=receivers_kernel,
        minima_solver=minima_solver,
        make_surface=make_surface_kernel,
        h_from_filled=h_from_filled_kernel,
        reset_counters=reset_counters_kernel,
        reset_queued_gen=reset_queued_gen_kernel,
        q_init=q_init_kernel,
        accum=accum_kernel,
        core=core_kernel,
    )


@dataclass(frozen=True)
class FrozenGraphflood:
    """Immutable GraphFlood recipe, ready for explicit live-state binding.

    GraphFlood contains several independently compiled kernels and host-driven
    sequences, so its inert recipe is a frozen plan rather than a synthetic
    single kernel sequence. It owns no Parameters, DataHandles, bounds, or
    compiled callables.

    Author: B.G (09/2026)
    """

    be: Backend
    grid: object
    config: MappingProxyType


def make_graphflood(
    be: Backend,
    grid,
    *,
    kind: str = "vanilla_sfd",
    n_flat: int,
    nx: int,
    ny: int,
    fill_method: str = "jump",
    accum_method: str = "atomic",
    depression_method: str = "optimized",
    topology: str = "D8",
    diagonal_partition_correction: bool = True,
    friction_law: str = "manning",
    outlet_behavior: str = "fixed_h",
    block_size: int = 256,
    max_passes: int | None = None,
):
    """Build an inert GraphFlood recipe and return ``(recipe, params)``.

    Live Parameters and DataHandles deliberately do not enter this factory.
    Bind them with :func:`bind_graphflood`, which returns the compiled
    GraphFlood timestep callable. GraphFlood has no factory-created
    Parameters, hence the returned parameter mapping is empty.

    Author: B.G (09/2026)
    """
    be = require_backend(be)
    if kind not in _KINDS:
        raise ValueError(f"make_graphflood: kind must be one of {sorted(_KINDS)}, got {kind!r}")
    if fill_method not in _FILL_METHODS:
        raise ValueError(f"make_graphflood: fill_method must be one of {sorted(_FILL_METHODS)}, got {fill_method!r}")
    if accum_method not in _ACCUM_METHODS:
        raise ValueError(f"make_graphflood: accum_method must be one of {sorted(_ACCUM_METHODS)}, got {accum_method!r}")
    if depression_method not in _DEP_METHODS:
        raise ValueError(f"make_graphflood: depression_method must be one of {sorted(_DEP_METHODS)}, got {depression_method!r}")
    if topology not in _TOPOLOGY_NN:
        raise ValueError(f"make_graphflood: topology must be one of {sorted(_TOPOLOGY_NN)}, got {topology!r}")
    if outlet_behavior not in _OUTLET_BEHAVIORS:
        raise ValueError(
            f"make_graphflood: outlet_behavior must be one of {sorted(_OUTLET_BEHAVIORS)}, got {outlet_behavior!r}"
        )
    if kind == "vanilla_mfd" and be.family != "cupy":
        raise ValueError("make_graphflood: kind='vanilla_mfd' is cupy-only")
    config = MappingProxyType({
        "kind": kind,
        "n_flat": int(n_flat), "nx": int(nx), "ny": int(ny),
        "fill_method": fill_method, "accum_method": accum_method,
        "depression_method": depression_method, "topology": topology,
        "diagonal_partition_correction": bool(diagonal_partition_correction),
        "friction_law": friction_law, "outlet_behavior": outlet_behavior,
        "block_size": int(block_size), "max_passes": max_passes,
    })
    return FrozenGraphflood(be, grid, config), {}


def bind_graphflood(frozen: FrozenGraphflood, grid_params: dict, **bindings):
    """Bind live GraphFlood state to an inert recipe and compile its timestep.

    ``bindings`` contains the DataHandles and Parameters documented by the
    selected ``kind``/``fill_method``. The returned object is caller-owned and
    must be closed after use.

    Author: B.G (09/2026)
    """
    if not isinstance(frozen, FrozenGraphflood):
        raise TypeError("bind_graphflood() requires a FrozenGraphflood from make_graphflood()")
    overlap = set(frozen.config).intersection(bindings)
    if overlap:
        raise TypeError(
            "bind_graphflood() received recipe configuration as live binding(s): "
            f"{sorted(overlap)}; pass these to make_graphflood()"
        )
    return _compile_graphflood(frozen.be, frozen.grid, grid_params, **frozen.config, **bindings)
