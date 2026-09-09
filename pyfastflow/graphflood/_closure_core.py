"""
Taichi/Quadrants (closure) block templates behind make_graphflood's per-step
core - compute_qo/apply_divergence (the friction-law update, ported from
../../flood/flood_graphflood_kernels.py's graphflood_core_kernel, split into
two kernels rather than one two-pass kernel - see the module docstring of
../graphflood/__init__.py for why the split alone already avoids the race
the legacy two-pass dh buffer existed for) plus make_surface/h_from_filled
(the small dedicated kernels the "fill on the z+h surface" local-minima
option needs to move an elevation-space fill result back into h).

Author: B.G (08/2026)
"""

from ..core import FrozenKernel, KernelBuilder
from ..flow._closure_receivers import build_distance_slope_helpers
from ..flow._closure_shared import _tensor_annotation
from ._closure_friction import build_friction_qo


_OUTLET_BEHAVIORS = frozenset({"fixed_h", "free", "fixed_s"})


def build_compute_qo(
    *, backend: str, backend_mod, grid, topology: str, diagonal_partition_correction: bool,
    law: str = "manning", outlet_behavior: str = "fixed_h",
) -> FrozenKernel:
    """
    compute_qo FrozenKernel, data args (z, h, Qo): elsewhere Qo[i] =
    friction(h[i], steepest h-aware slope to a valid neighbour) - what
    happens on a can_out node is picked, at build time, by
    `outlet_behavior` (never branched on inside one kernel body - see
    build_receivers' own `mode`/`h_aware` dispatch, ../flow/
    _closure_receivers.py, for the same pattern). All three behaviors reuse
    the exact same `friction` FrozenHelper (build_friction_qo, composed
    once) - it already takes h/slope as plain call arguments (no MANNING/
    EXPO baked in beyond its own wired PARAM slots), so a fixed boundary
    slope is just a different second argument to the same call, not a
    second copy of the friction law:

      "fixed_h": Qo[i] = 0 - unused, apply_divergence(outlet_behavior=
        "fixed_h") overrides h there directly instead of routing it
        through the friction law.
      "free": no special case at all - every node, can_out or not, gets a
        real friction-law Qo from its own steepest h-aware slope; a
        can_out node's "outward" side then behaves like any other open
        boundary, letting whatever water is actually present drain out at
        the rate the friction law gives it rather than pinning h to a
        fixed stage.
      "fixed_s": like "free" (a real friction-law Qo, no h pinning), but a
        can_out node's Qo uses a caller-supplied constant hydraulic slope
        (`BOUNDARY_SLOPE`, wired here) instead of its own locally-computed
        steepest slope - a "normal/uniform flow" outfall, for when the
        real exit slope is known (e.g. a surveyed channel grade) rather
        than well represented by this cell's own local geometry.

    Composes its own `grid`/`slope`/`friction` occurrences, build-phase-
    shared across every grid PARAM name.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    grid : FrozenGroup
    topology : str
        "D4" or "D8".
    diagonal_partition_correction : bool
    law : str, optional
        Friction law passed to build_friction_qo (default "manning").
    outlet_behavior : str, optional
        "fixed_h" (default), "free" or "fixed_s" - must match the value
        passed to build_apply_divergence for the same pipeline.

    Returns
    -------
    FrozenKernel

    Raises
    ------
    ValueError
        If `outlet_behavior` is not recognised.

    Author: B.G (08/2026)
    """
    if outlet_behavior not in _OUTLET_BEHAVIORS:
        raise ValueError(
            f"build_compute_qo: outlet_behavior must be one of {sorted(_OUTLET_BEHAVIORS)}, got {outlet_behavior!r}"
        )
    T = _tensor_annotation(backend_mod, backend)
    slope = build_distance_slope_helpers(
        grid, topology=topology, diagonal_partition_correction=diagonal_partition_correction
    )["slope_from_values_k"]
    friction = build_friction_qo(law, grid)

    if outlet_behavior == "fixed_h":

        def compute_qo_tmpl(ctx, z: T, h: T, Qo: T):
            for i in z:
                if ctx.grid.can_out(i):
                    Qo[i] = 0.0
                    continue
                best_s = 0.0
                nk = ctx.grid.N_NEIGHBOURS.get(0)
                for k in range(nk):
                    j = ctx.grid.neighbour(i, k)
                    if j != -1:
                        s = ctx.slope(z[i], h[i], z[j], h[j], k)
                        if s > best_s:
                            best_s = s
                Qo[i] = ctx.friction(h[i], best_s)

    elif outlet_behavior == "free":

        def compute_qo_tmpl(ctx, z: T, h: T, Qo: T):
            for i in z:
                best_s = 0.0
                nk = ctx.grid.N_NEIGHBOURS.get(0)
                for k in range(nk):
                    j = ctx.grid.neighbour(i, k)
                    if j != -1:
                        s = ctx.slope(z[i], h[i], z[j], h[j], k)
                        if s > best_s:
                            best_s = s
                Qo[i] = ctx.friction(h[i], best_s)

    else:  # "fixed_s"

        def compute_qo_tmpl(ctx, z: T, h: T, Qo: T):
            for i in z:
                if ctx.grid.can_out(i):
                    Qo[i] = ctx.friction(h[i], ctx.BOUNDARY_SLOPE.get(0))
                    continue
                best_s = 0.0
                nk = ctx.grid.N_NEIGHBOURS.get(0)
                for k in range(nk):
                    j = ctx.grid.neighbour(i, k)
                    if j != -1:
                        s = ctx.slope(z[i], h[i], z[j], h[j], k)
                        if s > best_s:
                            best_s = s
                Qo[i] = ctx.friction(h[i], best_s)

    kb = KernelBuilder(compute_qo_tmpl)
    kb.compose("grid", grid).compose("slope", slope).compose("friction", friction)
    kb.share_identical("grid")
    return kb.freeze()


def build_apply_divergence(*, backend: str, backend_mod, grid, outlet_behavior: str = "fixed_h") -> FrozenKernel:
    """
    apply_divergence FrozenKernel, data args (h, Q_in, Qo): interior nodes
    always get h[i] = max(0, h[i] + DT*(Q_in[i] - Qo[i])/DX**2), clamped to
    GF_MIN_INCREMENT away from zero whenever Q_in/Qo disagree in sign of net
    change (ported from graphflood_core_kernel's dh clamp). What happens on
    a can_out node is picked, at build time, by `outlet_behavior` - see
    build_compute_qo's own docstring for the matching Qo-side half of each
    behavior:

      "fixed_h": h[i] = BOUNDARY_H, unconditionally - a Dirichlet stage
        boundary. Wires its own `BOUNDARY_H` PARAM slot.
      "free"/"fixed_s": no special case, identical kernel body for both -
        a can_out node runs through the exact same divergence formula as
        every interior node, against the real Qo compute_qo(outlet_
        behavior="free"|"fixed_s") already computed for it (only compute_qo
        itself distinguishes the two - which slope value fed the friction
        law). Does not wire `BOUNDARY_H` at all - nothing here reads it.

    Composes its own `grid` occurrence for can_out/DX.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    grid : FrozenGroup
    outlet_behavior : str, optional
        "fixed_h" (default), "free" or "fixed_s" - must match the value
        passed to build_compute_qo for the same pipeline.

    Returns
    -------
    FrozenKernel

    Raises
    ------
    ValueError
        If `outlet_behavior` is not recognised.

    Author: B.G (08/2026)
    """
    if outlet_behavior not in _OUTLET_BEHAVIORS:
        raise ValueError(
            f"build_apply_divergence: outlet_behavior must be one of {sorted(_OUTLET_BEHAVIORS)}, "
            f"got {outlet_behavior!r}"
        )
    T = _tensor_annotation(backend_mod, backend)

    if outlet_behavior == "fixed_h":

        def apply_divergence_tmpl(ctx, h: T, Q_in: T, Qo: T):
            dx = ctx.grid.DX.get(0)
            area = dx * dx
            for i in h:
                if ctx.grid.can_out(i):
                    h[i] = ctx.BOUNDARY_H.get(0)
                    continue
                dt = ctx.DT.get(0)
                d = (Q_in[i] - Qo[i]) / area * dt
                min_inc = ctx.GF_MIN_INCREMENT.get(0)
                if Q_in[i] > Qo[i] and d < min_inc:
                    d = min_inc
                elif Qo[i] > Q_in[i] and d > -min_inc:
                    d = -min_inc
                hh = h[i] + d
                h[i] = hh if hh > 0.0 else 0.0

    else:  # "free"

        def apply_divergence_tmpl(ctx, h: T, Q_in: T, Qo: T):
            dx = ctx.grid.DX.get(0)
            area = dx * dx
            for i in h:
                dt = ctx.DT.get(0)
                d = (Q_in[i] - Qo[i]) / area * dt
                min_inc = ctx.GF_MIN_INCREMENT.get(0)
                if Q_in[i] > Qo[i] and d < min_inc:
                    d = min_inc
                elif Qo[i] > Q_in[i] and d > -min_inc:
                    d = -min_inc
                hh = h[i] + d
                h[i] = hh if hh > 0.0 else 0.0

    kb = KernelBuilder(apply_divergence_tmpl)
    kb.compose("grid", grid)
    return kb.freeze()


def build_make_surface(*, backend: str, backend_mod) -> FrozenKernel:
    """
    make_surface FrozenKernel, data args (z, h, surface): surface[i] =
    z[i] + h[i] - the elevation-space input the "reconstruct" local-minima
    option's fill_reconstruct pass runs against.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.

    Returns
    -------
    FrozenKernel

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def make_surface_tmpl(ctx, z: T, h: T, surface: T):
        for i in z:
            surface[i] = z[i] + h[i]

    return KernelBuilder(make_surface_tmpl).freeze()


def build_h_from_filled(*, backend: str, backend_mod) -> FrozenKernel:
    """
    h_from_filled FrozenKernel, data args (z, filled, h): h[i] = max(0,
    filled[i] - z[i]) - pulls the "reconstruct" local-minima option's
    filled elevation-space surface back into a depth field.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.

    Returns
    -------
    FrozenKernel

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def h_from_filled_tmpl(ctx, z: T, filled: T, h: T):
        for i in z:
            hh = filled[i] - z[i]
            h[i] = hh if hh > 0.0 else 0.0

    return KernelBuilder(h_from_filled_tmpl).freeze()


def build_reset_reconstruct_scratch(*, backend: str, backend_mod) -> dict:
    """
    "counters" (data arg (counters,): counters[i] = 0) and "queued_gen"
    (data arg (queued_gen,): queued_gen[i] = -1) FrozenKernels - the
    "zeroed/filled once, before the first call" caller-side init
    make_fill_reconstruct_solver's own docstring documents (../flow/
    __init__.py) is a single-solve contract: calling that solver again on a
    changed surface without re-doing this init reads stale per-pass counts
    and dedup generations left over from the previous call. GraphFlood's
    "reconstruct" fill_method calls the solver once per timestep on a
    surface that changes every timestep, so make_graphflood runs these two
    resets immediately before every such call - see its own module
    docstring.

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.

    Returns
    -------
    dict
        {"counters": FrozenKernel, "queued_gen": FrozenKernel}.

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def reset_counters_tmpl(ctx, counters: T):
        for i in counters:
            counters[i] = 0

    def reset_queued_gen_tmpl(ctx, queued_gen: T):
        for i in queued_gen:
            queued_gen[i] = -1

    return {
        "counters": KernelBuilder(reset_counters_tmpl).freeze(),
        "queued_gen": KernelBuilder(reset_queued_gen_tmpl).freeze(),
    }


def build_distribute(
    *, backend: str, backend_mod, grid, topology: str, diagonal_partition_correction: bool,
) -> FrozenKernel:
    """
    distribute FrozenKernel, data args (z, h, Q_in, Q_next) - the
    graphflood_unstable per-step local redistribution, ported from
    ../../flood/flood_graphflood_kernels.py's distribute_flow_local_kernel:
    no receiver graph, no accumulation, no depression handling - every node
    walks only its own immediate neighbours, splitting its own current
    inflow Q_in[i] across every downslope (h-aware) neighbour in proportion
    to that neighbour's own slope share, then adding the caller's SOURCE
    (rain) into Q_next fresh every step.

    A node with no downslope neighbour (sum_s <= 0 - a local pit under the
    current (z, h)) keeps its own inflow in place (Q_next[i] += qi, so it
    is retried next step once h has risen) and nudges h[i] up by
    GF_MIN_INCREMENT - the same "dig it out gradually over many steps"
    heuristic legacy used, replacing an exact depression solve with
    something the outer step loop converges towards instead. This is what
    makes the method "unstable": no acyclic-graph guarantee, no filled
    surface, just local redistribution repeated every timestep.

    Composes its own `grid`/`slope` occurrences, build-phase-shared across
    every grid PARAM name. Wires its own `SOURCE`/`GF_MIN_INCREMENT` PARAM
    slots (any mode - a caller binds Parameters there after `.build()`).

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.
    grid : FrozenGroup
    topology : str
        "D4" or "D8".
    diagonal_partition_correction : bool

    Returns
    -------
    FrozenKernel

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    slope = build_distance_slope_helpers(
        grid, topology=topology, diagonal_partition_correction=diagonal_partition_correction
    )["slope_from_values_k"]

    def distribute_tmpl(ctx, z: T, h: T, Q_in: T, Q_next: T):
        for i in Q_next:
            Q_next[i] = ctx.SOURCE.get(i)
        for i in Q_in:
            if ctx.grid.can_out(i):
                continue
            qi = Q_in[i]
            if qi <= 0.0:
                continue
            slopes = ctx.bk.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            sum_s = 0.0
            nk = ctx.grid.N_NEIGHBOURS.get(0)
            for k in range(nk):
                j = ctx.grid.neighbour(i, k)
                if j != -1:
                    s = ctx.slope(z[i], h[i], z[j], h[j], k)
                    s = s if s > 0.0 else 0.0
                    slopes[k] = s
                    sum_s += s
            if sum_s <= 0.0:
                ctx.bk.atomic_add(Q_next[i], qi)
                ctx.bk.atomic_add(h[i], ctx.GF_MIN_INCREMENT.get(0))
            else:
                for k in range(nk):
                    j = ctx.grid.neighbour(i, k)
                    if j != -1 and slopes[k] > 0.0:
                        ctx.bk.atomic_add(Q_next[j], qi * slopes[k] / sum_s)

    kb = KernelBuilder(distribute_tmpl)
    kb.compose("grid", grid).compose("slope", slope)
    kb.share_identical("grid")
    return kb.freeze()


def build_copy_q(*, backend: str, backend_mod) -> FrozenKernel:
    """
    copy_q FrozenKernel, data args (Q_next, Q_in): Q_in[i] = Q_next[i] -
    graphflood_unstable's own step boundary between build_distribute's
    output buffer and the next call's input (and the buffer compute_qo/
    apply_divergence read as this step's Q_in).

    Parameters
    ----------
    backend : str
        "taichi" or "quadrants".
    backend_mod
        The bound `ti`/`qd` module.

    Returns
    -------
    FrozenKernel

    Author: B.G (08/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def copy_q_tmpl(ctx, Q_next: T, Q_in: T):
        for i in Q_next:
            Q_in[i] = Q_next[i]

    return KernelBuilder(copy_q_tmpl).freeze()
