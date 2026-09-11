"""GraphFlood shallow-water building blocks.

Build a recipe with :func:`make_graphflood`, bind terrain and model state,
then call the resulting object for each timestep. ``vanilla_sfd`` routes one
receiver per cell, ``unstable`` redistributes flow locally, and CuPy-only
``vanilla_mfd`` splits flow between downslope neighbours.
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
    make_mfd_topology,
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


def _compile_bound(bound, be: Backend):
    """Compile one bound structure and release its binding hold."""
    compiled = bound.compile(be)
    bound.close()
    return compiled


class GraphfloodVanillaSFD:
    """Compiled single-flow GraphFlood timestep.

    Call :meth:`step` once per model timestep; the bound terrain and flow
    arrays are updated in place.
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
        """Run one timestep in place."""
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
        """Release every compiled step's bindings."""
        for step in (self._receivers, self._minima_solver, self._make_surface,
                     self._h_from_filled, self._reset_counters, self._reset_queued_gen,
                     self._q_init, self._accum, self._core):
            if step is not None:
                step.close()


class GraphfloodUnstable:
    """Compiled unstable-flow GraphFlood timestep."""

    def __init__(self, routine):
        self._routine = routine

    def step(self) -> None:
        """Run one GraphFlood timestep in place."""
        self._routine()

    def __call__(self) -> None:
        """Alias for step()."""
        self.step()

    def close(self) -> None:
        """Release the compiled routine's bindings."""
        self._routine.close()


class GraphfloodVanillaMFD:
    """Compiled CuPy multiple-flow GraphFlood timestep.

    The MFD topology and accumulation state are rebuilt at each timestep.
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
        """Run one GraphFlood timestep in place."""
        self._make_surface()
        self._reset_counters()
        self._reset_queued_gen()
        self._minima_solver()
        self._h_from_filled()
        self._hops_init()
        # An even count leaves the final hop state in the primary buffers.
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
        """Release every compiled step's bindings."""
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
    """Bind and compile a GraphFlood timestep.

    This internal builder receives the complete live state. Public callers
    normally use :func:`make_graphflood` followed by :func:`bind_graphflood`.
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
        routine = _compile_bound(bound, be)
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
        from ..flow._cupy_mfd_accum import build_persistent_mfd, init_frontier_mfd
        from . import _cupy_reconstruct_epsilon

        make_surface_fk = core_blocks.build_make_surface(n_flat=n_flat)
        ms_bound = make_surface_fk.build()
        ms_bound.bind("z", z)
        ms_bound.bind("h", h)
        ms_bound.bind("surface", surface)
        make_surface_kernel = _compile_bound(ms_bound, be)

        h_from_filled_fk = core_blocks.build_h_from_filled(n_flat=n_flat)
        hf_bound = h_from_filled_fk.build()
        hf_bound.bind("z", z)
        hf_bound.bind("filled", filled)
        hf_bound.bind("h", h)
        h_from_filled_kernel = _compile_bound(hf_bound, be)

        resolved_max_passes = max_passes if max_passes is not None else 4 * max(int(nx), int(ny))
        reset_fks = core_blocks.build_reset_reconstruct_scratch(n_flat=n_flat, counters_size=resolved_max_passes + 2)
        rc_bound = reset_fks["counters"].build()
        rc_bound.bind("counters", counters)
        reset_counters_kernel = _compile_bound(rc_bound, be)
        rq_bound = reset_fks["queued_gen"].build()
        rq_bound.bind("queued_gen", queued_gen)
        reset_queued_gen_kernel = _compile_bound(rq_bound, be)

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
        minima_solver = _compile_bound(recon_bound, be)

        # Build a path-distance perturbation to resolve flat-lake ties in the
        # MFD topology. Keep it separate from the filled elevation.
        hops_init_fk = _cupy_reconstruct_epsilon.build_hops_init(n_flat=n_flat)
        hi_bound = hops_init_fk.build()
        hi_bound.bind("parent", parent)
        hi_bound.bind("filled", filled)
        hi_bound.bind("dist", dist)
        hi_bound.bind("anc", anc)
        hops_init_kernel = _compile_bound(hi_bound, be)

        hops_jump_fk = _cupy_reconstruct_epsilon.build_hops_jump(n_flat=n_flat)
        hj_fwd_bound = hops_jump_fk.build()
        hj_fwd_bound.bind("dist_in", dist)
        hj_fwd_bound.bind("anc_in", anc)
        hj_fwd_bound.bind("dist_out", dist2)
        hj_fwd_bound.bind("anc_out", anc2)
        hops_jump_fwd_kernel = _compile_bound(hj_fwd_bound, be)

        hj_bwd_bound = hops_jump_fk.build()
        hj_bwd_bound.bind("dist_in", dist2)
        hj_bwd_bound.bind("anc_in", anc2)
        hj_bwd_bound.bind("dist_out", dist)
        hj_bwd_bound.bind("anc_out", anc)
        hops_jump_bwd_kernel = _compile_bound(hj_bwd_bound, be)

        # rounded up to even so alternating fwd/bwd always ends back in the
        # primary dist/anc buffers - see build_hops_jump's own docstring.
        hops_rounds = math.ceil(math.log2(max(2, n_flat))) + 1
        if hops_rounds % 2 != 0:
            hops_rounds += 1

        topo = make_mfd_topology(
            be, grid, method="surface", n_flat=n_flat, topology=topology,
            diagonal_partition_correction=diagonal_partition_correction,
        )
        dw_bound = topo["dirs_weights"].build()
        dw_bound.bind("filled", filled)
        dw_bound.bind("dist", dist)
        dw_bound.bind("dirs", dirs)
        dw_bound.bind("mfd_w", mfd_w)
        dw_bound.bind_leaf(grid_params)
        dirs_weights_kernel = _compile_bound(dw_bound, be)

        ir_bound = topo["indegree_reset"].build()
        ir_bound.bind("indegree", indegree)
        indegree_reset_kernel = _compile_bound(ir_bound, be)

        ic_bound = topo["indegree_count"].build()
        ic_bound.bind("dirs", dirs)
        ic_bound.bind("indegree", indegree)
        ic_bound.bind_leaf(grid_params)
        indegree_count_kernel = _compile_bound(ic_bound, be)

        nn = _TOPOLOGY_NN[topology]
        persistent = build_persistent_mfd(grid=grid, n_flat=n_flat, n_neighbours=nn)
        qi_bound = persistent["q_init"].build()
        qi_bound.bind("SOURCE", source_p)
        qi_bound.bind("accum", Q_in)
        qi_bound.bind_leaf(grid_params, prefix=("grid",))
        persistent_q_init_kernel = _compile_bound(qi_bound, be)

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
        persistent_accum_kernel = _compile_bound(pa_bound, be)

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
        core_kernel = _compile_bound(core_bound, be)

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
        receivers_kernel = _compile_bound(recv_bound, be)

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
        minima_solver = _compile_bound(deps_bound, be)
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
        make_surface_kernel = _compile_bound(ms_bound, be)

        hf_bound = h_from_filled_fk.build()
        hf_bound.bind("z", z)
        hf_bound.bind("filled", filled)
        hf_bound.bind("h", h)
        h_from_filled_kernel = _compile_bound(hf_bound, be)

        resolved_max_passes = max_passes if max_passes is not None else 4 * max(int(nx), int(ny))
        if closure:
            reset_fks = core_blocks.build_reset_reconstruct_scratch(backend=backend, backend_mod=backend_mod)
        else:
            reset_fks = core_blocks.build_reset_reconstruct_scratch(
                n_flat=n_flat, counters_size=resolved_max_passes + 2
            )
        rc_bound = reset_fks["counters"].build()
        rc_bound.bind("counters", counters)
        reset_counters_kernel = _compile_bound(rc_bound, be)
        rq_bound = reset_fks["queued_gen"].build()
        rq_bound.bind("queued_gen", queued_gen)
        reset_queued_gen_kernel = _compile_bound(rq_bound, be)

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
        minima_solver = _compile_bound(recon_bound, be)
        rec_for_accum = parent

    # ------------------------------------------------------------------
    # 2. full downstream accumulation
    # ------------------------------------------------------------------
    accum = make_accumulation(be, grid, method="atomic", n_flat=n_flat)
    if "q_init" in accum:
        qi_bound = accum["q_init"].build()
        qi_bound.bind("SOURCE", source_p)
        qi_bound.bind("q", Q_in)
        q_init_kernel = _compile_bound(qi_bound, be)
    a_bound = accum["accum"].build()
    a_bound.bind("SOURCE", source_p)
    a_bound.bind("rec", rec_for_accum)
    a_bound.bind("q", Q_in)
    accum_kernel = _compile_bound(a_bound, be)

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
    core_kernel = _compile_bound(core_bound, be)

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
    """Immutable GraphFlood recipe, ready for live-state binding."""

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
    """Return an unbound GraphFlood recipe.

    Bind terrain, flow, and model parameters with :func:`bind_graphflood`.
    The resulting object updates its bound arrays in place on each call.

    Parameters
    ----------
    be : Backend
        Target backend. ``vanilla_mfd`` requires CuPy.
    grid : FrozenGroup
        Grid topology helpers.
    kind : {"vanilla_sfd", "unstable", "vanilla_mfd"}
        Flow-routing formulation.
    n_flat, nx, ny : int
        Raster size and dimensions.
    fill_method : {"jump", "reconstruct"}
        Depression treatment for ``vanilla_sfd``. ``jump`` reroutes receiver
        paths; ``reconstruct`` fills the water-surface field.
    accum_method : {"atomic"}
        Downstream accumulation method. This is currently the sole choice.
    depression_method : {"vanilla", "optimized"}
        Basin labelling method used by ``fill_method="jump"``.
    topology : {"D4", "D8"}
        Flow neighbourhood.
    diagonal_partition_correction : bool
        Apply diagonal-distance correction on D8 grids.
    friction_law : str
        Flow-resistance law.
    outlet_behavior : {"fixed_h", "free", "fixed_s"}
        Boundary treatment at outlets. ``fixed_h`` requires a boundary-depth
        parameter; ``fixed_s`` requires a boundary-slope parameter.
    block_size, max_passes : int, optional
        CuPy launch size and reconstruction iteration limit.

    Returns
    -------
    tuple
        A :class:`FrozenGraphflood` recipe and an empty parameter mapping.
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
    """Bind live GraphFlood state and compile its timestep.

    Parameters
    ----------
    frozen : FrozenGraphflood
        Recipe returned by :func:`make_graphflood`.
    grid_params : dict
        Parameters associated with the recipe's grid group.
    **bindings
        Live arrays and parameters. Every recipe requires ``z``, ``h``,
        ``Q_in``, ``Qo``, ``source_p``, ``manning_p``,
        ``friction_exponent_p``, ``dt_p``, and ``gf_min_increment_p``.
        ``fixed_h`` additionally needs ``boundary_h_p``; ``fixed_s`` needs
        ``boundary_slope_p``. ``jump`` recipes need receiver and basin
        scratch arrays; ``reconstruct`` recipes need surface and frontier
        scratch arrays. ``unstable`` needs ``Q_next``. CuPy MFD recipes need
        reconstruction, topology, frontier, and hop-distance scratch arrays.

    Returns
    -------
    GraphfloodVanillaSFD, GraphfloodUnstable, or GraphfloodVanillaMFD
        A callable timestep object. Call :meth:`close` when it is no longer used.
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
