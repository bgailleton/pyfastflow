"""Recipe shared by every GraphFlood program (CuPy only).

``graphflood_builder(name)`` returns a ProgramBuilder holding what all
GraphFlood programs have in common:

- config: ``dx``, ``topology`` (D4|D8), ``boundary``, ``outlet``,
  ``nodata`` (the grid bundle), ``friction_law`` and ``mfd_local_minima``
  (how the MFD graph crosses local minima; default carve_cordonnier);
- params: ``precipitation`` (the only input: rain rate, m/s, const, scalar
  or field; point discharges enter as Q / dx^2 on their cell),
  ``friction_coefficient``, ``friction_exponent`` and ``carve_slope_min``;
- data: ``z`` (input), ``h`` (state), ``Qi``, ``Qo`` and ``rec`` (outputs) plus the
  routing buffers;
- steps: initial surfaces (``reset_h``, ``initialize_h_from_fill``,
  ``fill_hydraulic_surface``), the local-minima dispatches, the MFD
  topology, ``prepare_frontier`` and ``accumulate`` (rain accumulated into
  Qi). ``TOPOLOGY`` is that chain in pipeline order.

``graphflood_builder(name, supports_tree=True)`` adds the ``tree`` config: a D4
quadtree replaces the grid bundle, every array has ``tree.capacity`` entries on
one row, and ``dx``, ``boundary``, ``outlet`` and ``nodata`` belong to the tree.

``add_warmup(b)`` adds the analytical hillslope warm-up: ``warmup_relaxation``
and ``depth_cap`` params and the private pipeline ``_warmup_pass`` (TOPOLOGY
then a local analytical update with target and depth capped at depth_cap).

``finish_program(program)`` attaches the shared host methods
(``fill_topography``) and the ``outlet_mask``/``nodata_mask`` accessors.

Author: B.G (10/2026)
"""

from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.core.context.slot import ProgramError
from pyfastflow.flow._program_cupy import (
    _cupy_only, _grid_leaf_plan, _noop_factory, _reconstruct_epsilon_factory,
)
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.grid._mask import GridMaskAccessor as _GridMaskAccessor

from ._relax import _capped_local_analytical_update_factory, _warmup_update_plan
from ._routing import (
    _accumulation_factory, _carve_topology_factory, _carve_topology_plan,
    _filled_topology_factory, _filled_topology_plan, _frontier_factory,
    _reconstructed_topology_factory, _reconstructed_topology_plan,
    _topology_factory, _topology_plan,
)
from ._surface import (
    _carve_factory, _carve_plan, _copy_fill_depth_factory,
    _cordonnier_carve_factory, _cordonnier_carve_plan,
    _cordonnier_fill_factory, _cordonnier_fill_plan, _fill_surface_plan,
    _make_surface_factory, _merge_fill_depth_factory, _rank_factory,
    _rank_plan, _refresh_hydraulic_surface_factory, _reset_h_factory,
    _route_factory, _snapshot_factory,
)

FLAT = Dim("ny") * Dim("nx")
SHAPE = (Dim("ny"), Dim("nx"))

TOPOLOGY = (
    "make_surface", "route_local_minima", "snapshot_local_minima",
    "resolve_minima", "prepare_mfd_surface", "refresh_hydraulic_surface",
    "build_topology", "prepare_frontier", "accumulate",
)

_CORDONNIER = ("rank_cordonnier", "fill_cordonnier", "carve_cordonnier")


def _minima_cases(cordonnier, reconstruct):
    cases = dict.fromkeys(_CORDONNIER, cordonnier) if isinstance(
        cordonnier, str) else dict(zip(_CORDONNIER, cordonnier))
    cases["reconstruct_epsilon"] = reconstruct
    return cases


def graphflood_builder(name, *, supports_tree=False):
    """ProgramBuilder with the shared GraphFlood recipe (see module doc)."""
    b = ProgramBuilder(name)
    if supports_tree:
        b.config("tree", default=None)
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("dx", default=1.0)
    b.config("topology", choices=("D4", "D8"), default="D8")
    b.config("boundary", choices=("normal", "periodic_EW", "periodic_NS"),
             default="normal")
    b.config("outlet", choices=("edge", "mask"), default="edge")
    b.config("nodata", choices=(False, True), default=False)
    b.config("friction_law", choices=("manning",), default="manning")
    b.config("mfd_local_minima",
             choices=_CORDONNIER + ("reconstruct_epsilon",),
             default="carve_cordonnier")

    b.param("precipitation", "auto", "f32", value=0.0, shape=SHAPE)
    b.param("friction_coefficient", "auto", "f32", value=0.033, shape=SHAPE)
    b.param("friction_exponent", "auto", "f32", value=2.0 / 3.0, shape=SHAPE)
    b.param("carve_slope_min", "auto", "f32", value=1.0e-4, shape=SHAPE)
    b.param("ndep", "scalar", "i32", value=0)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)

    b.data("z", "f32", SHAPE, role="input", shape_source=True)
    b.data("h", "f32", SHAPE, role="state")
    b.data("Qi", "f32", SHAPE, role="output")
    b.data("Qo", "f32", SHAPE, role="output")
    b.data("surface", "f32", SHAPE, role="internal")
    b.data("hydraulic_surface", "f64", SHAPE, role="internal")
    b.data("rec", "i32", SHAPE, role="output")
    for data in ("rec_initial", "rank_ancestor", "rank_ancestor_alt",
                 "rank", "rank_alt", "bid", "basin_saddlenode",
                 "basin_route", "b_rcv", "mfd_frontier0", "mfd_frontier1",
                 "indegree"):
        b.data(data, "i32", (FLAT,), role="internal")
    for data in ("z_prime", "steepest_slope", "flow_width"):
        b.data(data, "f32", (FLAT,), role="internal")
    b.data("weights", "f32", (8 * FLAT,), role="internal")
    for data in ("is_border", "directions", "hydraulic_conditioned"):
        b.data(data, "u8", (FLAT,), role="internal")
    for data in ("basin_saddle", "basin_outlet"):
        b.data(data, "i64", (FLAT,), role="internal")
    b.data("mfd_count", "i32", (3,), role="internal")
    b.data("mfd_barrier", "u32", (1,), role="internal")
    # One-shot fill scratch, released to the pool after each call.
    for data in ("fill_parent", "fill_epsilon_ancestor",
                 "fill_epsilon_ancestor_work", "fill_counters",
                 "fill_queued_gen"):
        b.data(data, "i32", (FLAT,), lifetime="temp")
    for data in ("fill_epsilon_distance", "fill_epsilon_distance_work"):
        b.data(data, "f32", (FLAT,), lifetime="temp")
    b.data("fill_frontier", "i32", (2 * FLAT,), lifetime="temp")

    def grid_structure(be, *, nx, ny, dx, topology, boundary, outlet, nodata,
                       tree=None):
        _cupy_only(be)
        if tree is not None:
            if (nx, ny) != (tree.capacity, 1):
                raise ProgramError(f"tree: dims must be tree.dims {tree.dims}, got nx={nx}, ny={ny}")
            if tree.n_neighbours > 8:
                raise ProgramError("tree: GraphFlood stores 8 slots per cell, use a D4 tree")
            if topology != tree.topology:
                raise ProgramError(f"tree: topology {topology!r} differs from the tree's {tree.topology!r}")
            if (dx, boundary, outlet, nodata) != (1.0, "normal", "edge", False):
                raise ProgramError("tree: dx, boundary, outlet and nodata belong to the tree")
            return tree.group
        return make_grid_group(be, topology=topology, boundary=boundary,
                               outlet=outlet, nodata=nodata)

    def grid_params(be, pool, *, nx, ny, dx, topology, boundary, outlet,
                    nodata, tree=None):
        del boundary
        if tree is not None:
            return tree.parameters
        return make_grid_parameters(be, pool, nx, ny, dx, topology=topology,
                                    outlet=outlet, nodata=nodata)

    b.bundle("grid", grid_structure, grid_params, dims=("nx", "ny"),
             config=("dx", "topology", "boundary", "outlet", "nodata")
             + (("tree",) if supports_tree else ()))

    # Initial surfaces.
    b.add("reset_h", _reset_h_factory, bind={"h": "h"})
    b.add("reconstruct_fill_surface", _reconstruct_epsilon_factory,
          bind=_fill_surface_plan("z", "surface"))
    b.add("copy_fill_depth", _copy_fill_depth_factory,
          bind={"z": "z", "filled": "surface", "h": "h"})
    b.pipeline("initialize_h_from_fill",
               ("reconstruct_fill_surface", "copy_fill_depth"))
    b.add("make_surface", _make_surface_factory, bind={
        "z": "z", "h": "h", "surface": "surface",
        "hydraulic_surface": "hydraulic_surface",
    })
    b.add("refresh_hydraulic_surface", _refresh_hydraulic_surface_factory,
          bind={"z": "z", "h": "h", "hydraulic_surface": "hydraulic_surface"})
    b.add("reconstruct_hydraulic_surface", _reconstruct_epsilon_factory,
          bind=_fill_surface_plan("surface", "z_prime", distance="surface"))
    b.add("copy_hydraulic_fill_depth", _merge_fill_depth_factory, bind={
        "z": "z", "filled": "z_prime", "h": "h", "conditioned": "is_border",
    })
    b.pipeline("fill_hydraulic_surface", (
        "make_surface", "reconstruct_hydraulic_surface",
        "copy_hydraulic_fill_depth", "refresh_hydraulic_surface",
    ))

    # Local minima and MFD topology.
    b.add("route", _route_factory, bind={
        "grid": "grid", "surface": "hydraulic_surface", "rec": "rec",
    })
    b.add("snapshot_receivers", _snapshot_factory,
          bind={"rec": "rec", "rec_initial": "rec_initial"})
    b.add("skip_local_minima", _noop_factory, bind={})
    b.add("resolve_cordonnier", _carve_factory, bind=_carve_plan)
    b.add("compute_rank", _rank_factory, bind=_rank_plan)
    b.add("compute_cordonnier_fill", _cordonnier_fill_factory,
          bind=_cordonnier_fill_plan)
    b.add("compute_cordonnier_carve", _cordonnier_carve_factory,
          bind=_cordonnier_carve_plan)
    b.add("build_rank_topology", _topology_factory, bind=_topology_plan)
    b.add("build_fill_topology", _filled_topology_factory,
          bind=_filled_topology_plan)
    b.add("build_carve_topology", _carve_topology_factory,
          bind=_carve_topology_plan)
    b.add("build_reconstructed_topology", _reconstructed_topology_factory,
          bind=_reconstructed_topology_plan)
    on = "mfd_local_minima"
    b.dispatch("route_local_minima", on=on,
               cases=_minima_cases("route", "skip_local_minima"))
    b.dispatch("snapshot_local_minima", on=on,
               cases=_minima_cases("snapshot_receivers", "skip_local_minima"))
    b.dispatch("resolve_minima", on=on, cases=_minima_cases(
        "resolve_cordonnier", "reconstruct_hydraulic_surface"))
    b.dispatch("prepare_mfd_surface", on=on, cases=_minima_cases(
        ("compute_rank", "compute_cordonnier_fill",
         "compute_cordonnier_carve"), "copy_hydraulic_fill_depth"))
    b.dispatch("build_topology", on=on, cases=_minima_cases(
        ("build_rank_topology", "build_fill_topology",
         "build_carve_topology"), "build_reconstructed_topology"))

    # Rain accumulation over the MFD graph.
    b.add("prepare_frontier", _frontier_factory, bind={
        "clear.count": "mfd_count", "clear.barrier": "mfd_barrier",
        "compact.indegree": "indegree", "compact.frontier": "mfd_frontier0",
        "compact.count": "mfd_count",
    })
    b.add("accumulate", _accumulation_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "PRECIPITATION": "precipitation", "Qi": "Qi",
              "frontier0": "mfd_frontier0", "frontier1": "mfd_frontier1",
              "count": "mfd_count", "barrier": "mfd_barrier",
              "dirs": "directions", "mfd_w": "weights", "accum": "Qi",
              "indegree": "indegree",
          }))
    return b


def add_warmup(b):
    """Add the capped analytical hillslope warm-up (see module doc)."""
    b.param("warmup_relaxation", "auto", "f32", value=0.6)
    b.param("depth_cap", "auto", "f32", value=0.5)
    b.add("update_depth_capped", _capped_local_analytical_update_factory,
          bind=_warmup_update_plan)
    b.pipeline("_warmup_pass", TOPOLOGY + ("update_depth_capped",))
    return b


def fill_topography(self):
    """Replace z by its reconstructed epsilon fill and set h to 0.

    Depressions become topography (flat lakes with epsilon gradients).
    ``initialize_h_from_fill`` instead keeps z and fills them with water.
    """
    self.reconstruct_fill_surface()
    z = self._handle("z").array
    z[...] = self._handle("surface").array.reshape(z.shape)
    self.reset_h()


def finish_program(program):
    """Attach the shared host methods and mask accessors to ``program``."""
    program.fill_topography = fill_topography
    program.outlet_mask = property(
        lambda self: _GridMaskAccessor(self, "OUTLET_MASK", "outlet='mask'"))
    program.nodata_mask = property(
        lambda self: _GridMaskAccessor(self, "NODATA_MASK", "nodata=True"))
    return program
