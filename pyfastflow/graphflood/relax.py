"""GraphFloodRelax: GraphFlood by analytical depth relaxation (CuPy only).

Shared config, inputs and initial surfaces: see ``_base``.

``run(n)``: n passes of MFD topology on z+h (local minima handled by
``mfd_local_minima``), rain accumulation into Qi, then h relaxed by
``relaxation`` towards the depth that carries Qi:

- ``analytical_solver="local"``: the Manning depth of Qi at the frozen
  steepest slope;
- ``analytical_solver="bottom_up"``: receivers first along each cell's
  controlling (steepest) receiver, each depth solved against its receiver's
  solved head.

``warmup(n=5)``: n passes of the local solve relaxed by
``warmup_relaxation`` (0.6) with target and depth capped at ``depth_cap``
(0.5 m), to fill the hillslopes without walls of water in the rivers.

Convergence checks (``convergence()``, ``run_until_converged``): see
``_convergence``.

Typical use::

    gf = GraphFloodRelax(be, nx=nx, ny=ny, dx=dx, precipitation=rain)
    gf.z.from_numpy(z)
    gf.fill_topography()
    gf.warmup()
    gf.run(100)

On a D4 quadtree, ``GraphFloodRelax(be, tree=tree, topology="D4", **tree.dims)``
runs the same passes; ``reconstruct_epsilon`` and ``fill_topography`` need the
raster and are unavailable there.

Author: B.G (10/2026)
"""

from ._base import (
    FLAT, SHAPE, TOPOLOGY, add_warmup, finish_program, graphflood_builder,
)
from ._convergence import add_convergence, attach_convergence
from ._relax import (
    _bottom_up_analytical_update_factory, _bottom_up_analytical_update_plan,
    _local_analytical_update_factory, _local_analytical_update_plan,
)


def build_graphflood_relax() -> type:
    """Build the GraphFloodRelax class."""
    b = add_convergence(add_warmup(graphflood_builder("GraphFloodRelax", supports_tree=True)))
    b.config("analytical_solver", choices=("local", "bottom_up"),
             default="bottom_up")
    b.param("relaxation", "auto", "f32", value=0.1, shape=SHAPE)
    b.data("hydraulic_cut", "u8", (FLAT,), role="internal")
    b.add("update_depth_local", _local_analytical_update_factory,
          bind=_local_analytical_update_plan)
    b.add("update_depth_bottom_up", _bottom_up_analytical_update_factory,
          bind=_bottom_up_analytical_update_plan)
    b.dispatch("update_depth", on="analytical_solver", cases={
        "local": "update_depth_local", "bottom_up": "update_depth_bottom_up",
    })
    b.pipeline("run", TOPOLOGY + ("update_depth",))
    program = b.freeze()

    def warmup(self, n=5):
        """Capped analytical hillslope warm-up (see module doc)."""
        self._warmup_pass(n)

    program.warmup = warmup
    return attach_convergence(finish_program(program))


GraphFloodRelax = build_graphflood_relax()

__all__ = ["GraphFloodRelax", "build_graphflood_relax"]
