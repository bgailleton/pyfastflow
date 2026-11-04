"""GraphFlood programs: Vanilla (iterative), Relax (analytical relaxation)
and Particles (live particle processors), on one shared recipe (``_base``).
``quadtree`` runs Relax on a D4 quadtree refined from its own results.
"""

from ._convergence import ConvergenceRule
from .particles import GraphFloodParticles, build_graphflood_particles
from .quadtree import QuadtreeFlood, WaterLeft
from .relax import GraphFloodRelax, build_graphflood_relax
from .vanilla import GraphFloodVanilla, build_graphflood_vanilla

__all__ = [
    "ConvergenceRule", "QuadtreeFlood", "WaterLeft",
    "GraphFloodParticles", "GraphFloodRelax", "GraphFloodVanilla",
    "build_graphflood_particles", "build_graphflood_relax",
    "build_graphflood_vanilla",
]
