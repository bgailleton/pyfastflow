"""CuPy-only shallow-water flood programs."""

from .fusw import FuswProgram, build_fusw_program
from .inertial import InertialFloodProgram, build_inertial_flood_program
from .quadtree import QuadtreeFusw, QuadtreeInertial

__all__ = [
    "FuswProgram", "InertialFloodProgram", "QuadtreeFusw", "QuadtreeInertial",
    "build_fusw_program", "build_inertial_flood_program",
]
