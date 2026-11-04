"""Flood solvers on a D4 quadtree refined from their own solution (CuPy only).

Modules: ``fusw`` (:class:`QuadtreeFusw`), ``inertial`` (:class:`QuadtreeInertial`).

Author: B.G (11/2026)
"""

from .fusw import QuadtreeFusw, TreeFuswProgram
from .inertial import QuadtreeInertial, TreeInertialProgram

__all__ = ["QuadtreeFusw", "QuadtreeInertial", "TreeFuswProgram", "TreeInertialProgram"]
