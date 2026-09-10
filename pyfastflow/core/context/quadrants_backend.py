"""Quadrants Parameter implementation."""

import quadrants as qd

from ._closure_backend import ClosureBackendParameter


class QuadrantsParameter(ClosureBackendParameter):
    """
    Parameter backed by a Quadrants const value or a pooled QuadrantsDataHandle.

    """

    _BACKEND_NAME = "quadrants"

    _backend = qd
