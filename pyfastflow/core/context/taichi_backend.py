"""Taichi Parameter implementation."""

import taichi as ti

from ._closure_backend import ClosureBackendParameter


class TaichiParameter(ClosureBackendParameter):
    """
    Parameter backed by a Taichi const value or a pooled TaichiDataHandle.

    """

    _BACKEND_NAME = "taichi"

    _backend = ti
