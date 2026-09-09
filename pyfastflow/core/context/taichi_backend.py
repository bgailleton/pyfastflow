"""
Taichi implementation of Parameter.

A Parameter's device view is a python def specialized by rebuilding it with
the bound objects spliced into its globals, then handed to ti.func;
_closure_backend.py holds that machinery, shared with Quadrants. Everything
below just names Taichi as the backend to use.

The kernel/helper compile path (KernelBuilder/HelperBuilder -> FrozenKernel ->
BoundKernel.compile("taichi")) is compile_closure.py, which reaches this
module only for `ti` itself, imported directly there.

Author: B.G (07/2026)
"""

import taichi as ti

from ._closure_backend import ClosureBackendParameter


class TaichiParameter(ClosureBackendParameter):
    """
    Parameter backed by a Taichi const value or a pooled TaichiDataHandle.

    Author: B.G (07/2026)
    """

    _BACKEND_NAME = "taichi"

    _backend = ti
