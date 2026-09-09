"""
Quadrants implementation of Parameter.

A Parameter's device view is a python def specialized by splicing bound
objects into its globals before handing it to qd.func - the mechanism in
_closure_backend.py, shared with Taichi.

The kernel/helper compile path (KernelBuilder/HelperBuilder -> FrozenKernel ->
BoundKernel.compile("quadrants")) is compile_closure.py, which reaches this
module only for `qd` itself, imported directly there. There, a kernel
template may type its data arguments qd.Tensor to accept either a field- or
ndarray-backed value at call time - Taichi has no equivalent. Field-mode
Parameters, on the other hand, must be field-backed, because they reach
device code as globals and Quadrants rejects an ndarray referenced as a
global inside a func.

Caching
-------
Leave Quadrants' src_ll fast cache off - do not mark templates compiled here
with @qd.pure or qd.kernel(fastcache=True), even to skip the python-side AST
trace. That cache keys a kernel on its source text, re-read from disk by file
path and line range (_fast_caching/function_hasher.py), together with argument
and config hashes. It never inspects __globals__, and globals are exactly what
distinguishes one specialization from another here. Two compiles of one
template with different bound consts, or a different helper under the same
name, hash identically; a hit then skips AST transformation, so nothing
downstream can catch the mismatch.

The IR-keyed caches are safe and stay on: Quadrants' own offline_cache, and
Taichi's, hash generated IR, which contains the baked literals and therefore
tells specializations apart.

Author: B.G (07/2026)
"""

import quadrants as qd

from ._closure_backend import ClosureBackendParameter


class QuadrantsParameter(ClosureBackendParameter):
    """
    Parameter backed by a Quadrants const value or a pooled QuadrantsDataHandle.

    Author: B.G (07/2026)
    """

    _BACKEND_NAME = "quadrants"

    _backend = qd
