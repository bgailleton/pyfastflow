"""Shared closure-backend flow helpers."""


def _tensor_annotation(backend_mod, backend: str):
    """
    The data-argument annotation a kernel template needs on this closure
    backend: `ti.template()` for Taichi, `qd.Tensor` for Quadrants - mirrors
    ../ops/_closure_blocks.py's _tensor_annotation.

    """
    return backend_mod.template() if backend == "taichi" else backend_mod.Tensor
