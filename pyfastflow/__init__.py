"""
PyFastFlow package root.

GPU geomorphology / hydraulics on a flat 1D-indexed grid, backend-agnostic
across Taichi, Quadrants and cupy.

Layout:
  core         the machinery only - `core.context` (the kernel build ->
               freeze -> bind -> compile metaprogramming layer) and
               `core.pool` (device-buffer pooling). Nothing domain-specific.
  grid, noise, flow, graphflood, ops, visu
               the feature layer built on `core`: each a set of
               `make_*_group` / `make_*_parameters` factories returning pure
               structure plus caller-owned Parameters, no stateful context
               classes.
  legacy       the pre-rewrite single-backend codebase, dead code kept for
               reference (see CONTEXT_legacy.md). Not imported here.
  experimental empty landing space for in-progress work.

Author: B.G (08/2026)
"""

__version__ = "0.2.0"
__author__ = "B.G."

# Lazy submodule loading to avoid heavy backend side effects at import time.
_LAZY_SUBMODULES = [
    "core",
    "experimental",
    "grid",
    "quadtree",
    "noise",
    "flow",
    "flood",
    "graphflood",
    "ops",
    "visu",
]

__all__ = list(_LAZY_SUBMODULES)


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
