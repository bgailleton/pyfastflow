"""Components: switchable pieces of a Program step, selected by a config
tuple of names.

A :class:`Component` is one named unit of a time step: a factory
``build(bundles, config) -> Node`` (a frozen kernel or routine), the program
parameters and data it needs, and a free ``payload`` (for example a kernel
body for :class:`Registry` fusion). A :class:`Registry` holds the components
one config tuple selects from:

  - ``registry.declare(builder)`` declares the config (default ``()``) and
    every parameter / data any component needs, each gated with ``when`` on
    the components that are active (a name shared by several components,
    or registries, is active when any of them is; a name the builder
    already declares unconditionally stays unconditional);
  - ``registry.steps(rb, bundles, config)`` appends the active components,
    in tuple order (a name may repeat), to a ``RoutineBuilder``: each as its
    own step followed by the registry's ``after`` step if any, or, with
    ``fuse``, all of them as the single step ``fuse(components, bundles,
    config)``.

Parameters are given as ``name: default`` (an ``auto`` f32 scalar) or
``name: dict(**ProgramBuilder.param kwargs)`` (``mode``, ``dtype``, ``value``,
``shape``); data as ``name: dict(**ProgramBuilder.data kwargs)`` (``dtype``,
``shape``, ...). Kernels refer to them by their upper-case names, which the
Program's ``bind="auto"`` plan resolves.

Author: B.G (10/2026)
"""

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .slot import ProgramBuilderError


@dataclass(frozen=True)
class Component:
    """One switchable unit of a step (see the module docstring)."""
    name: str
    build: Callable = None
    params: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    payload: Any = None


def _uses(key, names):
    names = frozenset(names)
    return lambda config: any(n in names for n in config[key])


def _either(a, b):
    return lambda config: a(config) or b(config)


class Registry:
    """The components selectable by the config tuple ``key``."""

    def __init__(self, key, components, *, after=None, fuse=None):
        self.key = key
        self.components = {c.name: c for c in components}
        self.after = after
        self.fuse = fuse

    def active(self, config):
        """The selected components, in tuple order."""
        out = []
        for name in config[self.key]:
            if name not in self.components:
                raise ValueError(f"unknown {self.key} entry {name!r}, "
                                 f"expected one of {tuple(self.components)}")
            out.append(self.components[name])
        return out

    def declare(self, b):
        """Declare the config and the gated parameters / data on ``b``."""
        b.config(self.key, default=())
        users = {}
        for comp in self.components.values():
            for kind, specs in (("param", comp.params), ("data", comp.data)):
                for name, spec in specs.items():
                    users.setdefault((kind, name), (spec, []))[1].append(
                        comp.name)
        for (kind, name), (spec, comps) in users.items():
            when = _uses(self.key, comps)
            store = b._params if kind == "param" else b._data
            old = store.get(name)
            if old is not None:
                if old.when is not None:
                    store[name] = replace(old, when=_either(old.when, when))
                continue
            if kind == "param":
                kw = dict(spec) if isinstance(spec, dict) else {"value": spec}
                b.param(name, kw.pop("mode", "auto"), kw.pop("dtype", "f32"),
                        when=when, **kw)
            else:
                if not isinstance(spec, dict):
                    raise ProgramBuilderError(
                        f"component data {name!r}: expected a dict")
                kw = dict(spec)
                b.data(name, kw.pop("dtype"), kw.pop("shape"), when=when,
                       **kw)
        return b

    def steps(self, rb, bundles, config):
        """Append the active components to the RoutineBuilder ``rb``."""
        comps = self.active(config)
        if not comps:
            return rb
        if self.fuse is not None:
            return rb.step(self.key, self.fuse(comps, bundles, config))
        for k, comp in enumerate(comps):
            rb = rb.step(f"{self.key}_{k}_{comp.name}",
                         comp.build(bundles, config))
            if self.after is not None:
                rb = rb.step(f"{self.key}_{k}_after",
                             self.after(bundles, config))
        return rb


__all__ = ["Component", "Registry"]
