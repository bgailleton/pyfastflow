"""The stateful, embeddable orchestration layer for frozen PyFastFlow nodes.

``ProgramBuilder`` only records schema and factories. A built Program owns
parameters and data, builds each frozen node once dimensions are known, binds
it through one declarative map, and compiles it lazily or eagerly.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .backends import require_backend
from .frozen import Node
from .host_block import FrozenHostBlock
from .slot import ProgramBuilderError, ProgramError

_NP_DTYPES = {"i32": np.int32, "i64": np.int64, "f32": np.float32, "u8": np.uint8, "u32": np.uint32}
_REQUIRED = object()


class Dim:
    """A named symbolic shape dimension."""
    __slots__ = ("name",)
    def __init__(self, name): self.name = name
    def __mul__(self, other): return _Prod((self, other))
    def __rmul__(self, other): return _Prod((other, self))
    def __repr__(self): return f"Dim({self.name!r})"


class _Prod:
    __slots__ = ("factors",)
    def __init__(self, factors):
        self.factors = tuple(x for f in factors for x in (f.factors if isinstance(f, _Prod) else (f,)))
    def __mul__(self, other): return _Prod(self.factors + (other,))
    def __rmul__(self, other): return _Prod((other,) + self.factors)


def _axis_names(axis):
    if isinstance(axis, Dim): return {axis.name}
    if isinstance(axis, _Prod): return set().union(*(_axis_names(x) for x in axis.factors))
    if isinstance(axis, int): return set()
    raise ProgramBuilderError(f"shape axis must be a Dim, product, or int; got {type(axis).__name__}")


def _resolve_axis(axis, dims):
    if isinstance(axis, Dim): return int(dims[axis.name])
    if isinstance(axis, _Prod):
        n = 1
        for x in axis.factors: n *= _resolve_axis(x, dims)
        return n
    if isinstance(axis, int): return axis
    raise ProgramError(f"invalid shape axis {axis!r}")


def _resolve_shape(shape, dims): return tuple(_resolve_axis(x, dims) for x in shape)


@dataclass(frozen=True)
class _ParamSpec:
    name: str; mode: str; dtype: str; value: Any; shape: tuple; shape_source: bool

@dataclass(frozen=True)
class _DataSpec:
    name: str; dtype: str; shape: tuple; lifetime: str; role: Any; flat: bool; shape_source: bool

@dataclass(frozen=True)
class _BundleSpec:
    name: str; structure: Callable; params: Callable; dims: tuple; config: tuple

@dataclass(frozen=True)
class _SeqSpec:
    name: str; factory: Callable; bind: Any

@dataclass(frozen=True)
class _ConfigSpec:
    name: str; choices: Any; default: Any

@dataclass(frozen=True)
class _DispatchSpec:
    name: str; on: str; cases: dict

@dataclass(frozen=True)
class _Recipe:
    name: str; dims: tuple; params: dict; data: dict; bundles: dict; sequences: dict; config: dict; dispatch: dict


class ProgramBuilder:
    """Author a Program recipe with declarative bundles and bind maps."""
    def __init__(self, name):
        self._name = name; self._dims = []; self._params = {}; self._data = {}
        self._bundles = {}; self._sequences = {}; self._config = {}; self._dispatch = {}

    def dim(self, name):
        if name in self._dims: raise ProgramBuilderError(f"dim {name!r} already declared")
        self._dims.append(name); return self

    def config(self, name, *, choices=None, default=_REQUIRED):
        choices = tuple(choices) if choices is not None else None
        if choices is not None and default is not _REQUIRED and default not in choices:
            raise ProgramBuilderError(f"config {name!r}: default {default!r} is not among {list(choices)}")
        self._config[name] = _ConfigSpec(name, choices, default); return self

    def param(self, name, mode, dtype, *, value=0, shape=(), shape_source=False):
        if mode not in ("const", "scalar", "field"): raise ProgramBuilderError(f"{name!r}: invalid parameter mode {mode!r}")
        if dtype not in _NP_DTYPES: raise ProgramBuilderError(f"{name!r}: unsupported dtype {dtype!r}")
        if mode != "field" and shape: raise ProgramBuilderError(f"{name!r}: only field parameters have a shape")
        if shape_source and mode != "field": raise ProgramBuilderError(f"{name!r}: shape_source requires a field parameter")
        self._params[name] = _ParamSpec(name, mode, dtype, value, tuple(shape), shape_source); return self

    def data(self, name, dtype, shape, *, lifetime="persistent", role=None, flat=True, shape_source=False):
        if dtype not in _NP_DTYPES: raise ProgramBuilderError(f"{name!r}: unsupported dtype {dtype!r}")
        if lifetime not in ("persistent", "temp"): raise ProgramBuilderError(f"{name!r}: lifetime must be persistent or temp")
        if shape_source and lifetime == "temp": raise ProgramBuilderError(f"{name!r}: temp data cannot be a shape source")
        self._data[name] = _DataSpec(name, dtype, tuple(shape), lifetime, role, flat, shape_source); return self

    def bundle(self, name, structure, params, *, dims=(), config=()):
        self._bundles[name] = _BundleSpec(name, structure, params, tuple(dims), tuple(config)); return self

    def add(self, name, factory, *, bind):
        if not isinstance(bind, dict) and not callable(bind):
            raise ProgramBuilderError(f"sequence {name!r}: bind must be a mapping or a declarative binding-plan callable")
        self._sequences[name] = _SeqSpec(name, factory, dict(bind) if isinstance(bind, dict) else bind); return self

    def dispatch(self, name, *, on, cases):
        self._dispatch[name] = _DispatchSpec(name, on, dict(cases)); return self

    def freeze(self):
        dims = tuple(self._dims); dimset = set(dims); seen = {}
        for kind, names in (("dim", dims), ("param", self._params), ("data", self._data), ("bundle", self._bundles), ("sequence", self._sequences), ("dispatch", self._dispatch)):
            for name in names:
                if not name.isidentifier(): raise ProgramBuilderError(f"{kind} name {name!r} is not an identifier")
                if name in seen: raise ProgramBuilderError(f"name {name!r} declared twice ({seen[name]}, {kind})")
                seen[name] = kind
        for name in self._config:
            if not name.isidentifier(): raise ProgramBuilderError(f"config name {name!r} is not an identifier")
            if name in seen and seen[name] != "dim": raise ProgramBuilderError(f"name {name!r} declared twice ({seen[name]}, config)")
        for spec in list(self._params.values()) + list(self._data.values()):
            unknown = set().union(*(_axis_names(a) for a in spec.shape)) - dimset
            if unknown: raise ProgramBuilderError(f"{spec.name!r}: undeclared dims {sorted(unknown)}")
            if spec.shape_source and any(not isinstance(a, (Dim, int)) for a in spec.shape): raise ProgramBuilderError(f"{spec.name!r}: shape-source axes must be bare Dim or int")
        for spec in self._bundles.values():
            selected = set(spec.dims) | set(spec.config)
            if len(selected) != len(spec.dims) + len(spec.config): raise ProgramBuilderError(f"bundle {spec.name!r}: duplicate selected name")
            if set(spec.dims) - dimset or set(spec.config) - set(self._config): raise ProgramBuilderError(f"bundle {spec.name!r}: unknown dim/config selection")
        for spec in self._dispatch.values():
            if spec.on not in self._config or set(spec.cases.values()) - set(self._sequences): raise ProgramBuilderError(f"dispatch {spec.name!r}: invalid selector or case")
        recipe = _Recipe(self._name, dims, dict(self._params), dict(self._data), dict(self._bundles), dict(self._sequences), dict(self._config), dict(self._dispatch))
        return type(self._name, (_Program,), {"_recipe": recipe})


class _Accessor:
    __slots__ = ("_prog", "_name", "_kind")
    def __init__(self, prog, name, kind): self._prog, self._name, self._kind = prog, name, kind
    def set(self, value):
        if self._kind != "scalar": raise ProgramError(f"{self._name!r} is not a scalar parameter")
        self._prog._set_scalar(self._name, value)
    def read(self):
        self._prog._check_open()
        if self._kind == "const": return self._prog._recipe.params[self._name].value
        if self._kind != "scalar": raise ProgramError(f"{self._name!r} is not a scalar parameter")
        return self._prog._read_scalar(self._name)
    def from_numpy(self, array):
        if self._kind not in ("field", "data"): raise ProgramError(f"{self._name!r} is not array storage")
        self._prog._set_array(self._name, array)
    def to_numpy(self):
        if self._kind not in ("field", "data"): raise ProgramError(f"{self._name!r} is not array storage")
        return self._prog._get_array(self._name)
    def adopt(self, array):
        if self._kind != "data": raise ProgramError(f"{self._name!r} is not persistent data")
        self._prog._adopt(self._name, array)
    @property
    def array(self): return self._prog._handle(self._name).array
    @property
    def shape(self): return self._prog._host_shape(self._name)
    @property
    def dtype(self): return np.dtype(_NP_DTYPES[self._prog._dtype(self._name)])


class _SeqState:
    __slots__ = ("compiled", "bound", "temp_plan", "placeholders", "before", "after")
    def __init__(self): self.compiled = self.bound = None; self.temp_plan = []; self.placeholders = []; self.before = self.after = ""


class _BundleView(dict):
    """The mapping passed to an ``add`` factory.

    Normal mapping operations deliberately expose only frozen bundle Nodes, as
    promised by the public factory contract.  The two read-only helpers are
    for feature factories whose *structure* needs an already-owned Parameter
    (for example a host-loop counter); they do not bind anything and cannot
    expose raw backend arrays.
    """
    __slots__ = ("_program",)

    def __init__(self, program):
        super().__init__(program._bundles)
        self._program = program

    def param(self, name):
        try:
            return self._program._params[name]
        except KeyError as exc:
            raise ProgramError(f"no program parameter {name!r}") from exc

    def data(self, name):
        try:
            return self._program._data[name]
        except KeyError as exc:
            raise ProgramError(f"no persistent program data {name!r}") from exc

    def bundle_params(self, name):
        try:
            return dict(self._program._bundle_params[name])
        except KeyError as exc:
            raise ProgramError(f"no program bundle {name!r}") from exc


class _Program:
    _recipe: _Recipe
    def __init__(self, be, pool=None, **config):
        self._be = require_backend(be); self._closed = False; self._owns_pool = pool is None
        if pool is not None and not isinstance(pool, self._be.PoolCls): raise ProgramError(f"pool does not match {self._be.name!r}")
        self._pool = self._be.pool() if pool is None else pool
        self._config = self._resolve_config(config); self._dim_vals = {n: int(v) for n, v in self._config.items() if n in self._recipe.dims}
        self._params = {}; self._data = {}; self._bundles = {}; self._bundle_params = {}; self._owned_params = []; self._allocated = False
        self._pending_arrays = {}; self._pending_scalars = {}; self._states = {n: _SeqState() for n in self._recipe.sequences}
        self._install(); self._maybe_allocate()

    def _check_open(self):
        if self._closed: raise ProgramError("program is closed")
    def _resolve_config(self, supplied):
        extra = set(supplied) - set(self._recipe.config)
        if extra: raise ProgramError(f"unknown config {sorted(extra)}")
        out = {}
        for name, spec in self._recipe.config.items():
            value = supplied.get(name, spec.default)
            if value is _REQUIRED: raise ProgramError(f"config {name!r} is required")
            if spec.choices is not None and value not in spec.choices: raise ProgramError(f"config {name!r}={value!r} is invalid")
            out[name] = value
        return out
    def _install(self):
        for n, s in self._recipe.params.items(): object.__setattr__(self, n, _Accessor(self, n, {"const":"const", "scalar":"scalar", "field":"field"}[s.mode]))
        for n, s in self._recipe.data.items():
            if s.lifetime == "persistent" and s.role != "internal": object.__setattr__(self, n, _Accessor(self, n, "data"))
        for n in self._recipe.sequences:
            def _runner(n=1, _name=n):
                return self._run_sequence(_name, n)
            object.__setattr__(self, n, _runner)
        for n in self._recipe.dispatch:
            def _dispatcher(n=1, _name=n):
                return self.run(_name, n)
            object.__setattr__(self, n, _dispatcher)
    def __getattr__(self, name):
        if name in self._dim_vals: return self._dim_vals[name]
        if name in self._config: return self._config[name]
        raise AttributeError(name)
    def get(self, name):
        if hasattr(self, name): return getattr(self, name)
        raise ProgramError(f"no public program member {name!r}")
    def _dtype(self, name): return (self._recipe.params.get(name) or self._recipe.data[name]).dtype
    def _spec(self, name): return self._recipe.params.get(name) or self._recipe.data.get(name)
    def _host_shape(self, name):
        self._check_open()
        if not self._allocated: raise ProgramError("shapes are unresolved")
        return _resolve_shape(self._spec(name).shape, self._dim_vals)
    def _device_shape(self, spec):
        shape = _resolve_shape(spec.shape, self._dim_vals)
        return (int(np.prod(shape)),) if spec.flat else shape
    def _bind_source_dims(self, name, arr):
        spec = self._spec(name)
        if len(arr.shape) != len(spec.shape): raise ProgramError(f"{name!r}: source rank mismatch")
        for axis, size in zip(spec.shape, arr.shape):
            if isinstance(axis, int):
                if size != axis: raise ProgramError(f"{name!r}: expected axis {axis}, got {size}")
            else:
                old = self._dim_vals.get(axis.name)
                if old is not None and old != size: raise ProgramError(f"dim {axis.name!r} conflicts ({old}, {size})")
                self._dim_vals[axis.name] = int(size)
    def _maybe_allocate(self):
        if not self._allocated and all(x in self._dim_vals for x in self._recipe.dims): self._allocate()
    def _allocate(self):
        for name, spec in self._recipe.bundles.items():
            selected = {x:self._dim_vals[x] for x in spec.dims}; selected.update({x:self._config[x] for x in spec.config})
            node = spec.structure(self._be, **selected)
            if not isinstance(node, Node): raise ProgramError(f"bundle {name!r} structure returned {type(node).__name__}, not Node")
            params = dict(spec.params(self._be, self._pool, **selected))
            self._bundles[name], self._bundle_params[name] = node, params; self._owned_params.extend(params.values())
        for name, spec in self._recipe.params.items():
            shape = _resolve_shape(spec.shape, self._dim_vals) if spec.mode == "field" else ()
            value = spec.value(self._dim_vals) if callable(spec.value) else spec.value
            p = self._be.ParameterCls(name, dtype=spec.dtype, mode=spec.mode, value=value, pool=self._pool, shape=shape)
            self._params[name] = p; self._owned_params.append(p)
        for name, spec in self._recipe.data.items():
            if spec.lifetime == "persistent": self._data[name] = self._pool.get_data(self._be.dtypes[spec.dtype], self._device_shape(spec))
        self._allocated = True
        for name, arr in tuple(self._pending_arrays.items()): self._write_array(name, arr)
        self._pending_arrays.clear()
        for name, value in tuple(self._pending_scalars.items()): self._params[name].set(value)
        self._pending_scalars.clear()
    def _set_array(self, name, arr):
        self._check_open(); arr = np.asarray(arr)
        if arr.dtype != np.dtype(_NP_DTYPES[self._dtype(name)]): raise ProgramError(f"{name!r}: dtype mismatch")
        if self._spec(name).shape_source: self._bind_source_dims(name, arr); self._maybe_allocate()
        if not self._allocated: self._pending_arrays[name] = arr; return
        self._write_array(name, arr)
    def _write_array(self, name, arr):
        shape = self._host_shape(name)
        if tuple(arr.shape) != shape: raise ProgramError(f"{name!r}: expected shape {shape}, got {tuple(arr.shape)}")
        raw = arr.reshape(-1) if getattr(self._spec(name), "flat", True) else arr
        if name in self._params: self._params[name].set(raw)
        else: self._data[name].from_numpy(raw)
    def _get_array(self, name):
        self._check_open(); out = self._data[name].to_numpy() if name in self._data else self._params[name].handle().to_numpy()
        return out.reshape(self._host_shape(name)) if getattr(self._spec(name), "flat", True) else out
    def _handle(self, name):
        self._check_open()
        if name not in self._data: raise ProgramError(f"{name!r} is not persistent data")
        return self._data[name]
    def _set_scalar(self, name, value):
        self._check_open()
        if not self._allocated: self._pending_scalars[name] = value
        else: self._params[name].set(value)
    def _read_scalar(self, name):
        self._check_open()
        if not self._allocated: raise ProgramError("shapes are unresolved")
        return self._params[name].read()
    def _adopt(self, name, array):
        self._check_open()
        if not self._allocated: raise ProgramError("adopt requires resolved shapes")
        if any(s.compiled is not None for s in self._states.values()): raise ProgramError("adopt before compile; compiled objects retain bindings")
        h = self._be.wrap(array, owned=False); spec = self._recipe.data[name]
        if h.dtype != spec.dtype or tuple(h.shape) != self._device_shape(spec): raise ProgramError(f"{name!r}: adopted buffer has incompatible dtype or shape")
        self._pool.release_data(self._data[name]); self._data[name] = h
    def _value(self, name):
        if isinstance(name, str) and name.endswith(".handle"):
            param_name = name[:-7]
            try:
                return self._params[param_name].handle()
            except KeyError as exc:
                raise ProgramError(f"bind map refers to unknown parameter {param_name!r}") from exc
        if name in self._params: return self._params[name]
        if name in self._data: return self._data[name]
        raise ProgramError(f"bind map refers to unknown program value {name!r}")
    def _bind_bundle(self, bound, root, name, used):
        try:
            actual = bound.root_at(root)
        except Exception:
            # A bundle's parameter may also be named directly when a feature
            # has deliberately collapsed its grid root with ``share(as_=...)``.
            # This is still an exact address, not a leaf-name heuristic.
            addr = tuple(root.split("."))
            params = self._bundle_params[name]
            if addr not in bound.addresses() or addr[-1] not in params:
                raise ProgramError(f"bundle bind {root!r}: not a root or a parameter leaf of bundle {name!r}")
            if addr in used: raise ProgramError(f"duplicate expanded target {root!r}")
            bound.bind(addr, params[addr[-1]])
            used.add(addr)
            return
        if actual is not self._bundles[name]: raise ProgramError(f"bundle bind {root!r} is not the program's {name!r} node")
        prefix = tuple(root.split(".")); params = self._bundle_params[name]
        # A bundle root may be re-rooted through share(as_=...) so its leaves
        # are not necessarily physically below ``root`` in addresses(): map
        # those collapsed paths back to their canonical bindable addresses.
        expanded = {addr for addr in bound.addresses() if addr[:len(prefix)] == prefix}
        expanded.update(canonical for source, canonical in bound._redirect.items() if source[:len(prefix)] == prefix)
        for addr in expanded:
            if addr in used: raise ProgramError(f"duplicate expanded target {'.'.join(addr)!r}")
            if addr[-1] not in params: raise ProgramError(f"bundle {name!r} lacks parameter {addr[-1]!r}")
            bound.bind(addr, params[addr[-1]]); used.add(addr)
    def _ensure_compiled(self, name):
        self._check_open(); state = self._states[name]
        if state.compiled is not None: return
        if not self._allocated: raise ProgramError("shapes are unresolved")
        spec = self._recipe.sequences[name]
        # Factories do not receive a separate legacy ``dims`` argument.  The
        # resolved dimension values are nevertheless useful for feature
        # structures whose immutable recipe needs an extent (notably CUDA), so
        # expose them alongside user configuration in this read-only snapshot.
        factory_config = {**self._config, **self._dim_vals}
        node = spec.factory(self._be, _BundleView(self), factory_config)
        if not isinstance(node, Node): raise ProgramError(f"sequence {name!r} factory must return a frozen Node")
        bindings = spec.bind(node, self._be) if callable(spec.bind) else spec.bind
        if not isinstance(bindings, dict):
            raise ProgramError(f"sequence {name!r}: binding plan returned {type(bindings).__name__}, expected dict")
        bound = node.build(); state.before = bound.inspect(); used = set(); temp_addrs = {}; temp_placeholders = {}
        try:
            for target, value_name in bindings.items():
                if value_name in self._bundles: self._bind_bundle(bound, target, value_name, used); continue
                addr = tuple(target.split("."))
                if addr in used: raise ProgramError(f"duplicate bind target {target!r}")
                data = self._recipe.data.get(value_name)
                if data is not None and data.lifetime == "temp":
                    placeholder = temp_placeholders.get(value_name)
                    if placeholder is None:
                        placeholder = self._pool.get_data(self._be.dtypes[data.dtype], self._device_shape(data))
                        temp_placeholders[value_name] = placeholder
                    bound.bind(addr, placeholder)
                    temp_addrs.setdefault(value_name, []).append(addr)
                else:
                    bound.bind(addr, self._value(value_name))
                used.add(addr)
            if bound.unmet(): raise ProgramError(f"sequence {name!r} has unmet bindings:\n{bound.inspect()}")
            for temp_name, addrs in temp_addrs.items():
                state.placeholders.append(temp_placeholders[temp_name]); state.temp_plan.append((temp_name, addrs))
            state.after = bound.inspect()
            state.compiled = bound.compile() if isinstance(node, FrozenHostBlock) else bound.compile(self._be)
            state.bound = bound
        except Exception:
            bound.close()
            for h in state.placeholders: self._pool.release_data(h)
            state.placeholders.clear(); state.temp_plan.clear(); raise
    def _run_sequence(self, name, n=1):
        self._ensure_compiled(name); state = self._states[name]; acquired = []
        try:
            for (temp, addrs), placeholder in zip(state.temp_plan, state.placeholders):
                data = self._recipe.data[temp]; h = self._pool.get_data(self._be.dtypes[data.dtype], self._device_shape(data)); acquired.append(h)
                for addr in addrs: state.compiled.swap(addr, h)
            for _ in range(max(0, int(n))): state.compiled()
        finally:
            for (_temp, addrs), placeholder in zip(state.temp_plan, state.placeholders):
                for addr in addrs: state.compiled.swap(addr, placeholder)
            for h in acquired: self._pool.release_data(h)
        return state.compiled
    def run(self, name, n=1):
        if name in self._recipe.sequences: return self._run_sequence(name, n)
        if name in self._recipe.dispatch:
            d = self._recipe.dispatch[name]; return self._run_sequence(d.cases[self._config[d.on]], n)
        raise ProgramError(f"unknown sequence {name!r}")
    def compile(self):
        self._check_open()
        if not self._allocated: raise ProgramError("shapes are unresolved")
        for name in self._states: self._ensure_compiled(name)
    def inspect(self): return "\n\n".join(f"[{n}]\n{s.after or s.before or '(not built)'}" for n, s in self._states.items())
    def describe(self): return f"{self._recipe.name}: config={list(self._recipe.config)}, params={list(self._recipe.params)}, data={list(self._recipe.data)}, bundles={list(self._recipe.bundles)}, sequences={list(self._recipe.sequences)}"
    def state(self):
        self._check_open()
        if not self._allocated: raise ProgramError("shapes are unresolved")
        return {"schema": self._recipe.name, "version": 1, "dims":dict(self._dim_vals), "data":{n:self._get_array(n) for n,s in self._recipe.data.items() if s.lifetime == "persistent"}, "params":{n:self._read_scalar(n) for n,s in self._recipe.params.items() if s.mode == "scalar"}}
    def load_state(self, payload):
        self._check_open()
        if payload.get("schema") != self._recipe.name or payload.get("version") != 1: raise ProgramError("state schema/version mismatch")
        if payload.get("dims") != self._dim_vals: raise ProgramError("state dimensions do not match this program")
        expected_data = {n for n, s in self._recipe.data.items() if s.lifetime == "persistent"}
        expected_params = {n for n, s in self._recipe.params.items() if s.mode == "scalar"}
        if set(payload.get("data", ())) != expected_data or set(payload.get("params", ())) != expected_params:
            raise ProgramError("state names do not match this program")
        checked = []
        for n, arr in payload.get("data", {}).items():
            arr = np.asarray(arr)
            if n not in self._recipe.data or arr.dtype != np.dtype(_NP_DTYPES[self._dtype(n)]) or tuple(arr.shape) != self._host_shape(n): raise ProgramError(f"invalid state data {n!r}")
            checked.append((n, arr))
        for n in payload.get("params", {}):
            if n not in self._recipe.params or self._recipe.params[n].mode != "scalar": raise ProgramError(f"invalid state parameter {n!r}")
        for n, arr in checked: self._write_array(n, arr)
        for n, value in payload.get("params", {}).items(): self._params[n].set(value)
    def close(self):
        if self._closed: return
        self._closed = True
        for state in self._states.values():
            if state.compiled is not None and hasattr(state.compiled, "close"): state.compiled.close()
            elif state.bound is not None: state.bound.close()
            for h in state.placeholders:
                try: self._pool.release_data(h)
                except Exception: pass
        for p in self._owned_params:
            try: p.destroy()
            except Exception: pass
        for h in self._data.values():
            try: self._pool.release_data(h)
            except Exception:
                try: h.destroy()
                except Exception: pass
        self._params.clear(); self._data.clear()
        if self._owns_pool:
            try: self._pool.clear_all(force=True)
            except Exception: pass
    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
    def __repr__(self): return f"{type(self).__name__}(backend={self._be.name!r}, {'closed' if self._closed else 'open'})"
