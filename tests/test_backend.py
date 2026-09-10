"""
Unit 5: the Backend object and backend consistency checked at bind/compile.

Covers a DATA-only kernel (no recorded backend -> compile needs an explicit
one), mixed-backend PARAM binding (rejected), and compile() backend inference
from a bound Parameter. Mixed-backend DATA binding is deferred to Unit 8 (DATA
bindings are still raw arrays, which carry no `.backend`).

Author: B.G (09/2026)
"""

import importlib

import pytest

from pyfastflow.core.context.backends import Backend
from pyfastflow.core.context.builder import KernelBuilder
from pyfastflow.core.context.compile_shared import CompileError


def _available(name: str) -> bool:
    try:
        mod = importlib.import_module(name)
    except Exception:
        return False
    if name == "cupy":
        try:
            return mod.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False
    return True


def _taichi():
    if not _available("taichi"):
        pytest.skip("taichi not available")
    import taichi as ti

    ti.init(arch=ti.gpu)
    from pyfastflow.core.pool.taichi_pool import TaichiPool

    return ti, TaichiPool()


def test_backend_object_identity_and_fields():
    be = Backend.from_name("cupy") if _available("cupy") else None
    if be is None:
        pytest.skip("cupy not available")
    assert be.name == "cupy" and be.family == "cupy"
    assert be.default_block == 256 and be.tensor is None
    assert Backend.from_name("cupy") is be  # cached
    assert Backend.from_name(be) is be  # idempotent
    assert be == Backend.from_name("cupy") and be != Backend.from_name("taichi") if _available("taichi") else True


def test_compile_infers_recorded_backend_and_rejects_mismatch():
    ti, pool = _taichi()
    be = Backend.from_name("taichi")
    T = be.tensor

    def _k_param(ctx, z: T):
        for i in z:
            z[i] = ctx.K.get(0)

    b = KernelBuilder(_k_param, domain="z").freeze().build()
    b.bind("K", be.ParameterCls("K", dtype="f32", mode="const", value=3.0, pool=pool))
    zh = pool.get_data(ti.f32, (8,))
    b.bind("z", zh)
    run = b.compile()  # no backend argument: inferred from the bound taichi Parameter
    run()
    ti.sync()
    assert float(zh.array.to_numpy()[0]) == 3.0

    if _available("cupy"):
        with pytest.raises(CompileError, match="Backend object"):
            b.compile("cupy")
        with pytest.raises(CompileError, match="cupy"):
            b.compile(Backend.from_name("cupy"))  # differs from the recorded taichi backend
    run.close()
    b.close()
    pool.clear_all(force=True)


def test_data_only_kernel_needs_explicit_backend():
    ti, pool = _taichi()
    T = Backend.from_name("taichi").tensor

    def _k_dataonly(ctx, z: T):
        for i in z:
            z[i] = 1.0

    b = KernelBuilder(_k_dataonly, domain="z").freeze().build()
    zh = pool.get_data(ti.f32, (8,))
    b.bind("z", zh)
    run = b.compile()  # inferred from the DataHandle
    run()
    ti.sync()
    assert float(zh.array.to_numpy()[0]) == 1.0
    run.close()
    b.close()
    pool.clear_all(force=True)


def test_mixed_backend_param_binding_rejected():
    if not (_available("taichi") and _available("cupy")):
        pytest.skip("needs both taichi and cupy")
    import taichi as ti

    ti.init(arch=ti.gpu)
    tbe = Backend.from_name("taichi")
    cbe = Backend.from_name("cupy")
    T = tbe.tensor

    def _two(ctx, z: T):
        for i in z:
            z[i] = ctx.A.get(0) + ctx.B.get(0)

    b = KernelBuilder(_two, domain="z").freeze().build()
    b.bind("A", tbe.ParameterCls("A", dtype="f32", mode="const", value=1.0, pool=None))
    from pyfastflow.core.context.bound import BindError

    with pytest.raises(BindError, match="single-backend"):
        b.bind("B", cbe.ParameterCls("B", dtype="f32", mode="const", value=2.0, pool=None))
