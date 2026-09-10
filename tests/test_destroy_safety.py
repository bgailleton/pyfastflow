"""
Unit 6: destroy safety. A Parameter/handle is refcounted by every Bound and
compiled object that holds it (`_bound_by`); destroy()/release() refuse while
that count is non-zero, and close() (idempotent) is what decrements it. Covers
rebind/swap decrementing the old object, closing an uncompiled Bound, a
compiled routine retaining its bindings until close, and destruction/release
succeeding only after every owner has closed.

Author: B.G (09/2026)
"""

import importlib

import pytest

from pyfastflow.core.context.backends import Backend
from pyfastflow.core.context.bound import BindError
from pyfastflow.core.context.builder import KernelBuilder
from pyfastflow.core.context.errors import ParameterError
from pyfastflow.core.pool.base import PoolError


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

    return ti, Backend.from_name("taichi"), TaichiPool()


def _param(be, pool, name, mode="scalar", value=0):
    return be.ParameterCls(name, dtype="i32", mode=mode, value=value, pool=pool)


def test_rebind_decrements_old_param():
    ti, be, pool = _taichi()

    def _k(ctx, z: be.tensor):
        for i in z:
            z[i] = ctx.K.get(0)

    b = KernelBuilder(_k, domain="z").freeze().build()
    p1, p2 = _param(be, pool, "K"), _param(be, pool, "K2")
    b.bind("K", p1)
    assert p1._bound_by == 1
    b.bind("K", p2)  # rebind decrements the old
    assert p1._bound_by == 0 and p2._bound_by == 1
    b.close()
    p1.destroy()
    p2.destroy()
    pool.clear_all(force=True)


def test_close_uncompiled_bound_releases_and_is_idempotent():
    ti, be, pool = _taichi()

    def _k(ctx, z: be.tensor):
        for i in z:
            z[i] = ctx.K.get(0)

    b = KernelBuilder(_k, domain="z").freeze().build()
    p = _param(be, pool, "K")
    b.bind("K", p)
    with pytest.raises(ParameterError, match="still bound"):
        p.destroy()
    b.close()
    b.close()  # idempotent
    assert p._bound_by == 0
    p.destroy()  # now allowed
    pool.clear_all(force=True)


def test_bound_rejects_bind_after_close():
    ti, be, pool = _taichi()

    def _k(ctx, z: be.tensor):
        for i in z:
            z[i] = ctx.K.get(0)

    b = KernelBuilder(_k, domain="z").freeze().build()
    b.close()
    with pytest.raises(BindError, match="closed"):
        b.bind("K", _param(be, pool, "K"))
    pool.clear_all(force=True)


def test_handle_release_blocked_while_bound_to_data_slot():
    ti, be, pool = _taichi()

    def _k(ctx, z: be.tensor):
        for i in z:
            z[i] = 1.0

    b = KernelBuilder(_k, domain="z").freeze().build()
    h = pool.get_data(ti.i32, (8,))
    b.bind("z", h)  # a DataHandle bound directly to a DATA slot
    assert h._bound_by == 1
    with pytest.raises(PoolError, match="still bound"):
        pool.release_data(h)
    b.close()
    assert h._bound_by == 0
    pool.release_data(h)  # now allowed


def test_compiled_routine_retains_binding_until_close():
    ti, be, pool = _taichi()
    from pyfastflow.core.context.routine import RoutineBuilder

    T = be.tensor

    def _k(ctx, z: T):
        for i in z:
            z[i] = ctx.K.get(0)

    k = KernelBuilder(_k, domain="z").freeze()
    rb = RoutineBuilder().step("s1", k).step("s2", k)
    rb.share("s1.K", "s2.K", as_="K")
    rbound = rb.freeze().build()
    p = _param(be, pool, "K", value=5)
    rbound.bind("K", p)
    z1, z2 = pool.get_data(ti.i32, (8,)), pool.get_data(ti.i32, (8,))
    rbound.bind("s1.z", z1)
    rbound.bind("s2.z", z2)
    assert p._bound_by == 1  # routine bound holds it once so far

    run = rbound.compile()  # builds per-step bounds, each binds K via bind_into
    assert p._bound_by > 1  # the compiled routine's step bounds retain it too
    with pytest.raises(ParameterError):
        p.destroy()

    run.close()  # releases the step bounds
    rbound.close()  # releases the routine bound
    assert p._bound_by == 0
    p.destroy()
    z1.release()
    z2.release()
    pool.clear_all(force=True)


@pytest.mark.skipif(not _available("cupy"), reason="cupy/CUDA device not available")
def test_swap_decrements_old_handle():
    import cupy as cp  # noqa: F401

    be = Backend.from_name("cupy")
    pool = be.pool()
    tmpl = (
        'extern "C" __global__ void fill(int* out) {\n'
        "  int i = blockIdx.x*blockDim.x+threadIdx.x;\n"
        "  if (i < $ctx.N.get(0)$) out[i] = 1;\n"
        "}"
    )
    b = KernelBuilder(tmpl, domain="out").freeze().build()
    b.bind("N", be.ParameterCls("N", dtype="i32", mode="const", value=8, pool=None))
    h1 = pool.get_data(be.dtypes["i32"], (8,))
    b.bind("out", h1)
    run = b.compile()  # the compiled kernel now holds h1 (bound_by incremented)
    assert h1._bound_by >= 1
    h2 = pool.get_data(be.dtypes["i32"], (8,))
    run.swap("out", h2)  # decrements h1, increments h2
    assert h2._bound_by >= 1
    run.close()
    b.close()
    assert h1._bound_by == 0 and h2._bound_by == 0
