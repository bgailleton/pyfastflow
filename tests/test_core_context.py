"""
Tier 2: the build -> freeze -> bind -> compile machinery, every layer on
every backend present.

Five checks, each parametrized over taichi/quadrants/cupy:

  test_kernel_param_modes  one KernelBuilder reading a const, a scalar and a
                           field PARAM plus two DATA slots; also that .build()
                           does not consume the builder and a re-bind with a
                           new scalar value gives a new result.
  test_helper_compose      a KernelBuilder that compose()s a HelperBuilder and
                           calls it as ctx.h(...).
  test_routine             a RoutineBuilder chaining two kernels over one
                           shared DATA address, compiled to one callable.
  test_sequence_host_loop  a SequenceBuilder whose loop predicate is a
                           HostBlockBuilder reading a scalar Parameter with
                           .read().
  test_build_contracts     the build-phase guards: missing slot, re-ingest of
                           a frozen builder, bind(None), compile with an
                           unbound slot, set() on a const Parameter.

Author: B.G (08/2026)
"""

import importlib

import numpy as np
import pytest

from pyfastflow.core.context.backends import Backend
from pyfastflow.core.context.bound import BindError
from pyfastflow.core.context.builder import HelperBuilder, KernelBuilder
from pyfastflow.core.context.contract import ContractError
from pyfastflow.core.context.frozen import FrozenError
from pyfastflow.core.context.host_block import HostBlockBuilder
from pyfastflow.core.context.compile_shared import CompileError
from pyfastflow.core.context.routine import RoutineBuilder
from pyfastflow.core.context.sequence import SequenceBuilder
from pyfastflow.core.pool.base import new_uid

_BACKENDS = ("taichi", "quadrants", "cupy")


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


@pytest.fixture(params=_BACKENDS)
def backend(request):
    name = request.param
    if not _available(name):
        pytest.skip(f"{name} not available")
    if name == "taichi":
        import taichi as ti

        ti.init(arch=ti.gpu)
    elif name == "quadrants":
        import quadrants as qd

        qd.init(arch=qd.gpu)
    return name


def _env(name):
    bk = Backend.from_name(name)
    if name == "taichi":
        from pyfastflow.core.pool.taichi_pool import TaichiPool as P

        T = bk.module.template()
    elif name == "quadrants":
        from pyfastflow.core.pool.quadrants_pool import QuadrantsPool as P

        T = bk.module.Tensor
    else:
        from pyfastflow.core.pool.cupy_pool import CupyPool as P

        T = None
    return P(), bk.ParameterCls, bk.dtypes, T


def _closure(name):
    return name in ("taichi", "quadrants")


# ---------------------------------------------------------------------------


def test_kernel_param_modes(backend):
    pool, Param, dt, T = _env(backend)
    f32 = dt["f32"]
    n = 1024

    if _closure(backend):
        def tmpl(ctx, arr: T, out: T):
            for i in arr:
                out[i] = arr[i] * ctx.K.get(0) + ctx.S.get(0) + ctx.F.get(i)
        template = tmpl
    else:
        t = f"pf{new_uid()}"
        template = f"""
extern "C" __global__ void {t}_k(const float* arr, float* out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    out[i] = arr[i] * $ctx.K.get(0)$ + $ctx.S.get(0)$ + $ctx.F.get(i)$;
}}
"""

    frozen = KernelBuilder(template, domain="arr").freeze()

    arr_np = np.arange(n, dtype=np.float32)
    fld_np = (np.arange(n, dtype=np.float32) * 0.5)
    arr = pool.get_data(f32, (n,))
    out = pool.get_data(f32, (n,))
    arr.from_numpy(arr_np)

    kp = Param("K", dtype="f32", mode="const", value=2.0, pool=pool)
    sp = Param("S", dtype="f32", mode="scalar", value=10.0, pool=pool)
    fp = Param("F", dtype="f32", mode="field", value=fld_np, pool=pool, shape=(n,))

    bound = frozen.build()
    bound.bind("K", kp)
    bound.bind("S", sp)
    bound.bind("F", fp)
    bound.bind("arr", arr)
    bound.bind("out", out)
    run = bound.compile()
    run()

    assert np.allclose(out.to_numpy(), arr_np * 2.0 + 10.0 + fld_np, rtol=1e-5)

    # builder not consumed; re-bind the same frozen with a new scalar value
    sp2 = Param("S", dtype="f32", mode="scalar", value=100.0, pool=pool)
    bound2 = frozen.build()
    bound2.bind("K", kp)
    bound2.bind("S", sp2)
    bound2.bind("F", fp)
    bound2.bind("arr", arr)
    bound2.bind("out", out)
    run2 = bound2.compile()
    run2()
    assert np.allclose(out.to_numpy(), arr_np * 2.0 + 100.0 + fld_np, rtol=1e-5)

    run.close()
    bound.close()
    run2.close()
    bound2.close()
    pool.clear_all(force=True)


def test_helper_compose(backend):
    pool, Param, dt, T = _env(backend)
    f32 = dt["f32"]
    n = 1024

    if _closure(backend):
        def bump_tmpl(ctx, x):
            return x + ctx.B.get(0)

        def k_tmpl(ctx, arr: T, out: T):
            for i in arr:
                out[i] = ctx.h(arr[i])
        helper = HelperBuilder(bump_tmpl).freeze()
        ktemplate = k_tmpl
    else:
        t = f"pf{new_uid()}"
        helper = HelperBuilder(
            f"__device__ float {t}_bump(float x) {{ return x + $ctx.B.get(0)$; }}"
        ).freeze()
        ktemplate = f"""
extern "C" __global__ void {t}_k(const float* arr, float* out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    out[i] = $ctx.h(arr[i])$;
}}
"""

    frozen = KernelBuilder(ktemplate, domain="arr").compose("h", helper).freeze()

    arr_np = np.arange(n, dtype=np.float32)
    arr = pool.get_data(f32, (n,))
    out = pool.get_data(f32, (n,))
    arr.from_numpy(arr_np)

    bp = Param("B", dtype="f32", mode="scalar", value=7.0, pool=pool)
    bound = frozen.build()
    bound.bind(("h", "B"), bp)
    bound.bind("arr", arr)
    bound.bind("out", out)
    run = bound.compile()
    run()

    assert np.allclose(out.to_numpy(), arr_np + 7.0, rtol=1e-5)
    run.close()
    bound.close()
    pool.clear_all(force=True)


def test_routine(backend):
    pool, Param, dt, T = _env(backend)
    i32 = dt["i32"]
    n = 1024

    if _closure(backend):
        def add1_tmpl(ctx, buf: T):
            for i in buf:
                buf[i] += 1

        def mul3_tmpl(ctx, buf: T):
            for i in buf:
                buf[i] *= 3
        add1, mul3 = add1_tmpl, mul3_tmpl
    else:
        t = f"pf{new_uid()}"
        add1 = f"""
extern "C" __global__ void {t}_add1(int* buf) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    buf[i] += 1;
}}
"""
        mul3 = f"""
extern "C" __global__ void {t}_mul3(int* buf) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    buf[i] *= 3;
}}
"""

    k_add = KernelBuilder(add1, domain="buf").freeze()
    k_mul = KernelBuilder(mul3, domain="buf").freeze()

    rb = RoutineBuilder()
    rb.step("add1", k_add)
    rb.step("mul3", k_mul)
    frozen = rb.freeze()

    buf = pool.get_data(i32, (n,))
    buf.from_numpy(np.full(n, 5, dtype=np.int32))

    bound = frozen.build()
    bound.bind(("add1", "buf"), buf)
    bound.bind(("mul3", "buf"), buf)
    run = bound.compile()
    run()

    assert np.array_equal(buf.to_numpy(), np.full(n, (5 + 1) * 3, dtype=np.int32))
    run.close()
    bound.close()
    pool.clear_all(force=True)


def test_sequence_host_loop(backend):
    pool, Param, dt, T = _env(backend)
    i32 = dt["i32"]

    if _closure(backend):
        def bump_tmpl(ctx):
            ctx.CNT.set_node(0, ctx.CNT.get(0) + 1)
        bump = bump_tmpl
    else:
        t = f"pf{new_uid()}"
        bump = f"""
extern "C" __global__ void {t}_bump() {{
    int cur = $ctx.CNT.get(0)$;
    $ctx.CNT.set_node(0, cur + 1)$;
}}
"""

    def stop_tmpl(ctx):
        return int(ctx.CNT.read()) >= 3

    k_bump = KernelBuilder(bump, domain=1, block=1).freeze()
    stop_hb = HostBlockBuilder(stop_tmpl).freeze()

    sb = SequenceBuilder()
    sb.add("bump", k_bump)
    sb.add("stop", stop_hb)
    sb.loop(body=["bump"], max_times=10, until="stop")
    frozen = sb.freeze()

    cnt = Param("CNT", dtype="i32", mode="scalar", value=0, pool=pool)
    bound = frozen.build()
    bound.bind(("bump", "CNT"), cnt)
    bound.bind(("stop", "CNT"), cnt)
    compiled = bound.compile()
    compiled()

    assert int(cnt.read()) == 3
    assert compiled.last_trip_counts == (3,)
    pool.clear_all(force=True)


def test_build_contracts(backend):
    pool, Param, dt, T = _env(backend)
    f32 = dt["f32"]
    n = 256

    # missing composition: a root used as a bare call (not a `.get`/`.set_node`
    # PARAM access) and never composed - PARAM is derived, but this is not a
    # PARAM shape, so freeze() reports it missing. (A bare `ctx.Z.get(0)` now
    # DERIVES a PARAM slot rather than raising - the Unit 3 semantics change.)
    if _closure(backend):
        def bad_tmpl(ctx, arr: T):
            for i in arr:
                arr[i] = ctx.foo(i)
        bad = bad_tmpl

        def ok_tmpl(ctx, arr: T):
            for i in arr:
                arr[i] = ctx.Z.get(0)
        ok = ok_tmpl
    else:
        t = f"pf{new_uid()}"
        bad = f'extern "C" __global__ void {t}_bad(float* arr) {{ arr[0] = $ctx.foo(0)$; }}'
        ok = f'extern "C" __global__ void {t}_ok(float* arr) {{ arr[0] = $ctx.Z.get(0)$; }}'

    with pytest.raises(ContractError):
        KernelBuilder(bad, domain="arr").freeze()

    # a frozen builder cannot be reused
    kb = KernelBuilder(ok, domain="arr")
    kb.freeze()
    with pytest.raises(FrozenError):
        kb.freeze()

    # bind(None)
    bound = KernelBuilder(ok, domain="arr").freeze().build()
    with pytest.raises(BindError):
        bound.bind("Z", None)

    # compile with an unbound slot
    with pytest.raises(CompileError):
        bound.compile(Backend.from_name(backend))

    # set() on a const Parameter
    cp = Param("Z", dtype="f32", mode="const", value=1.0, pool=pool)
    with pytest.raises(Exception):
        cp.set(2.0)

    pool.clear_all(force=True)
