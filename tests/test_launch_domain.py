"""
Unit 4: launch domain declared once on the kernel. A cupy kernel built with
`domain=<a DATA arg>` computes its grid from that buffer's length at every
launch, so `__call__()` takes no grid/block and a swap() to a shorter buffer
launches only the shorter extent.

cupy-only (the closure backends range over the template's own loop, so a
domain is inert there); skipped when no CUDA device is present. `block=1` is
used so the launched thread count equals the extent exactly, isolating the
grid-from-domain behaviour from within-block bounds handling.

Author: B.G (09/2026)
"""

import importlib

import pytest


def _cupy_available() -> bool:
    try:
        cp = importlib.import_module("cupy")
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cupy_available(), reason="cupy/CUDA device not available")

_TMPL = (
    'extern "C" __global__ void fill(int* out) {\n'
    "  int i = blockIdx.x*blockDim.x+threadIdx.x;\n"
    "  int guard = $ctx.N.get(0)$;\n"  # a PARAM so the kernel has a contract; value 0
    "  if (i < 1000000 + guard) out[i] = 7;\n"
    "}"
)


def _bound(domain):
    from pyfastflow.core.context.backends import Backend
    from pyfastflow.core.context.builder import KernelBuilder

    bk = Backend.from_name("cupy")
    b = KernelBuilder(_TMPL, domain=domain, block=1).freeze().build()
    b.bind("N", bk.ParameterCls("N", dtype="i32", mode="const", value=0, pool=None))
    return b


def test_domain_launch_no_grid_block_and_swap_shortens_extent():
    import cupy as cp
    from pyfastflow.core.context.backends import Backend

    b = _bound("out")
    pool = Backend.from_name("cupy").pool()
    big = pool.get_data("i32", (64,))
    big.from_numpy(cp.zeros(64, dtype=cp.int32))
    b.bind("out", big)
    run = b.compile()  # backend inferred from the bound parameter/handle

    run()
    cp.cuda.runtime.deviceSynchronize()
    assert int((big.array != 0).sum()) == 64  # grid sized from the 64-long buffer

    small = pool.get_data("i32", (16,))
    small.from_numpy(cp.zeros(16, dtype=cp.int32))
    run.swap("out", small)
    run()
    cp.cuda.runtime.deviceSynchronize()
    assert int((small.array != 0).sum()) == 16  # only the shorter extent launched
    run.close()
    b.close()
    pool.clear_all(force=True)


def test_int_domain_fixes_extent():
    import cupy as cp
    from pyfastflow.core.context.backends import Backend

    b = _bound(8)  # fixed extent 8
    pool = Backend.from_name("cupy").pool()
    buf = pool.get_data("i32", (64,))
    buf.from_numpy(cp.zeros(64, dtype=cp.int32))
    b.bind("out", buf)
    run = b.compile()
    run()
    cp.cuda.runtime.deviceSynchronize()
    assert int((buf.array != 0).sum()) == 8  # only the fixed extent launched
    run.close()
    b.close()
    pool.clear_all(force=True)
