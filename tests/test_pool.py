"""
Tier 1: the device-buffer pool works on every backend present.

Two checks per backend: the acquire -> write/read -> release -> reuse ->
free happy path, and the lifecycle-misuse guards (double release, foreign
handle, clear_all over a live handle).

Author: B.G (08/2026)
"""

import importlib

import numpy as np
import pytest

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


def _pool_and_f32(name):
    from pyfastflow.core.context.backends import Backend

    if name == "taichi":
        from pyfastflow.core.pool.taichi_pool import TaichiPool as P
    elif name == "quadrants":
        from pyfastflow.core.pool.quadrants_pool import QuadrantsPool as P
    else:
        from pyfastflow.core.pool.cupy_pool import CupyPool as P
    return P(), Backend.from_name(name).dtypes["f32"]


def test_pool_lifecycle(backend):
    pool, f32 = _pool_and_f32(backend)
    n = 1024

    h = pool.get_data(f32, (n,))
    assert h.in_use
    assert pool.stats() == {"total": 1, "in_use": 1, "available": 0}

    h.from_numpy(np.arange(n, dtype=np.float32))
    assert np.array_equal(h.to_numpy(), np.arange(n, dtype=np.float32))

    pool.release_data(h)
    assert not h.in_use
    assert pool.stats() == {"total": 1, "in_use": 0, "available": 1}

    h2 = pool.get_data(f32, (n,))
    assert h2 is h
    assert pool.stats()["total"] == 1

    pool.release_data(h2)
    pool.clear_all()
    assert pool.stats() == {"total": 0, "in_use": 0, "available": 0}


def test_pool_misuse_guards(backend):
    from pyfastflow.core.pool.base import PoolError

    pool, f32 = _pool_and_f32(backend)
    other, _ = _pool_and_f32(backend)
    n = 256

    h = pool.get_data(f32, (n,))

    with pytest.raises(PoolError):
        other.release_data(h)

    with pytest.raises(PoolError):
        pool.clear_all()

    pool.release_data(h)
    with pytest.raises(PoolError):
        pool.release_data(h)

    pool.clear_all(force=True)
    other.clear_all(force=True)
