"""
Unit 7: the public surface. `pyfastflow.core` and `pyfastflow.core.context`
export exactly the agreed set of names, and every one resolves. Plus the
host-block DATA capability (review point 5): a host block takes DATA arguments
in its signature after ctx, resolved at compile to the bound DataHandle.

Author: B.G (09/2026)
"""

import importlib

import pytest

_EXPECTED = {
    "Backend", "require_backend", "Parameter", "Pool", "DataHandle",
    "KernelBuilder", "HelperBuilder", "GroupBuilder", "HostBlockBuilder",
    "freeze_helper", "freeze_kernel",
    "RoutineBuilder", "SequenceBuilder", "ProgramBuilder", "Dim",
    "Node", "FrozenKernel", "FrozenHelper", "FrozenGroup", "SlotKind",
    "CompiledKernel", "CompiledRoutine", "CompiledSequence", "new_uid",
    "share_leaf", "find_param_paths",
    "PyFastFlowError", "BuildError", "ContractError", "FrozenError", "BindError",
    "CompileError", "ParameterError", "PoolError", "ProgramError",
}


def test_core_all_is_exact():
    import pyfastflow.core as core

    assert set(core.__all__) == _EXPECTED
    for name in core.__all__:
        assert getattr(core, name) is not None


def test_core_context_all_is_exact():
    import pyfastflow.core.context as ctx

    assert set(ctx.__all__) == _EXPECTED
    for name in ctx.__all__:
        assert getattr(ctx, name) is not None


def test_cupy_mfd_topology_public_recipes():
    from pyfastflow.core import Backend
    from pyfastflow.flow import make_mfd_topology
    from pyfastflow.grid import make_grid_group

    be = Backend.from_name("cupy")
    grid = make_grid_group(be, topology="D8")
    surface = make_mfd_topology(be, grid, method="surface", n_flat=16)
    assert set(surface) == {"dirs_weights", "indegree_reset", "indegree_count"}

    ranked = make_mfd_topology(be, grid, method="cordonnier_rank", n_flat=16)
    assert set(ranked) == {
        "snapshot_receivers", "receiver_rank", "dirs_weights",
        "indegree_reset", "indegree_count",
    }
    bound = ranked["receiver_rank"].build()
    addresses = bound.addresses()
    assert ("init", "rec") in addresses
    assert ("forward", "ancestor_in") in addresses
    assert ("backward", "rank_out") in addresses
    bound.close()
    bound = ranked["dirs_weights"].build()
    addresses = bound.addresses()
    for name in ("z", "rec_initial", "rec", "rank", "dirs", "mfd_w"):
        assert (name,) in addresses
    bound.close()

    with pytest.raises(ValueError, match="method must be one of"):
        make_mfd_topology(be, grid, method="unknown", n_flat=16)
    with pytest.raises(ValueError, match="cupy-only"):
        other = Backend.from_name("taichi")
        make_mfd_topology(other, make_grid_group(other), n_flat=16)


def _taichi_available():
    try:
        import taichi  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _taichi_available(), reason="taichi not available")
def test_host_block_takes_data_argument():
    import taichi as ti

    ti.init(arch=ti.gpu)
    from pyfastflow.core.context.backends import Backend
    from pyfastflow.core.context.host_block import HostBlockBuilder

    be = Backend.from_name("taichi")
    pool = be.pool()

    def _hb(ctx, counts):
        # counts is a DATA argument, resolved to the bound DataHandle; the
        # device->host copy is visible right here in the block's own source.
        ctx.N.set(int(counts.to_numpy().sum()))

    hb = HostBlockBuilder(_hb).freeze()
    # PARAM N is derived (ctx.N.set) and counts is a DATA slot (signature)
    from pyfastflow.core.context.slot import SlotKind

    assert hb.slots.names(SlotKind.PARAM) == {"N"}
    assert hb.slots.names(SlotKind.DATA) == {"counts"}

    b = hb.build()
    n_p = be.ParameterCls("N", dtype="i32", mode="scalar", value=0, pool=pool)
    counts = pool.get_data(ti.i32, (5,))
    counts.from_numpy(__import__("numpy").array([1, 2, 3, 4, 5], dtype="int32"))
    b.bind("N", n_p)
    b.bind("counts", counts)
    run = b.compile()  # host block: name resolution, backend ignored
    run()
    assert int(n_p.read()) == 15
    b.close()
    counts.release()
    n_p.destroy()
    pool.clear_all(force=True)
