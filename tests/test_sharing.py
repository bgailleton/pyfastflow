"""
Unit 2: build-phase sharing - share()/share_identical(), leaf and root, with
and without as_, and the redirect/inspect surface it produces.

All structural (build/bind/inspect/addresses), no device compile, so no backend
init is needed: a Node tree and its walk are pure python. The Appendix A
diffusion routine exercises the same machinery end-to-end on every backend via
the hand-run scratchpad acceptance script.

Author: B.G (09/2026)
"""

import pytest

from pyfastflow.core.context.bound import BindError
from pyfastflow.core.context.backends import Backend
from pyfastflow.core.context.builder import GroupBuilder, HelperBuilder, KernelBuilder, share_leaf
from pyfastflow.core.context.routine import RoutineBuilder
from pyfastflow.core.context.slot import BuildError


def _row(ctx, i):
    return ctx.NX.get(i)


def _make_grid():
    """A group with an internal leaf share: NX read by two composed children, collapsed to the group's own NX."""
    row = HelperBuilder(_row).freeze()
    gb = GroupBuilder().param("NX").compose("r1", row).compose("r2", row)
    share_leaf(gb, "NX")  # r1.NX, r2.NX -> NX
    return gb.freeze()


def _kernel_using(grid):
    def _k(ctx, z):
        for i in z:
            z[i] = ctx.g.NX.get(0)

    return KernelBuilder(_k).compose("g", grid).freeze()


def test_group_internal_leaf_share_collapses():
    grid = _make_grid()
    addrs = {".".join(a) for a in grid.build().addresses()}
    assert addrs == {"NX"}  # r1.NX / r2.NX collapsed into NX


def test_identical_root_share_gives_one_set():
    grid = _make_grid()
    rb = RoutineBuilder().step("A", _kernel_using(grid)).step("B", _kernel_using(grid))
    rb.share_identical("A.g", as_="grid")
    b = rb.freeze().build()
    addrs = {".".join(a) for a in b.addresses()}
    assert addrs == {"grid.NX", "A.z", "B.z"}


def test_outermost_wins_nested_internal_share():
    # A.g.r1.NX is internally shared (grid) to A.g.NX; the routine root-share
    # then redirects the whole A.g subtree to grid - so the deep leaf resolves,
    # in one compressed hop, to grid.NX, never to a further-redirected address.
    grid = _make_grid()
    rb = RoutineBuilder().step("A", _kernel_using(grid)).step("B", _kernel_using(grid))
    rb.share_identical("A.g", as_="grid")
    b = rb.freeze().build()
    assert b._redirect[("A", "g", "r1", "NX")] == ("grid", "NX")
    assert b._redirect[("A", "g", "NX")] == ("grid", "NX")
    assert ("grid", "NX") in b.addresses()


def test_leaf_share_as_mints_synthetic_leaf():
    grid = _make_grid()
    rb = RoutineBuilder().step("A", _kernel_using(grid)).step("B", _kernel_using(grid))
    rb.share("A.z", "B.z", as_="zbuf")
    b = rb.freeze().build()
    addrs = {".".join(a) for a in b.addresses()}
    assert "zbuf" in addrs
    assert "A.z" not in addrs and "B.z" not in addrs
    # a DATA-handle bind at the synthetic canonical is visible through both sources
    pool = Backend.from_name("cupy").pool()
    handle = pool.get_data("f32", (1,))
    b.bind("zbuf", handle)
    assert b.value_at(("A", "z")) is handle
    assert b.value_at(("B", "z")) is handle
    b.close()
    pool.clear_all(force=True)


def test_leaf_share_no_as_keeps_canonical():
    grid = _make_grid()
    rb = RoutineBuilder().step("A", _kernel_using(grid)).step("B", _kernel_using(grid))
    rb.share("A.z", "B.z")  # A.z survives, B.z redirects
    b = rb.freeze().build()
    addrs = {".".join(a) for a in b.addresses()}
    assert "A.z" in addrs and "B.z" not in addrs
    assert b._redirect[("B", "z")] == ("A", "z")


def test_bind_redirected_address_raises_naming_canonical():
    grid = _make_grid()
    rb = RoutineBuilder().step("A", _kernel_using(grid)).step("B", _kernel_using(grid))
    rb.share_identical("A.g", as_="grid")
    b = rb.freeze().build()
    with pytest.raises(BindError, match="grid.NX"):
        b.bind("A.g.NX", object())


def test_root_at_resolves_real_and_synthetic():
    grid = _make_grid()
    rb = RoutineBuilder().step("A", _kernel_using(grid)).step("B", _kernel_using(grid))
    rb.share_identical("A.g", as_="grid")
    b = rb.freeze().build()
    assert b.root_at("grid") is grid  # synthetic canonical root
    assert b.root_at("A") is not None  # a real step root
    with pytest.raises(BindError):
        b.root_at("grid.NX")  # a leaf, not a root
    with pytest.raises(BindError):
        b.root_at("nope")


def test_as_collision_raises():
    grid = _make_grid()
    rb = RoutineBuilder().step("grid", _kernel_using(grid)).step("B", _kernel_using(grid))
    with pytest.raises(BuildError, match="collides"):
        rb.share("grid.g", "B.g", as_="grid")  # 'grid' already a step name


def test_reshare_same_path_raises():
    grid = _make_grid()
    rb = RoutineBuilder().step("A", _kernel_using(grid)).step("B", _kernel_using(grid))
    rb.share("A.z", "B.z", as_="zbuf")
    with pytest.raises(BuildError, match="already shared"):
        rb.share("A.z", "B.z", as_="zbuf2")


def test_share_identical_kind_must_be_root():
    grid = _make_grid()
    rb = RoutineBuilder().step("A", _kernel_using(grid)).step("B", _kernel_using(grid))
    with pytest.raises(BuildError, match="not a child root"):
        rb.share_identical("A.z", as_="q")  # a DATA leaf


def test_share_root_requires_identical_object():
    g1, g2 = _make_grid(), _make_grid()  # structurally alike, different objects
    rb = RoutineBuilder().step("A", _kernel_using(g1)).step("B", _kernel_using(g2))
    with pytest.raises(BuildError, match="identical frozen object"):
        rb.share("A.g", "B.g", as_="grid")


def test_share_identical_no_op_when_unique():
    grid = _make_grid()
    other = _make_grid()
    rb = RoutineBuilder().step("A", _kernel_using(grid)).step("B", _kernel_using(other))
    rb.share_identical("A.g", as_="grid")  # nothing else IS grid -> no-op
    addrs = {".".join(a) for a in rb.freeze().build().addresses()}
    assert "A.g.NX" in addrs  # not re-rooted, since no-op


def test_inspect_deterministic_and_reports_shares():
    grid = _make_grid()
    rb = RoutineBuilder().step("A", _kernel_using(grid)).step("B", _kernel_using(grid))
    rb.share_identical("A.g", as_="grid")
    rb.share("A.z", "B.z", as_="zbuf")
    b = rb.freeze().build()
    r1, r2 = b.inspect(), b.inspect()
    assert r1 == r2
    assert "A.g.* -> grid.*" in r1
    assert "B.g.* -> grid.*" in r1
    assert "A.z -> zbuf" in r1
