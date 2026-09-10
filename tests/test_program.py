"""Unit 9 Program orchestration contract."""

import numpy as np
import pytest
import taichi as ti

from pyfastflow.core import Backend, HostBlockBuilder, KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.core.context.slot import ProgramError
from pyfastflow.flow import make_receivers
from pyfastflow.grid import make_grid_group, make_grid_parameters


def _increment(ctx):
    ctx.counter.set(ctx.counter.read() + 1)


def _copy(ctx, src: ti.template(), dst: ti.template()):
    for i in src:
        dst[i] = src[i]


def _counter_program():
    return (
        ProgramBuilder("CounterProgram")
        .param("counter", "scalar", "i32", value=0)
        .add("tick", lambda be, bundles, config: HostBlockBuilder(_increment).freeze(), bind={"counter": "counter"})
        .freeze()
    )


def test_program_rejects_backend_strings_before_allocating():
    with pytest.raises(TypeError, match="Backend"):
        _counter_program()("taichi")


def test_program_host_orchestration_state_and_close():
    import taichi as ti

    ti.init(arch=ti.cpu)
    prog = _counter_program()(Backend.from_name("taichi"))
    prog.tick(3)
    assert prog.counter.read() == 3
    snapshot = prog.state()
    assert snapshot["schema"] == "CounterProgram" and snapshot["version"] == 1
    prog.load_state(snapshot)
    prog.close()
    prog.close()
    with pytest.raises(ProgramError, match="closed"):
        prog.counter.read()


def _grid_structure(be, *, nx, ny):
    return make_grid_group(be, topology="D8", boundary="normal", outlet="edge")


def _grid_params(be, pool, *, nx, ny):
    return make_grid_parameters(be, pool, nx, ny, 1.0, topology="D8", outlet="edge")


def test_program_bundle_expands_shared_grid_root():
    """A shared grid root must bind its redirected canonical leaves too."""
    import taichi as ti

    ti.init(arch=ti.cpu)
    P = (
        ProgramBuilder("ReceiverProgram")
        .dim("ny").dim("nx")
        .config("ny", default=3).config("nx", default=4)
        .data("z", "f32", (Dim("ny"), Dim("nx")), role="input")
        .data("rec", "i32", (Dim("ny"), Dim("nx")), role="output")
        .bundle("grid", _grid_structure, _grid_params, dims=("nx", "ny"))
        .add(
            "receivers",
            lambda be, bundles, config: make_receivers(be, bundles["grid"], topology="D8", mode="steepest")["receivers"],
            bind={"grid": "grid", "z": "z", "rec": "rec"},
        )
        .freeze()
    )
    prog = P(Backend.from_name("taichi"))
    prog.z.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
    prog.receivers()
    assert prog.rec.to_numpy().shape == (3, 4)
    prog.close()


def test_program_adopt_keeps_foreign_storage_alive():
    import taichi as ti

    ti.init(arch=ti.cpu)
    P = (
        ProgramBuilder("AdoptProgram")
        .dim("n").config("n", default=4)
        .data("z", "f32", (Dim("n"),), flat=False)
        .freeze()
    )
    foreign = ti.field(dtype=ti.f32, shape=(4,))
    foreign.from_numpy(np.arange(4, dtype=np.float32))
    prog = P(Backend.from_name("taichi"))
    prog.z.adopt(foreign)
    assert prog.z.array is foreign
    prog.close()
    np.testing.assert_array_equal(foreign.to_numpy(), np.arange(4, dtype=np.float32))


def test_program_restores_temp_placeholder_after_run():
    import taichi as ti

    ti.init(arch=ti.cpu)
    P = (
        ProgramBuilder("TempProgram")
        .dim("n").config("n", default=4)
        .data("src", "f32", (Dim("n"),), flat=False)
        .data("work", "f32", (Dim("n"),), lifetime="temp", flat=False)
        .add(
            "copy",
            lambda be, bundles, config: KernelBuilder(_copy, domain="src").freeze(),
            bind={"src": "src", "dst": "work"},
        )
        .freeze()
    )
    prog = P(Backend.from_name("taichi"))
    prog.src.from_numpy(np.arange(4, dtype=np.float32))
    prog.copy()
    state = prog._states["copy"]
    assert state.compiled.data_at("dst") is state.placeholders[0]
    assert prog._pool.stats()["in_use"] == 2  # persistent source + reserved placeholder
    prog.close()
