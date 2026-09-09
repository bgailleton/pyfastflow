"""
Tier 1: the framework exception hierarchy.

Every exception the framework raises on purpose derives from the one root
`PyFastFlowError`, so a caller can catch any framework-level mistake with a
single except clause. This pins that invariant down against the phase bases
drifting off the root as later units move them around.

Author: B.G (09/2026)
"""

from pyfastflow.core.context.bk import BkError
from pyfastflow.core.context.bound import BindError
from pyfastflow.core.context.compile_shared import CompileError
from pyfastflow.core.context.contract import ContractError
from pyfastflow.core.context.errors import ParameterError, PyFastFlowError
from pyfastflow.core.context.frozen import FrozenError
from pyfastflow.core.context.routine import RoutineBuilderError
from pyfastflow.core.context.sequence import SequenceBuilderError
from pyfastflow.core.context.slot import (
    BuildError,
    ProgramBuilderError,
    ProgramError,
    SlotGroupError,
)
from pyfastflow.core.pool.base import PoolError

_ALL_FRAMEWORK_ERRORS = [
    BuildError,
    SlotGroupError,
    ProgramBuilderError,
    ContractError,
    FrozenError,
    RoutineBuilderError,
    SequenceBuilderError,
    BindError,
    CompileError,
    BkError,
    PoolError,
    ProgramError,
    ParameterError,
]


def _data_template(ctx, z):
    return None


def test_every_framework_error_derives_from_root():
    for cls in _ALL_FRAMEWORK_ERRORS:
        assert issubclass(cls, PyFastFlowError), cls


def test_build_phase_errors_stay_under_build_error():
    for cls in (
        SlotGroupError,
        ProgramBuilderError,
        ContractError,
        FrozenError,
        RoutineBuilderError,
        SequenceBuilderError,
    ):
        assert issubclass(cls, BuildError), cls


def test_root_is_a_plain_exception():
    assert issubclass(PyFastFlowError, Exception)
    # not a TypeError: wrong-python-type bugs stay outside the framework tree.
    assert not issubclass(PyFastFlowError, TypeError)


def test_parameter_constructors_accept_short_dtype_tags_without_storage():
    """Const parameters resolve the public dtype tags on every backend."""
    from pyfastflow.core.context.cupy_backend import CupyParameter
    from pyfastflow.core.context.quadrants_backend import QuadrantsParameter
    from pyfastflow.core.context.taichi_backend import TaichiParameter

    for cls in (TaichiParameter, QuadrantsParameter, CupyParameter):
        param = cls("p", dtype="f32", mode="const", value=1.5, pool=None)
        assert param.value == 1.5


def test_data_slots_reject_raw_arrays_before_compile():
    """DATA carries handles through bind; raw storage is compiler-only."""
    import numpy as np
    import pytest

    from pyfastflow.core.context.builder import KernelBuilder

    bound = KernelBuilder(_data_template).data("z").freeze().build()
    with pytest.raises(BindError, match="expected a DataHandle"):
        bound.bind("z", np.zeros(1, dtype=np.float32))


def test_feature_factories_require_backend_objects():
    """Feature factories own the strict Backend boundary in Unit 8."""
    import pytest

    from pyfastflow.grid import make_grid_group

    with pytest.raises(TypeError, match="require a Backend"):
        make_grid_group("cupy")


def test_graphflood_recipe_is_inert_and_requires_backend_object():
    """GraphFlood keeps live state out of its Unit 8 structure factory."""
    import pytest

    from pyfastflow.core import Backend
    from pyfastflow.graphflood import FrozenGraphflood, make_graphflood
    from pyfastflow.grid import make_grid_group

    be = Backend.from_name("cupy")
    grid = make_grid_group(be)
    frozen, params = make_graphflood(be, grid, n_flat=4, nx=2, ny=2)
    assert isinstance(frozen, FrozenGraphflood)
    assert params == {}
    assert frozen.config["n_flat"] == 4
    with pytest.raises(TypeError, match="require a Backend"):
        make_graphflood("cupy", grid, n_flat=4, nx=2, ny=2)
