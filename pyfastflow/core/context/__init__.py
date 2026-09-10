"""Public interfaces for PyFastFlow computation assembly."""

from ..pool.base import DataHandle, Pool, PoolError, new_uid
from .backends import Backend, require_backend
from .bound import BindError
from .builder import (
    GroupBuilder,
    HelperBuilder,
    KernelBuilder,
    freeze_helper,
    freeze_kernel,
    find_param_paths,
    share_leaf,
)
from .compile_shared import CompileError, CompiledKernel
from .contract import ContractError
from .errors import ParameterError, PyFastFlowError
from .frozen import FrozenError, FrozenGroup, FrozenHelper, FrozenKernel, Node
from .host_block import HostBlockBuilder
from .parameter import Parameter
from .program import Dim, ProgramBuilder
from .routine import CompiledRoutine, RoutineBuilder
from .sequence import CompiledSequence, SequenceBuilder
from .slot import BuildError, ProgramError, SlotKind

__all__ = [
    # backends / values / storage
    "Backend", "require_backend", "Parameter", "Pool", "DataHandle",
    # builders
    "KernelBuilder", "HelperBuilder", "GroupBuilder", "HostBlockBuilder",
    "freeze_helper", "freeze_kernel",
    "RoutineBuilder", "SequenceBuilder", "ProgramBuilder", "Dim",
    # frozen node + compiled results
    "Node", "FrozenKernel", "FrozenHelper", "FrozenGroup", "SlotKind",
    "CompiledKernel", "CompiledRoutine", "CompiledSequence", "new_uid",
    # build-phase sharing helpers
    "share_leaf", "find_param_paths",
    # error hierarchy
    "PyFastFlowError", "BuildError", "ContractError", "FrozenError", "BindError",
    "CompileError", "ParameterError", "PoolError", "ProgramError",
]
