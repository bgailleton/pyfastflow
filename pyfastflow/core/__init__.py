"""
New backend-agnostic core (Parameter/Helper/Kernel/Pool ABCs + backends).

The public surface is re-exported from `core.context`; feature packages import
these names from `pyfastflow.core`. See core/context/__init__.py.

Author: B.G (07/2026)
"""

from .context import (
    Backend,
    require_backend,
    BindError,
    BuildError,
    CompileError,
    CompiledKernel,
    CompiledRoutine,
    CompiledSequence,
    ContractError,
    DataHandle,
    Dim,
    FrozenError,
    FrozenGroup,
    FrozenHelper,
    FrozenKernel,
    GroupBuilder,
    HelperBuilder,
    HostBlockBuilder,
    KernelBuilder,
    freeze_helper,
    freeze_kernel,
    Node,
    Parameter,
    ParameterError,
    Pool,
    PoolError,
    ProgramBuilder,
    ProgramError,
    PyFastFlowError,
    RoutineBuilder,
    SequenceBuilder,
    SlotKind,
    find_param_paths,
    share_leaf,
    new_uid,
)
from .context import __all__ as _CONTEXT_ALL

__all__ = list(_CONTEXT_ALL)
