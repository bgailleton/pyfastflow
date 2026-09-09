"""
The one exception root of the framework, plus the phase-error bases that
carry no state of their own.

`PyFastFlowError` is the single ancestor of every exception the framework
raises on purpose. A caller that wants to catch any framework mistake -
regardless of the phase it came from - catches `PyFastFlowError`. Genuine
wrong-python-type arguments stay `TypeError`; they are bugs in the calling
code, not framework conditions, and are deliberately left outside this tree.

The phase bases each live in the module that owns their phase and are
re-parented onto `PyFastFlowError` there, so their raise sites are unchanged:

  BuildError      build phase (slot.py); ContractError, FrozenError,
                  SlotGroupError, RoutineBuilderError, SequenceBuilderError,
                  ProgramBuilderError are its subclasses.
  BindError       bind phase (bound.py).
  CompileError    compile phase (compile_shared.py).
  BkError         intrinsics-node misuse (bk.py).
  PoolError       device-buffer pool (pool/base.py).
  ProgramError    program run time (slot.py).
  ParameterError  host-facing Parameter misuse (below); its raise sites in
                  parameter.py are settled in a later unit.

Both classes defined here hold no state and add no behaviour; they exist only
to root the tree.

Author: B.G (09/2026)
"""


class PyFastFlowError(Exception):
    """
    Root of every exception the framework raises on purpose. Catch this to
    catch any framework-level mistake across every phase. See the module
    docstring for the phase bases that derive from it.

    Author: B.G (09/2026)
    """


class ParameterError(PyFastFlowError):
    """
    Raised on host-facing Parameter misuse - writing a const, reading a field
    on the host, or destroying one while a binding still holds it.

    Author: B.G (09/2026)
    """
