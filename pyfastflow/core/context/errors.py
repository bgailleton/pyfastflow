"""PyFastFlow exception roots and phase-specific error bases."""


class PyFastFlowError(Exception):
    """
    Root of every exception the framework raises on purpose. Catch this to
    catch any framework-level mistake across every phase. See the module
    docstring for the phase bases that derive from it.

    """


class ParameterError(PyFastFlowError):
    """
    Raised on host-facing Parameter misuse - writing a const, reading a field
    on the host, or destroying one while a binding still holds it.

    """
