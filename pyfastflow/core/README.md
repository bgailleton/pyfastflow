# Core

Composition and runtime tools for PyFastFlow.

`core` is the framework beneath PyFastFlow's geomorphology modules. It does
not implement a flow-routing, flooding, or landscape-evolution method itself.
Instead, it provides the common machinery for assembling those building
blocks into a simulation: connecting kernels, helpers, parameters, and data;
assembling them into routine, sequence or even whole plug-and-play programs
then preparing them for a chosen backend.

Most scientific workflows start in `grid`, `flow`, or `graphflood`. Come here
when you are composing those pieces into a custom simulation, or adding a
reusable building block of your own.

## Context and pool

**`context/`** describes a computation: its reusable device helpers and
kernels, the parameters and arrays they need, and how they are connected and
compiled. Feature modules provide the backend-specific kernel bodies; context
handles the structure around them.

**`pool/`** manages backend array storage. It allocates and reuses typed,
shaped buffers, and tracks who owns them. The same pool serves long-lived
model data and temporary working arrays.
