# Context

Tools for describing and preparing a computation.

`context` is the composition layer beneath PyFastFlow's feature modules. It
lets a component state which parameters, data arrays, and device helpers it
needs, then connect those pieces into kernels, routines, sequences, or a
whole `Program`. The numerical and backend-specific kernel bodies live in the
feature modules; context provides the common structure around them.

Most scientific workflows use this layer through a ready-made `Program` or
feature factory. Work here when combining components into a custom model, or
when writing a new reusable component for PyFastFlow.

## From recipe to runnable code

1. A builder records a reusable *structure*.
2. Freezing makes that structure immutable and safe to reuse.
3. Binding supplies the concrete parameters, data handles, and child pieces.
4. Compilation prepares the selected backend's callable; it can then run.

## Building blocks

### Data and configuration

| Component | Role | Example |
| --- | --- | --- |
| **Data** | A typed backend buffer (`DataHandle`) passed directly to a kernel as an input or output array. | A topographic elevation field, `z`. |
| **Parameter** | A named model value that may be a compile-time constant, a device scalar, or a device field. | A Manning roughness coefficient, uniform or spatially variable. |

### Computation and orchestration

| Component | Role | Example |
| --- | --- | --- |
| **Helper** | A reusable device-side function called from a kernel or another helper; it cannot run on its own. | A grid-neighbour lookup specialised for D4 or D8 connectivity. |
| **Kernel** | One callable device operation, with data arguments and any helpers it uses. | Compute the steepest-flow receiver for every grid cell. |
| **Routine** | An ordered, device-only series of kernels that share one set of bindings. | Initialise and update a drainage-accumulation pass. |
| **Sequence** | A host-driven schedule of kernels, routines, and host blocks; it can include loops and stopping conditions. | Repeat depression handling full routine until the terrain has no unresolved pits. |
| **Program** | A complete, stateful model instance that owns its parameters and data, and builds, binds, and runs its sequences. | A configured GraphFlood or landscape-evolution simulation for one DEM. |

## Files

- `parameter.py` defines typed parameters and their `const`, `scalar`, and
  `field` storage modes.
- `builder.py` defines builders for kernels, helpers, and groups, and derives
  their required slots from a template.
- `frozen.py` defines the immutable node recipes produced by builders.
- `bound.py` creates bindable instances of those recipes and resolves their
  named parameter, data, and child addresses.
- `routine.py` combines device kernels into one ordered, device-only launch
  sequence.
- `sequence.py` adds host-side steps and loops around kernels and routines.
- `host_block.py` defines host-side blocks used between device launches.
- `program.py` provides the stateful `Program` layer for owning data,
  parameters, and compiled sequences.
- `backends.py` defines the `Backend` object and selects backend-specific
  parameter, pool, and compiler implementations.
- `compile_shared.py` contains compile-time checks and the common compiled
  kernel wrapper.
- `compile_closure.py` compiles the Python-template path used by Taichi and
  Quadrants.
- `compile_cupy.py` compiles CUDA-source templates through CuPy.
- `_closure_backend.py` provides the shared parameter device-view machinery
  for Taichi and Quadrants.
- `taichi_backend.py` provides Taichi's parameter implementation.
- `quadrants_backend.py` provides Quadrants' parameter implementation.
- `cupy_backend.py` provides CuPy's parameter implementation and CUDA-source
  emission utilities.
- `contract.py` reads templates to determine the parameters, data, and child
  helpers they require.
- `ctx.py` defines the reserved `ctx` naming convention used by templates.
- `bk.py` supplies the backend intrinsic functions available as `ctx.bk` in
  Taichi and Quadrants templates.
- `slot.py` defines the named PARAM, DATA, and HELPER slots used throughout
  the build and bind phases.
- `errors.py` defines the common PyFastFlow exception hierarchy.
- `__init__.py` exports the public context API.
