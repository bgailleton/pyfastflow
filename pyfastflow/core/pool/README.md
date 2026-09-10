# Pool

Backend array allocation and lifetime management.

`pool` owns the backend-resident arrays used by a PyFastFlow computation. It
keeps buffers grouped by their type and shape, so a released buffer can be
used again instead of allocating new device memory for every operation.

Most workflows use a pool indirectly: `Program` obtains persistent model data
and temporary workspace from it, while `Parameter` uses it for stored values.
Use a pool directly when a custom component needs an array whose lifetime you
control, such as scratch space shared by several operations.

## Handles and lifetimes

A pool returns a `DataHandle`, not a raw backend array. The handle identifies
one typed, shaped buffer, provides NumPy transfer methods, and records whether
the buffer is checked out. It can be bound to a kernel as `DATA`; the compiler
unwraps the backend array only when the kernel is prepared.

Return a directly acquired handle with `release_data`, or prefer the scoped
form `with pool.data(dtype, shape) as handle:`. Releasing a handle keeps its
memory available for reuse; clearing a pool destroys its available buffers.

## Files

- `base.py` defines the common `Pool` and `DataHandle` interfaces and their
  lifecycle rules.
- `_bucketed_pool.py` implements buffer reuse by grouping handles by dtype and
  shape.
- `_fields_handle.py` shares the field-backed handle implementation used by
  Taichi and Quadrants.
- `taichi_handle.py` selects Taichi for the shared field-backed handle
  implementation.
- `quadrants_handle.py` selects Quadrants for the shared field-backed handle
  implementation.
- `cupy_handle.py` implements a handle backed by a CuPy array.
- `taichi_pool.py` defines the pool that allocates Taichi handles.
- `quadrants_pool.py` defines the pool that allocates Quadrants handles.
- `cupy_pool.py` defines the pool that allocates CuPy handles.
- `__init__.py` marks this directory as the pool package.
