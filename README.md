# PyFastFlow

> **Experimental.** PyFastFlow is under active development. The core model is
> usable, but public names and high-level Programs may still change before 1.0.

**Composable GPU routines for geomorphology and shallow-water flow.**

PyFastFlow has two layers:

- a backend-aware computation core for assembling, binding, compiling, and
  owning GPU kernels and multi-kernel algorithms;
- geomorphology building blocks for grids, routing, local-minima resolution,
  drainage accumulation, procedural terrain, visualisation, and GraphFlood.

The same assembly model targets **Taichi**, **Quadrants**, and **CuPy**. Taichi
and Quadrants use Python closure kernels; CuPy emits specialised CUDA C++ and
runs it with `RawKernel`. Algorithms can share the same high-level contract
while retaining an implementation suited to each backend.

[![License](https://img.shields.io/badge/license-CeCILL%20v2.1-red.svg)](./LICENSE)

## Why PyFastFlow?

Flow routing is awkward GPU work. Receiver trees, drainage accumulation,
depression handling, and shallow-water solvers contain non-local dependencies
that do not map to a single elementwise kernel.

PyFastFlow provides the machinery needed to express them as complete GPU
algorithms:

- `Parameter` represents a compile-time constant, mutable scalar, or spatial
  field;
- helpers and groups package reusable device logic such as D4/D8 neighbours,
  boundaries, outlets, and noise sampling;
- kernels, routines, and sequences compose work ranging from one launch to a
  host-controlled iterative solver;
- `ProgramBuilder` packages configuration, arrays, parameters, algorithm
  dispatch, temporary storage, and cleanup behind a small user-facing API.

Composition happens once. The resulting kernels are specialised for the
selected backend, topology, parameter layout, and algorithm.

## Quick start: a complete flow Program

The experimental CuPy Programs provide the shortest route from a DEM to SFD
drainage accumulation:

```python
import numpy as np

from pyfastflow.core import Backend
from pyfastflow.experimental.programs.flow import SFDFlowProgram

ny = nx = 1024
dem = np.random.default_rng(42).random((ny, nx), dtype=np.float32)
backend = Backend.from_name("cupy")

with SFDFlowProgram(
    backend,
    nx=nx,
    ny=ny,
    dx=30.0,
    local_minima="cordonnier_carve",
    accumulation="pointer_jump_push",
) as flow:
    flow.z.from_numpy(dem)
    flow.route()
    flow.resolve_minima()
    flow.accumulate()
    drainage = flow.drainage.to_numpy()
```

The high-level choices currently exposed by `SFDFlowProgram` are:

| Stage | Choices |
| --- | --- |
| Local minima | `none`, `cordonnier_carve`, `cordonnier_jump`, `reconstruct_epsilon` |
| SFD accumulation | `pointer_jump_push` (or `pj`), `rake_compress` |

`reconstruct_epsilon` constructs the acyclic receiver forest itself, so it is
used without the preceding `flow.route()` call.

For Perlin terrain and zero-copy composition between two Programs, run:

```bash
python examples/flow_acc_sfd_lm_program.py
```

Algorithm selection is available directly from the command line:

```bash
python examples/flow_acc_sfd_lm_program.py \
    --local-minima cordonnier_jump \
    --accumulation rake_compress \
    --no-plot
```

See [`examples/flow_acc_sfd_lm_program.py`](./examples/flow_acc_sfd_lm_program.py)
for the full example, including explicit cleanup without a context manager.

The CuPy `MFDFlowProgram` packages persistent Kahn accumulation with raw,
reconstructed-surface, or rank-gated Cordonnier topology. A complete zero-copy
Perlin → MFD → multishade composition is runnable with:

```bash
python examples/flow_acc_mfd_lm_program.py
```

Use `--local-minima none` to retain raw-surface sinks, or
`--local-minima reconstruct_epsilon` to use filling and flat resolution. The
example selects `hillshade` or four-direction `multishade` through the
standalone `HillshadeProgram`.

MFD Programs use max-normalized `uint8` routing scores by default. Pass
`quantized_weight=False` when constructing either `MFDFlowProgram` or
`GraphFloodProgram` to retain precomputed `float32` weights. Effective weights
are normalized by their integer sum during accumulation, so the quantized path
still partitions the complete discharge at every node.

The experimental CuPy `GraphFloodProgram` combines the same rank-gated
Cordonnier topology with persistent MFD accumulation and a Manning depth
update:

```python
from pyfastflow.experimental.programs.graphflood import GraphFloodProgram

with GraphFloodProgram(backend, nx=nx, ny=ny, dx=dx) as flood:
    flood.z.from_numpy(dem.astype("float32"))
    flood.reset_h()
    flood.precipitation.set(50e-3 / 3600)  # m s-1
    flood.friction_coefficient.set(0.033)
    flood.friction_exponent.set(2 / 3)
    flood.dt.set(1e-2)
    flood.run_n_step(100)
    depth = flood.h.to_numpy()
```

Each step rebuilds the hydraulic surface and its Cordonnier-carved,
rank-gated MFD graph before accumulating rainfall and updating water depth.
The only friction-law option is currently `friction_law="manning"`; it is
already a construction-time Program choice so more laws can be added without
changing the execution API.

## Programs and memory ownership

A Program owns its parameters, persistent buffers, compiled operations, and an
internal memory pool unless a pool is supplied explicitly. A context manager is
the simplest way to release those resources, but it is not required:

```python
flow = SFDFlowProgram(backend, nx=nx, ny=ny)
try:
    flow.z.from_numpy(dem)
    flow.route()
    flow.resolve_minima()
    flow.accumulate()
finally:
    flow.close()
```

`close()` releases the Program's ownership. CuPy may retain freed CUDA blocks
in its process-wide memory cache for reuse. If memory must be returned to CUDA
immediately after all relevant Programs have closed, the application can call
`cupy.get_default_memory_pool().free_all_blocks()`.

## Current building blocks

- **Grid:** D4/D8 topology, normal and periodic boundaries, no-data masks, and
  edge or masked outlets.
- **Flow routing:** steepest and stochastic receivers.
- **Drainage accumulation:** atomic SFD, rake-and-compress, pointer-jump/push,
  and CuPy persistent-kernel MFD accumulation.
- **MFD topology:** filled-surface routing, or CuPy rank-gated routing directly
  over a Cordonnier-carved receiver graph without topographic filling.
- **Local minima:** Cordonnier basin labelling with carve or jump rerouting,
  plus fill-and-reconstruct solvers.
- **Hydraulics:** GraphFlood SFD, unstable flow, and CuPy MFD variants, with
  configurable friction laws and outlet behaviour.
- **Terrain and utilities:** white/Perlin noise, hillshading, elementwise
  operations, scan, reduction, and reusable math/bit-packing helpers.

Not every algorithm exists on every backend. Backend-specific capabilities are
validated when their factory or Program is built. The ready-made
`PerlinNoiseProgram`, `HillshadeProgram`, `SFDFlowProgram`, and `MFDFlowProgram`
are currently CuPy-only; the lower-level
feature factories cover Taichi, Quadrants, and CuPy where implementations exist.

The rank-gated MFD path is assembled with
`make_mfd_topology(..., method="cordonnier_rank")`. Snapshot the initial
receivers, apply optimized Cordonnier carving, compute `receiver_rank`, then
build directions and indegrees before running
`make_accumulation(..., method="persistent_mfd")`. Rerouted cells retain one
forced carved link; all other MFD links must strictly decrease receiver rank,
which gives the persistent Kahn accumulator an acyclic graph.

## Working at the composition layer

The high-level Programs are built from the same public core available to model
authors:

```python
from pyfastflow.core import (
    Backend,
    GroupBuilder,
    KernelBuilder,
    ProgramBuilder,
    RoutineBuilder,
    SequenceBuilder,
)

backend = Backend.from_name("cupy")
```

The normal lifecycle is:

1. create reusable parameter/helper structure;
2. compose and freeze kernels, routines, or sequences;
3. bind their named slots to concrete parameters and arrays;
4. compile for a `Backend` and execute;
5. close the compiled objects and release their storage.

Model authors who want configuration, dispatch, and ownership handled as one
unit can define a reusable class with `ProgramBuilder`. Complete backend-level
examples live under [`examples/core`](./examples/core), and a fully authored
Program is shown in
[`examples/core/program/sfd_drainage.py`](./examples/core/program/sfd_drainage.py).

## Backends

Create backends through `Backend.from_name(...)`; feature factories accept the
resulting `Backend` object rather than a backend-name string.

```python
# CuPy (no separate runtime initialisation)
backend = Backend.from_name("cupy")

# Taichi
import taichi as ti
ti.init(arch=ti.gpu)
backend = Backend.from_name("taichi")

# Quadrants
import quadrants as qd
qd.init(arch=qd.gpu)
backend = Backend.from_name("quadrants")
```

## Installation

Install the source tree in editable mode:

```bash
git clone https://github.com/bgailleton/pyfastflow.git
cd pyfastflow
python -m pip install -e .
```

Install the GPU backend appropriate for your system in the same environment.
Taichi is currently declared by the package; CuPy must match the installed CUDA
toolkit, and Quadrants must be installed separately when that backend is used.
The current source is developed and tested primarily on modern Python versions
(Python 3.10 or newer is recommended).

For development:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Project status

The v1-style core and feature factories are the active implementation. Code in
`pyfastflow.legacy` is retained for reference and older command-line tools; it
is not the API demonstrated above. Ready-made Programs currently live under
`pyfastflow.experimental` while their user-facing contracts settle.

Useful starting points:

- [`examples/flow_acc_sfd_lm_program.py`](./examples/flow_acc_sfd_lm_program.py):
  high-level Perlin → routing → local minima → accumulation;
- [`examples/flow_acc_sfd_lm.py`](./examples/flow_acc_sfd_lm.py): the same type
  of pipeline assembled directly from feature factories;
- [`examples/flow_acc_sfd_lm_raw_cupy.py`](./examples/flow_acc_sfd_lm_raw_cupy.py)
  and
  [`examples/flow_acc_sfd_lm_raw_quadrants.py`](./examples/flow_acc_sfd_lm_raw_quadrants.py):
  hard-coded baselines for measuring the cost of the machinery;
- [`examples/core/graphflood`](./examples/core/graphflood): GraphFlood examples;
- [`examples/core/lem`](./examples/core/lem): landscape-evolution examples.

## License and authors

PyFastFlow is distributed under the CeCILL v2.1 license. See [`LICENSE`](./LICENSE).

Boris Gailleton (Géosciences Rennes) · Guillaume Cordonnier (Inria)
