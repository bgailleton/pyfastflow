"""
"reconstruct_epsilon": the cupy-only variant of fill-by-reconstruction that
make_graphflood's kind="vanilla_mfd" actually runs, on top of ../flow's
unmodified make_fill_reconstruct/make_fill_reconstruct_solver.

Why this exists
------------------
../flow's fill_reconstruct converges `filled` to an exact grayscale
reconstruction - genuinely, numerically flat across a resolved depression
(a real lake surface IS flat; see _closure_reconstruct.py's own module
docstring). SFD accumulation never has a problem with that: it walks
`parent` directly (a single, guaranteed-acyclic receiver per node, recorded
live during the relax kernel's own frontier expansion - never re-derived
from `filled`'s elevation values afterward), so a flat region is no
different from a sloped one as far as SFD's own graph walk is concerned.

MFD topology (../graphflood/_cupy_mfd_topology.py) is different: it derives
`dirs`/`mfd_w` from `filled`'s own elevation differences
(`slope(filled[i], filled[j]) > 0`), which is exactly zero for every
neighbour pair inside a flat region - every cell in a resolved lake gets
*zero* outgoing MFD edges, and accumulation can never cross it to reach the
real outlet on the far side.

This module fixes that without touching _cupy_mfd_topology.py's own
slope-based logic at all: it builds a per-cell perturbation `dist` that
grows strictly with distance-from-the-outlet along the exact direction
`parent` already established (never re-derived from slope), and hands
`dist` - alongside the real `filled` - to build_mfd_topology's own
dirs_weights kernel. That kernel feeds `dist` into the slope helper's
second additive term (its `h` argument):
`slope(filled[i], dist[i], filled[j], dist[j], k)` =
`((filled[i] - filled[j]) + (dist[i] - dist[j])) / d`, so the tie on a
numerically-flat lake surface (`filled[i] == filled[j]`) is broken by
`dist[i] - dist[j]` alone. Crucially `dist` is never added into `filled`
here: the two only ever meet as a *difference* inside the slope helper,
evaluated at `dist`'s own ~1e-7 magnitude. Folding `dist` up into
`filled`'s magnitude (0.3 ... 1e3 on a real DEM) would quantise the per-
hop signal - exactly one ULP of `filled` - to whole multiples of that ULP
and collapse adjacent hops to the same value; keeping the addition out of
`filled` is what avoids that. `filled` itself (used by h_from_filled to
derive the real, un-perturbed depth field) is untouched.

Why the perturbation is self-scaling (ULP-based), not a fixed constant
--------------------------------------------------------------------------
An earlier version added a fixed `MFD_EPSILON` (1e-5) per hop. That is
wrong in general: whether a fixed constant survives being added to
`filled[i]` in float32 depends entirely on `filled[i]`'s own magnitude -
float32's ULP grows with magnitude (`spacing(1.0) ~ 1.2e-7`,
`spacing(1300.0) ~ 1.5e-4`). On a real DEM with elevations in the
thousands, `1e-5` is *smaller than the ULP itself* for small hop counts,
so `filled[i] + 1e-5*hops[i]` silently rounds back to exactly `filled[i]`
unchanged - the tie never actually breaks. Confirmed empirically on
topotoolbox's "greenriver" DEM (z ~ 1300, ULP ~ 1.5e-4, chosen epsilon
1e-5): 37018 interior cells still ended up with zero outgoing MFD edges
despite the fix supposedly being in place.

The fix here seeds each cell's own contribution not with a fixed constant
but with `spacing(filled[i])` - the smallest representable float32
increment *at that cell's own magnitude*, via `nextafterf`. This is the
theoretical minimum perturbation that is guaranteed to survive floating-
point addition regardless of the terrain's absolute elevation, so it works
identically on a synthetic z~1 test grid and a real z~1300 DEM alike, with
no per-terrain tuning. It also bounds worst-case distortion of real relief
to the smallest amount physically possible: total accumulated perturbation
along a chain of `k` cells at similar elevation is `~k * spacing(filled)`,
proportional to how many cells share that one contiguous flat run, not to
a chosen constant that has to compromise between "big enough to survive
rounding" and "small enough not to matter" the way a fixed epsilon does.

A local `nextafter`/single epsilon bump per cell cannot do this alone: it
only breaks a tie between one cell and its own immediate neighbour, not
along a multi-hop chain across an entire flat (two cells several hops
apart inside the same lake could still tie, or invert). The perturbation
needs to scale with actual hop-distance to the outlet along `parent` -
`dist` below accumulates exactly that (each cell's own local ULP, summed
along the parent chain), computed the same pointer-jumping way
../flow/_closure_depressions.py's build_propagate_basin_iter/_final finds a
basin's root (path-compressing ancestor jumps, `ceil(log2(n_flat))+1`
rounds to guarantee full convergence for a chain up to n_flat long) - just
summing a per-cell float contribution along the way instead of counting
hops or finding a root id.

Author: B.G (08/2026)
"""

from ..core import FrozenKernel, KernelBuilder, new_uid


def build_hops_init(*, n_flat: int) -> FrozenKernel:
    """
    hops_init FrozenKernel, data args (parent, filled, dist, anc): dist[i]
    = 0.0 if `parent[i] == i` (the outlet itself) else `spacing(filled[i])`
    - the smallest float32 increment strictly greater than `filled[i]`
    itself (`nextafterf(filled[i], 1e30f) - filled[i]`), self-scaling to
    this cell's own elevation magnitude rather than a fixed constant (see
    the module docstring for why that matters). `anc[i] = parent[i]` - the
    seed state build_hops_jump's pointer-jumping rounds relax from.

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    FrozenKernel

    Author: B.G (08/2026)
    """
    t = f"gfh{new_uid()}"
    return (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_hops_init(const int* parent, const float* filled, float* dist, int* anc) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int p = parent[i];
    if (p == i) {{
        dist[i] = 0.0f;
    }} else {{
        float f = filled[i];
        dist[i] = nextafterf(f, 1.0e30f) - f;
    }}
    anc[i] = p;
}}
""", domain=n_flat).freeze()
    )


def build_hops_jump(*, n_flat: int) -> FrozenKernel:
    """
    hops_jump FrozenKernel, data args (dist_in, anc_in, dist_out, anc_out):
    one round of pointer-jumping path compression, reading the previous
    round's fully-settled state from `dist_in`/`anc_in` and writing this
    round's into `dist_out`/`anc_out` - `dist_out[i] = dist_in[i] +
    dist_in[anc_in[i]]; anc_out[i] = anc_in[anc_in[i]]` whenever `anc_in[i]
    != i` (not yet reached the outlet), `dist_out[i] = dist_in[i]`/
    `anc_out[i] = anc_in[i]` unchanged otherwise.

    This reads and writes disjoint buffer pairs deliberately - an earlier,
    in-place version (`dist[i] += dist[anc[i]]`) raced within a single
    kernel launch: thread i reading `dist[anc[i]]` has no ordering
    guarantee against thread `anc[i]` updating its own `dist` entry the
    same round, corrupting the sum for whichever threads happen to read a
    partially-updated neighbour. Unlike
    ../flow/_closure_depressions.py's build_propagate_basin_iter (which
    only races on *which* still-valid ancestor gets read, benign since any
    ancestor read is still a correct, if less-compressed, one), an
    accumulating `+=` has no such tolerance - the fix is the standard
    double-buffered pointer-jumping shape, not a relaxed round count.

    A caller alternates two composed occurrences of this kernel (bound
    oppositely, "forward": in=dist/anc, out=dist2/anc2; "backward": in=
    dist2/anc2, out=dist/anc - the same swap-free ping-pong
    ../flow/_cupy_accum.py's build_pointer_jump_push uses for its own
    "step_a"/"step_b" alternation) for a caller-rounded-up-to-even number
    of rounds, so the final, fully-converged result always lands back in
    `dist`/`anc` regardless of round count - see make_graphflood's own
    `hops_rounds` computation.

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    FrozenKernel

    Author: B.G (08/2026)
    """
    t = f"gfj{new_uid()}"
    return (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_hops_jump(
    const float* dist_in, const int* anc_in, float* dist_out, int* anc_out)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int a = anc_in[i];
    if (a != i) {{
        dist_out[i] = dist_in[i] + dist_in[a];
        anc_out[i] = anc_in[a];
    }} else {{
        dist_out[i] = dist_in[i];
        anc_out[i] = a;
    }}
}}
""", domain=n_flat).freeze()
    )
