"""CUDA epsilon reconstruction helper for GraphFlood."""

from ..core import FrozenKernel, KernelBuilder, new_uid


def build_hops_init(*, n_flat: int) -> FrozenKernel:
    """
    hops_init FrozenKernel, data args (parent, filled, dist, anc): dist[i]
    = 0.0 if `parent[i] == i` (the outlet itself) else `spacing(filled[i])`
    - the smallest float32 increment strictly greater than `filled[i]`
    itself (``nextafterf(filled[i], 1e30f) - filled[i]``), so the increment
    scales with local elevation magnitude. ``anc[i] = parent[i]`` seeds the
    subsequent pointer-jumping rounds.

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    FrozenKernel

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
