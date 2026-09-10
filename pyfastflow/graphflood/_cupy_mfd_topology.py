"""CUDA multiple-flow topology helpers for GraphFlood."""

from ..core import FrozenKernel, KernelBuilder, new_uid
from ..flow._cupy_receivers import build_distance_slope_helpers


def build_mfd_topology(*, grid, n_flat: int, topology: str, diagonal_partition_correction: bool) -> dict:
    """
    Three FrozenKernels: "dirs_weights" (data args (filled, dist, dirs,
    mfd_w): the bitmask/weight computation described in the module
    docstring),
    "indegree_reset" (data arg (indegree,): indegree[i] = 0 - must run
    before "indegree_count" every call, dirs/mfd_w/indegree all being
    recomputed fresh every GraphFlood step as the surface evolves) and
    "indegree_count" (data args (dirs, indegree): atomic-adds 1 into
    indegree[neighbour] for every bit set in dirs[i], via
    `ctx.grid.neighbour_raw` - trusted the same way _cupy_mfd_accum.py's
    own persistent kernel trusts it, since every bit `dirs` sets already
    passed a bounds-checked `ctx.grid.neighbour` in "dirs_weights").

    A caller runs all three, in order, every step, then
    `init_frontier_mfd` (../flow/_cupy_mfd_accum.py, host-side) to compact
    the zero-indegree cells into a frontier before launching
    `build_persistent_mfd`'s "accum" kernel.

    Parameters
    ----------
    grid : FrozenGroup
    n_flat : int
    topology : str
        "D4" or "D8" - sizes `mfd_w` (`n_flat * n_neighbours`) at the
        caller's own allocation, not here; only affects `slope`'s diagonal
        correction.
    diagonal_partition_correction : bool

    Returns
    -------
    dict
        {"dirs_weights": FrozenKernel, "indegree_reset": FrozenKernel,
        "indegree_count": FrozenKernel}.

    """
    slope = build_distance_slope_helpers(
        grid, topology=topology, diagonal_partition_correction=diagonal_partition_correction
    )["slope_from_values_k"]
    t = f"gfm{new_uid()}"

    dirs_weights_body = f"""
extern "C" __global__ void {t}_mfd_dirs_weights(const float* filled, const float* dist, unsigned char* dirs, float* mfd_w) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    unsigned char mask = 0;
    float slopes[8];
    float sum_s = 0.0f;
    if (!$ctx.grid.can_out(i)$ && !$ctx.grid.nodata(i)$) {{
        for (int k = 0; k < nk; k++) {{
            int j = $ctx.grid.neighbour(i, k)$;
            float s = 0.0f;
            if (j != -1) {{
                s = $ctx.slope(filled[i], dist[i], filled[j], dist[j], k)$;
                if (s < 0.0f) s = 0.0f;
            }}
            slopes[k] = s;
            if (s > 0.0f) {{ mask |= (1 << k); sum_s += s; }}
        }}
    }}
    for (int k = 0; k < nk; k++) {{
        mfd_w[i * nk + k] = sum_s > 0.0f ? slopes[k] / sum_s : 0.0f;
    }}
    dirs[i] = mask;
}}
"""

    dirs_weights_kb = KernelBuilder(dirs_weights_body, domain=n_flat)
    dirs_weights_kb.compose("grid", grid).compose("slope", slope)
    dirs_weights_kb.share_identical("grid")

    indegree_reset: FrozenKernel = (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_mfd_indegree_reset(int* indegree) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    indegree[i] = 0;
}}
""", domain=n_flat).freeze()
    )

    indegree_count_body = f"""
extern "C" __global__ void {t}_mfd_indegree_count(const unsigned char* dirs, int* indegree) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    unsigned char mask = dirs[i];
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        if (!(mask & (1 << k))) continue;
        int j = $ctx.grid.neighbour_raw(i, k)$;
        atomicAdd(&indegree[j], 1);
    }}
}}
"""
    indegree_count_kb = KernelBuilder(indegree_count_body, domain=n_flat)
    indegree_count_kb.compose("grid", grid)

    return {
        "dirs_weights": dirs_weights_kb.freeze(),
        "indegree_reset": indegree_reset,
        "indegree_count": indegree_count_kb.freeze(),
    }
