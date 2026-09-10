"""CUDA templates for flow receivers."""

from ..core import HelperBuilder, KernelBuilder, SlotKind, new_uid, share_leaf


def build_distance_slope_helpers(grid, *, topology: str, diagonal_partition_correction: bool):
    """
    dist_from_k_corrected/dist_between_nodes_corrected/slope_from_values_k/
    slope_between_nodes for the cupy backend - see
    _closure_receivers.py's own (identical structure, CUDA text instead of
    python defs).

    Returns {name: HelperBuilder}.

    """
    d8 = topology == "D8"
    correct = diagonal_partition_correction and d8
    t = f"fr{new_uid()}"

    if correct:
        dist_from_k_corrected = HelperBuilder(
            f"""
__device__ float {t}_dist_from_k_corrected(int k) {{
    float d = $ctx.grid.dist_from_k(k)$;
    if (k == 0 || k == 2 || k == 5 || k == 7) {{
        d = d / 1.4142135623730951f;
    }}
    return d;
}}
"""
        ).compose("grid", grid).freeze()
        dist_between_nodes_corrected = HelperBuilder(
            f"""
__device__ float {t}_dist_between_nodes_corrected(int i, int j) {{
    float d = $ctx.grid.dist_between_nodes(i, j)$;
    if (d > $ctx.grid.DX.get(0)$ * 1.1f) {{
        d = d / 1.4142135623730951f;
    }}
    return d;
}}
"""
        ).compose("grid", grid).freeze()
    else:
        dist_from_k_corrected = HelperBuilder(
            f"__device__ float {t}_dist_from_k_corrected(int k) {{ return $ctx.grid.dist_from_k(k)$; }}"
        ).compose("grid", grid).freeze()
        dist_between_nodes_corrected = HelperBuilder(
            f"__device__ float {t}_dist_between_nodes_corrected(int i, int j) {{ return $ctx.grid.dist_between_nodes(i, j)$; }}"
        ).compose("grid", grid).freeze()

    slope_from_values_k = HelperBuilder(
        f"""
__device__ float {t}_slope_from_values_k(float zi, float hi, float zj, float hj, int k) {{
    return ((zi - zj) + (hi - hj)) / $ctx.dist_from_k_corrected(k)$;
}}
"""
    ).compose("dist_from_k_corrected", dist_from_k_corrected).freeze()
    slope_between_nodes = HelperBuilder(
        f"""
__device__ float {t}_slope_between_nodes(float vi, float vj, int i, int j) {{
    return (vi - vj) / $ctx.dist_between_nodes_corrected(i, j)$;
}}
"""
    ).compose("dist_between_nodes_corrected", dist_between_nodes_corrected).freeze()

    return {
        "dist_from_k_corrected": dist_from_k_corrected,
        "dist_between_nodes_corrected": dist_between_nodes_corrected,
        "slope_from_values_k": slope_from_values_k,
        "slope_between_nodes": slope_between_nodes,
    }


def build_rand_unit(hash_u32):
    """
    rand_unit(i, k) HelperBuilder, wiring its own `SEED` PARAM slot and
    composing the caller-supplied `hash_u32` (../noise's public hash helper)
    rather than a private copy - see _closure_receivers.py's own docstring.

    """
    t = f"fr{new_uid()}"
    return (
        HelperBuilder(
            f"""
__device__ float {t}_rand_unit(int i, int k) {{
    unsigned int key = (unsigned int)$ctx.SEED.get(0)$;
    key ^= (unsigned int)i * 374761393u;
    key ^= (unsigned int)k * 668265263u;
    unsigned int hashed = $ctx.hash_u32(key)$;
    return (float)hashed / 4294967296.0f;
}}
"""
        ).compose("hash_u32", hash_u32).freeze()
    )


def build_receivers(
    *,
    grid,
    hash_u32,
    mode: str,
    topology: str,
    diagonal_partition_correction: bool,
    h_aware: bool,
):
    """
    Build one cupy `receivers` KernelBuilder plus the distance/slope (and,
    for mode="stochastic", rand_unit) HelperBuilders it is made of - see
    _closure_receivers.py's build_receivers (identical structure and
    sharing).

    Parameters
    ----------
    grid : FrozenGroup
    hash_u32 : FrozenHelper
        Required, and only used, when mode="stochastic".
    mode : str
        "steepest" or "stochastic".
    topology : str
        "D4" or "D8".
    diagonal_partition_correction : bool
    h_aware : bool

    Returns
    -------
    dict
        {name: HelperBuilder/KernelBuilder}.

    """
    out = build_distance_slope_helpers(grid, topology=topology, diagonal_partition_correction=diagonal_partition_correction)
    slope = out["slope_from_values_k"]

    if mode == "stochastic":
        rand_unit = build_rand_unit(hash_u32)
        out["rand_unit"] = rand_unit
        stochastic_insert = """
                        if (tsr > 0.0f) {
                            tsr = $ctx.rand_unit(i, k)$ * sqrtf(tsr);
                        }"""
    else:
        stochastic_insert = ""

    t = f"fr{new_uid()}"
    if h_aware:
        args = "const float* z, const float* h, int* rec"
        slope_call = "$ctx.slope(z[i], h[i], z[j], h[j], k)$"
    else:
        args = "const float* z, int* rec"
        slope_call = "$ctx.slope(z[i], 0.0f, z[j], 0.0f, k)$"

    body = f"""
extern "C" __global__ void {t}_receivers({args}) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = $ctx.grid.NX.get(0)$ * $ctx.grid.NY.get(0)$;
    if (i >= n) return;

    if ($ctx.grid.can_out(i)$) {{
        rec[i] = i;
        return;
    }}

    int r = i;
    float sr = 0.0f;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(i, k)$;
        int valid = j != -1;
        float tsr = -1.0f;
        if (valid) {{
            tsr = {slope_call};{stochastic_insert}
        }}
        int better = valid && (tsr > sr);
        sr = better ? tsr : sr;
        r = better ? j : r;
    }}
    rec[i] = r;
}}
"""

    kb = KernelBuilder(body, domain="z")
    grid_param_names = grid.slots.names(SlotKind.PARAM)
    for name in grid_param_names:
        kb.param(name)
    kb.compose("grid", grid)
    kb.compose("slope", slope)
    if mode == "stochastic":
        kb.compose("rand_unit", out["rand_unit"])

    for name in grid_param_names:
        share_leaf(kb, name)

    out["receivers"] = kb.freeze()
    return out
