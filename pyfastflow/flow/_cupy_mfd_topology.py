"""CuPy topology builders for persistent multiple-flow accumulation."""

import math

from ..core import FrozenKernel, KernelBuilder, SequenceBuilder, new_uid
from ._cupy_receivers import build_distance_slope_helpers


def _build_indegree(*, grid, n_flat: int, tag: str) -> dict:
    reset: FrozenKernel = KernelBuilder(
        f"""
extern "C" __global__ void {tag}_indegree_reset(int* indegree) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {n_flat}) indegree[i] = 0;
}}
""",
        domain=n_flat,
    ).freeze()

    count_kb = KernelBuilder(
        f"""
extern "C" __global__ void {tag}_indegree_count(const unsigned char* dirs, int* indegree) {{
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
""",
        domain=n_flat,
    )
    count_kb.compose("grid", grid)
    return {"indegree_reset": reset, "indegree_count": count_kb.freeze()}


def build_surface_mfd_topology(
    *, grid, n_flat: int, topology: str, diagonal_partition_correction: bool,
    quantized_weight: bool = False,
) -> dict:
    """Build MFD directions from a filled surface and flat-distance field."""
    slope = build_distance_slope_helpers(
        grid,
        topology=topology,
        diagonal_partition_correction=diagonal_partition_correction,
    )["slope_from_values_k"]
    tag = f"mfs{new_uid()}"
    weight_type = "unsigned char" if quantized_weight else "float"
    best_decl = "float max_s = 0.0f;" if quantized_weight else ""
    best_update = "if (s > max_s) max_s = s;" if quantized_weight else ""
    weight_write = (
        "mfd_w[i * nk + k] = slopes[k] > 0.0f "
        "? (unsigned char)max(1, __float2int_rn(255.0f * slopes[k] / max_s)) : 0;"
        if quantized_weight else
        "mfd_w[i * nk + k] = sum_s > 0.0f ? slopes[k] / sum_s : 0.0f;"
    )
    body = f"""
extern "C" __global__ void {tag}_dirs_weights(
    const float* filled, const float* dist, unsigned char* dirs, {weight_type}* mfd_w)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    unsigned char mask = 0;
    float slopes[8];
    float sum_s = 0.0f;
    {best_decl}
    if (!$ctx.grid.can_out(i)$ && !$ctx.grid.nodata(i)$) {{
        for (int k = 0; k < nk; k++) {{
            int j = $ctx.grid.neighbour(i, k)$;
            float s = 0.0f;
            if (j != -1) {{
                s = $ctx.slope(filled[i], dist[i], filled[j], dist[j], k)$;
                if (s < 0.0f) s = 0.0f;
            }}
            slopes[k] = s;
            if (s > 0.0f) {{ mask |= (1 << k); sum_s += s; {best_update} }}
        }}
    }} else {{
        for (int k = 0; k < nk; k++) slopes[k] = 0.0f;
    }}
    for (int k = 0; k < nk; k++) {{ {weight_write} }}
    dirs[i] = mask;
}}
"""
    kb = KernelBuilder(body, domain=n_flat)
    kb.compose("grid", grid).compose("slope", slope)
    kb.share_identical("grid")
    out = {"dirs_weights": kb.freeze()}
    out.update(_build_indegree(grid=grid, n_flat=n_flat, tag=tag))
    return out


def build_receiver_rank(*, n_flat: int):
    """Build a fixed-round pointer-jump sequence computing receiver depth.

    ``rank[i]`` is the number of receiver links from ``i`` to its terminal
    self-receiver. The receiver graph must therefore already be acyclic.
    The even ping-pong schedule leaves the result in the primary ``rank``
    and ``ancestor`` buffers.
    """
    tag = f"mfr{new_uid()}"
    init = KernelBuilder(
        f"""
extern "C" __global__ void {tag}_rank_init(
    const int* rec, int* ancestor, int* rank)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    ancestor[i] = rec[i];
    rank[i] = rec[i] == i ? 0 : 1;
}}
""",
        domain=n_flat,
    ).freeze()
    jump = KernelBuilder(
        f"""
extern "C" __global__ void {tag}_rank_jump(
    const int* ancestor_in, const int* rank_in,
    int* ancestor_out, int* rank_out)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int a = ancestor_in[i];
    ancestor_out[i] = ancestor_in[a];
    rank_out[i] = rank_in[i] + rank_in[a];
}}
""",
        domain=n_flat,
    ).freeze()

    rounds = math.ceil(math.log2(max(2, int(n_flat)))) + 1
    if rounds % 2:
        rounds += 1
    sb = SequenceBuilder()
    sb.add("init", init)
    sb.add("forward", jump)
    sb.add("backward", jump)
    sb.step("init")
    sb.loop(body=["forward", "backward"], max_times=rounds // 2)
    return sb.freeze()


def build_receiver_fill(*, n_flat: int):
    """Compute receiver distance and path-maximum elevation after carving.

    The final ``rank`` and ``filled`` fields define a lexicographic drainage
    potential: elevation decreases first, then receiver distance across flats.
    """
    tag = f"mff{new_uid()}"
    init = KernelBuilder(
        f"""
extern "C" __global__ void {tag}_fill_init(
    const int* rec, const float* z, int* ancestor, int* rank, float* filled)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    ancestor[i] = rec[i];
    rank[i] = rec[i] == i ? 0 : 1;
    filled[i] = z[i];
}}
""", domain=n_flat).freeze()
    jump = KernelBuilder(
        f"""
extern "C" __global__ void {tag}_fill_jump(
    const int* ancestor_in, const int* rank_in, const float* filled_in,
    int* ancestor_out, int* rank_out, float* filled_out)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int a = ancestor_in[i];
    if (a == i) {{
        ancestor_out[i] = i;
        rank_out[i] = rank_in[i];
        filled_out[i] = filled_in[i];
    }} else {{
        ancestor_out[i] = ancestor_in[a];
        rank_out[i] = rank_in[i] + rank_in[a];
        filled_out[i] = fmaxf(filled_in[i], filled_in[a]);
    }}
}}
""", domain=n_flat).freeze()

    rounds = math.ceil(math.log2(max(2, int(n_flat)))) + 1
    if rounds % 2:
        rounds += 1
    return (SequenceBuilder().add("init", init)
            .add("forward", jump).add("backward", jump).step("init")
            .loop(("forward", "backward"), max_times=rounds // 2).freeze())


def build_filled_rank_mfd_topology(
    *, grid, n_flat: int, topology: str, diagonal_partition_correction: bool,
    quantized_weight: bool = False,
) -> dict:
    """Build ordinary MFD on a filled elevation with rank as flat epsilon."""
    slope = build_distance_slope_helpers(
        grid, topology=topology,
        diagonal_partition_correction=diagonal_partition_correction,
    )["dist_from_k_corrected"]
    tag = f"mfc{new_uid()}"
    weight_type = "unsigned char" if quantized_weight else "float"
    best_decl = "float max_s = 0.0f;" if quantized_weight else ""
    best_update = "if (s > max_s) max_s = s;" if quantized_weight else ""
    weight_write = (
        "mfd_w[i * nk + k] = slopes[k] > 0.0f "
        "? (unsigned char)max(1, __float2int_rn(255.0f * slopes[k] / max_s)) : 0;"
        if quantized_weight else
        "mfd_w[i * nk + k] = sum_s > 0.0f ? slopes[k] / sum_s : 0.0f;"
    )
    body = f"""
extern "C" __global__ void {tag}_dirs_weights(
    const float* filled, const int* rank,
    unsigned char* dirs, {weight_type}* mfd_w)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    unsigned char mask = 0;
    float slopes[8];
    float sum_s = 0.0f;
    {best_decl}
    for (int k = 0; k < nk; ++k) slopes[k] = 0.0f;
    if (!$ctx.grid.can_out(i)$ && !$ctx.grid.nodata(i)$) {{
        float zi = filled[i];
        float ulp = nextafterf(zi, 1.0e30f) - zi;
        for (int k = 0; k < nk; ++k) {{
            int j = $ctx.grid.neighbour(i, k)$;
            if (j == -1) continue;
            float drop = 0.0f;
            if (zi > filled[j]) {{
                drop = zi - filled[j];
            }} else if (zi == filled[j] && rank[i] > rank[j]) {{
                drop = ulp * (float)(rank[i] - rank[j]);
            }}
            float s = drop > 0.0f ? drop / $ctx.distance(k)$ : 0.0f;
            slopes[k] = s;
            if (s > 0.0f) {{ mask |= (1 << k); sum_s += s; {best_update} }}
        }}
    }}
    for (int k = 0; k < nk; ++k) {{ {weight_write} }}
    dirs[i] = mask;
}}
"""
    kb = KernelBuilder(body, domain=n_flat)
    kb.compose("grid", grid).compose("distance", slope)
    kb.share_identical("grid")
    out = {"dirs_weights": kb.freeze()}
    out.update(_build_indegree(grid=grid, n_flat=n_flat, tag=tag))
    return out


def build_ranked_mfd_topology(
    *, grid, n_flat: int, topology: str, diagonal_partition_correction: bool,
    quantized_weight: bool = False,
) -> dict:
    """Build rank-gated MFD topology for a Cordonnier-carved SFD graph.

    Snapshot ``rec`` before carving, run the carve on ``rec``, compute rank
    on that final receiver graph, then build directions. Nodes whose receiver
    changed are forced along the carved SFD link. Other nodes retain all raw
    downslope MFD links that strictly decrease receiver rank.
    """
    slope = build_distance_slope_helpers(
        grid,
        topology=topology,
        diagonal_partition_correction=diagonal_partition_correction,
    )["slope_from_values_k"]
    tag = f"mfg{new_uid()}"
    weight_type = "unsigned char" if quantized_weight else "float"
    best_decl = "float max_s = 0.0f;" if quantized_weight else ""
    best_forced = "max_s = 1.0f;" if quantized_weight else ""
    best_update = "if (s > max_s) max_s = s;" if quantized_weight else ""
    weight_write = (
        "mfd_w[i * nk + k] = slopes[k] > 0.0f "
        "? (unsigned char)max(1, __float2int_rn(255.0f * slopes[k] / max_s)) : 0;"
        if quantized_weight else
        "mfd_w[i * nk + k] = sum_s > 0.0f ? slopes[k] / sum_s : 0.0f;"
    )
    snapshot: FrozenKernel = KernelBuilder(
        f"""
extern "C" __global__ void {tag}_snapshot_receivers(const int* rec, int* rec_initial) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {n_flat}) rec_initial[i] = rec[i];
}}
""",
        domain=n_flat,
    ).freeze()

    body = f"""
extern "C" __global__ void {tag}_dirs_weights(
    const float* z, const int* rec_initial, const int* rec, const int* rank,
    unsigned char* dirs, {weight_type}* mfd_w)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    unsigned char mask = 0;
    float slopes[8];
    float sum_s = 0.0f;
    {best_decl}
    for (int k = 0; k < nk; k++) slopes[k] = 0.0f;

    if (!$ctx.grid.can_out(i)$ && !$ctx.grid.nodata(i)$) {{
        if (rec_initial[i] != rec[i]) {{
            for (int k = 0; k < nk; k++) {{
                int j = $ctx.grid.neighbour(i, k)$;
                if (j == rec[i]) {{
                    mask = (unsigned char)(1 << k);
                    slopes[k] = 1.0f;
                    sum_s = 1.0f;
                    {best_forced}
                    break;
                }}
            }}
        }} else {{
            for (int k = 0; k < nk; k++) {{
                int j = $ctx.grid.neighbour(i, k)$;
                float s = 0.0f;
                if (j != -1 && rank[j] < rank[i]) {{
                    s = $ctx.slope(z[i], 0.0f, z[j], 0.0f, k)$;
                    if (s < 0.0f) s = 0.0f;
                }}
                slopes[k] = s;
                if (s > 0.0f) {{ mask |= (1 << k); sum_s += s; {best_update} }}
            }}
        }}
    }}
    for (int k = 0; k < nk; k++) {{ {weight_write} }}
    dirs[i] = mask;
}}
"""
    kb = KernelBuilder(body, domain=n_flat)
    kb.compose("grid", grid).compose("slope", slope)
    kb.share_identical("grid")
    out = {
        "snapshot_receivers": snapshot,
        "receiver_rank": build_receiver_rank(n_flat=n_flat),
        "dirs_weights": kb.freeze(),
    }
    out.update(_build_indegree(grid=grid, n_flat=n_flat, tag=tag))
    return out


# Private compatibility for the original GraphFlood import.
build_mfd_topology = build_surface_mfd_topology
