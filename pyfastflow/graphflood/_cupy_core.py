"""CUDA GraphFlood core templates."""

from ..core import FrozenKernel, KernelBuilder, new_uid
from ..flow._cupy_receivers import build_distance_slope_helpers
from ._cupy_friction import build_friction_qo


_OUTLET_BEHAVIORS = frozenset({"fixed_h", "free", "fixed_s"})


def build_compute_qo(
    *, grid, n_flat: int, topology: str, diagonal_partition_correction: bool,
    law: str = "manning", outlet_behavior: str = "fixed_h",
) -> FrozenKernel:
    """
    compute_qo FrozenKernel for the cupy backend - see _closure_core.py's
    build_compute_qo (identical contract, including `outlet_behavior`; all
    three behaviors reuse the same `friction` FrozenHelper call, just with
    a different second argument on a can_out node).

    Parameters
    ----------
    grid : FrozenGroup
    n_flat : int
    topology : str
        "D4" or "D8".
    diagonal_partition_correction : bool
    law : str, optional
        Default "manning".
    outlet_behavior : str, optional
        "fixed_h" (default), "free" or "fixed_s".

    Returns
    -------
    FrozenKernel

    Raises
    ------
    ValueError
        If `outlet_behavior` is not recognised.

    """
    if outlet_behavior not in _OUTLET_BEHAVIORS:
        raise ValueError(
            f"build_compute_qo: outlet_behavior must be one of {sorted(_OUTLET_BEHAVIORS)}, got {outlet_behavior!r}"
        )
    slope = build_distance_slope_helpers(
        grid, topology=topology, diagonal_partition_correction=diagonal_partition_correction
    )["slope_from_values_k"]
    friction = build_friction_qo(law, grid)
    t = f"gfc{new_uid()}"

    if outlet_behavior == "fixed_h":
        can_out_branch = """
    if ($ctx.grid.can_out(i)$) {
        Qo[i] = 0.0f;
        return;
    }"""
    elif outlet_behavior == "fixed_s":
        can_out_branch = """
    if ($ctx.grid.can_out(i)$) {
        float boundary_s = $ctx.BOUNDARY_SLOPE.get(0)$;
        Qo[i] = $ctx.friction(h[i], boundary_s)$;
        return;
    }"""
    else:  # "free"
        can_out_branch = ""

    body = f"""
extern "C" __global__ void {t}_compute_qo(const float* z, const float* h, float* Qo) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;{can_out_branch}
    float best_s = 0.0f;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(i, k)$;
        if (j != -1) {{
            float s = $ctx.slope(z[i], h[i], z[j], h[j], k)$;
            if (s > best_s) best_s = s;
        }}
    }}
    Qo[i] = $ctx.friction(h[i], best_s)$;
}}
"""

    kb = KernelBuilder(body, domain=n_flat)
    kb.compose("grid", grid).compose("slope", slope).compose("friction", friction)
    kb.share_identical("grid")
    return kb.freeze()


def build_apply_divergence(*, grid, n_flat: int, outlet_behavior: str = "fixed_h") -> FrozenKernel:
    """
    apply_divergence FrozenKernel for the cupy backend - see
    _closure_core.py's build_apply_divergence (identical contract,
    including `outlet_behavior`; "free" and "fixed_s" share one body and
    wire no `BOUNDARY_H`).

    Parameters
    ----------
    grid : FrozenGroup
    n_flat : int
    outlet_behavior : str, optional
        "fixed_h" (default), "free" or "fixed_s".

    Returns
    -------
    FrozenKernel

    Raises
    ------
    ValueError
        If `outlet_behavior` is not recognised.

    """
    if outlet_behavior not in _OUTLET_BEHAVIORS:
        raise ValueError(
            f"build_apply_divergence: outlet_behavior must be one of {sorted(_OUTLET_BEHAVIORS)}, "
            f"got {outlet_behavior!r}"
        )
    t = f"gfd{new_uid()}"

    can_out_branch = (
        """
    if ($ctx.grid.can_out(i)$) {
        h[i] = $ctx.BOUNDARY_H.get(0)$;
        return;
    }"""
        if outlet_behavior == "fixed_h"
        else ""
    )

    body = f"""
extern "C" __global__ void {t}_apply_divergence(float* h, const float* Q_in, const float* Qo) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;{can_out_branch}
    float dx = $ctx.grid.DX.get(0)$;
    float area = dx * dx;
    float dt = $ctx.DT.get(0)$;
    float d = (Q_in[i] - Qo[i]) / area * dt;
    float min_inc = $ctx.GF_MIN_INCREMENT.get(0)$;
    if (Q_in[i] > Qo[i] && d < min_inc) {{
        d = min_inc;
    }} else if (Qo[i] > Q_in[i] && d > -min_inc) {{
        d = -min_inc;
    }}
    float hh = h[i] + d;
    h[i] = hh > 0.0f ? hh : 0.0f;
}}
"""

    kb = KernelBuilder(body, domain=n_flat)
    kb.compose("grid", grid)
    return kb.freeze()


def build_make_surface(*, n_flat: int) -> FrozenKernel:
    """
    make_surface FrozenKernel for the cupy backend - see _closure_core.py's
    build_make_surface (identical contract).

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    FrozenKernel

    """
    t = f"gfs{new_uid()}"
    return (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_make_surface(const float* z, const float* h, float* surface) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    surface[i] = z[i] + h[i];
}}
""", domain=n_flat).freeze()
    )


def build_h_from_filled(*, n_flat: int) -> FrozenKernel:
    """
    h_from_filled FrozenKernel for the cupy backend - see
    _closure_core.py's build_h_from_filled (identical contract).

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
extern "C" __global__ void {t}_h_from_filled(const float* z, const float* filled, float* h) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    float hh = filled[i] - z[i];
    h[i] = hh > 0.0f ? hh : 0.0f;
}}
""", domain=n_flat).freeze()
    )


def build_reset_reconstruct_scratch(*, n_flat: int, counters_size: int) -> dict:
    """
    "counters"/"queued_gen" reset FrozenKernels for the cupy backend - see
    _closure_core.py's build_reset_reconstruct_scratch (identical
    contract/rationale).

    Parameters
    ----------
    n_flat : int
    counters_size : int
        `max_passes + 2` - the "counters" buffer's own length.

    Returns
    -------
    dict
        {"counters": FrozenKernel, "queued_gen": FrozenKernel}.

    """
    t = f"gfr{new_uid()}"
    counters_kb = (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_reset_counters(int* counters) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {counters_size}) return;
    counters[i] = 0;
}}
""", domain=counters_size).freeze()
    )
    queued_gen_kb = (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_reset_queued_gen(int* queued_gen) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    queued_gen[i] = -1;
}}
""", domain=n_flat).freeze()
    )
    return {"counters": counters_kb, "queued_gen": queued_gen_kb}


def build_distribute(*, grid, n_flat: int, topology: str, diagonal_partition_correction: bool) -> dict:
    """
    Two FrozenKernels for the cupy backend - see _closure_core.py's
    build_distribute for the algorithm. A single CUDA `__global__` has no
    portable grid-wide barrier the way one closure-backend kernel's two
    consecutive top-level `for` loops do (same reasoning
    ../flow/_cupy_accum.py's build_atomic splits "q_init"/"accum" on), so
    this is two real launches, keyed "zero" (data args (Q_next,): Q_next[i]
    = SOURCE.get(i)) and "route" (data args (z, h, Q_in, Q_next): the
    per-node redistribution) - a caller runs "zero" then "route", in that
    order, every step.

    Parameters
    ----------
    grid : FrozenGroup
    n_flat : int
    topology : str
        "D4" or "D8".
    diagonal_partition_correction : bool

    Returns
    -------
    dict
        {"zero": FrozenKernel, "route": FrozenKernel}.

    """
    slope = build_distance_slope_helpers(
        grid, topology=topology, diagonal_partition_correction=diagonal_partition_correction
    )["slope_from_values_k"]
    t = f"gfl{new_uid()}"

    zero_kb = (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_distribute_zero(float* Q_next) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    Q_next[i] = $ctx.SOURCE.get(i)$;
}}
""", domain=n_flat).freeze()
    )

    route_body = f"""
extern "C" __global__ void {t}_distribute_route(const float* z, float* h, const float* Q_in, float* Q_next) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if ($ctx.grid.can_out(i)$) return;
    float qi = Q_in[i];
    if (qi <= 0.0f) return;
    float slopes[8];
    float sum_s = 0.0f;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(i, k)$;
        float s = 0.0f;
        if (j != -1) {{
            s = $ctx.slope(z[i], h[i], z[j], h[j], k)$;
            if (s < 0.0f) s = 0.0f;
        }}
        slopes[k] = s;
        sum_s += s;
    }}
    if (sum_s <= 0.0f) {{
        atomicAdd(&Q_next[i], qi);
        atomicAdd(&h[i], $ctx.GF_MIN_INCREMENT.get(0)$);
    }} else {{
        for (int k = 0; k < nk; k++) {{
            int j = $ctx.grid.neighbour(i, k)$;
            if (j != -1 && slopes[k] > 0.0f) {{
                atomicAdd(&Q_next[j], qi * slopes[k] / sum_s);
            }}
        }}
    }}
}}
"""

    route_kb = KernelBuilder(route_body, domain=n_flat)
    route_kb.compose("grid", grid).compose("slope", slope)
    route_kb.share_identical("grid")

    return {"zero": zero_kb, "route": route_kb.freeze()}


def build_copy_q(*, n_flat: int) -> FrozenKernel:
    """
    copy_q FrozenKernel for the cupy backend - see _closure_core.py's
    build_copy_q (identical contract).

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    FrozenKernel

    """
    t = f"gfq{new_uid()}"
    return (
        KernelBuilder(
            f"""
extern "C" __global__ void {t}_copy_q(const float* Q_next, float* Q_in) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    Q_in[i] = Q_next[i];
}}
""", domain=n_flat).freeze()
    )
