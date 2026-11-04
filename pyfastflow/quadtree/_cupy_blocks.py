"""CUDA quadtree-helper templates for CuPy.

Author: B.G (11/2026)
"""

import math

from ..core import freeze_helper as _helper, new_uid

_SQRT2 = math.sqrt(2.0)
_FINE_FACE = math.sqrt(2.5)
_FINE_CORNER = 1.5 * _SQRT2
NODATA_BIT = 2


def build_group(group, *, topology, nodata):
    """Compose the CuPy quadtree helpers onto ``group``."""
    t = f"pf{new_uid()}"
    d8 = topology == "D8"
    corner = "(k == 0 || k == 3 || k == 8 || k == 11)" if d8 else "0"

    n_cells = _helper(f"__device__ int {t}_n() {{ return $ctx.CAPACITY.get(0)$; }}")
    n_leaves = _helper(f"__device__ int {t}_n_leaves() {{ return $ctx.N.get(0)$; }}")
    level = _helper(f"__device__ int {t}_level(int i) {{ return $ctx.LEVEL.get(i)$; }}")
    reverse = _helper(f"__device__ int {t}_reverse(int k) {{ return $ctx.N_NEIGHBOURS.get(0)$ - 1 - k; }}")

    neighbour_raw = _helper(
        f"""
__device__ int {t}_neighbour_raw(int i, int k) {{
    unsigned int m = $ctx.SLOTS.get(i)$;
    unsigned int bit = 1u << k;
    if (!(m & bit)) return -1;
    int p = $ctx.PTR.get(i)$ + __popc(m & (bit - 1u));
    return $ctx.ADJ.get(p)$;
}}
"""
    )
    if nodata:
        neighbour = _helper(
            f"""
__device__ int {t}_neighbour(int i, int k) {{
    int j = $ctx.neighbour_raw(i, k)$;
    if (j < 0) return -1;
    int fi = $ctx.FLAGS.get(i)$;
    int fj = $ctx.FLAGS.get(j)$;
    return ((fi | fj) & {NODATA_BIT}) ? -1 : j;
}}
""",
            helpers={"neighbour_raw": neighbour_raw},
        )
    else:
        neighbour = _helper(
            f"__device__ int {t}_neighbour(int i, int k) {{ return $ctx.neighbour_raw(i, k)$; }}",
            helpers={"neighbour_raw": neighbour_raw},
        )

    nbr_begin = _helper(f"__device__ int {t}_nbr_begin(int i) {{ return $ctx.PTR.get(i)$; }}")
    nbr_end = _helper(f"__device__ int {t}_nbr_end(int i) {{ return $ctx.PTR.get(i + 1)$; }}")
    nbr_at = _helper(f"__device__ int {t}_nbr_at(int p) {{ return $ctx.ADJ.get(p)$; }}")

    is_active = _helper(f"__device__ int {t}_is_active(int i) {{ return ($ctx.FLAGS.get(i)$ & {NODATA_BIT}) ? 0 : 1; }}")
    nodata_fn = _helper(f"__device__ int {t}_nodata(int i) {{ return ($ctx.FLAGS.get(i)$ & {NODATA_BIT}) ? 1 : 0; }}")
    can_out = _helper(f"__device__ int {t}_can_out(int i) {{ return ($ctx.FLAGS.get(i)$ & 1) ? 1 : 0; }}")

    dx = _helper(
        f"""
__device__ float {t}_dx(int i) {{
    int l = $ctx.LEVEL.get(i)$;
    return $ctx.DX0.get(0)$ * (float)(1 << l);
}}
"""
    )
    area = _helper(
        f"""
__device__ float {t}_area(int i) {{
    float d = $ctx.dx(i)$;
    return d * d;
}}
""",
        helpers={"dx": dx},
    )
    cx = _helper(
        f"""
__device__ float {t}_cx(int i) {{
    int l = $ctx.LEVEL.get(i)$;
    return $ctx.DX0.get(0)$ * ((float)$ctx.X.get(i)$ + 0.5f * (float)(1 << l));
}}
"""
    )
    cy = _helper(
        f"""
__device__ float {t}_cy(int i) {{
    int l = $ctx.LEVEL.get(i)$;
    return $ctx.DX0.get(0)$ * ((float)$ctx.Y.get(i)$ + 0.5f * (float)(1 << l));
}}
"""
    )

    link_dist = _helper(
        f"""
__device__ float {t}_link_dist(int i, int j, int k) {{
    int li = $ctx.LEVEL.get(i)$;
    int lj = $ctx.LEVEL.get(j)$;
    float d = $ctx.DX0.get(0)$ * (float)(1 << (li < lj ? li : lj));
    int corner = {corner};
    if (li == lj) return corner ? d * {_SQRT2}f : d;
    return d * (corner ? {_FINE_CORNER}f : {_FINE_FACE}f);
}}
"""
    )
    dist = _helper(
        f"""
__device__ float {t}_dist(int i, int k) {{
    int j = $ctx.neighbour_raw(i, k)$;
    if (j < 0) return $ctx.dx(i)$;
    return $ctx.link_dist(i, j, k)$;
}}
""",
        helpers={"neighbour_raw": neighbour_raw, "dx": dx, "link_dist": link_dist},
    )
    width = _helper(
        f"""
__device__ float {t}_width(int i, int k) {{
    int j = $ctx.neighbour_raw(i, k)$;
    int li = $ctx.LEVEL.get(i)$;
    int lj = li;
    if (j >= 0) lj = $ctx.LEVEL.get(j)$;
    float d = $ctx.DX0.get(0)$ * (float)(1 << (li < lj ? li : lj));
    return {corner} ? d * {_SQRT2}f : d;
}}
""",
        helpers={"neighbour_raw": neighbour_raw},
    )
    flow_width = _helper(
        f"""
__device__ float {t}_flow_width(int i, int k) {{
    float d = $ctx.dx(i)$;
    return {corner} ? d * {_SQRT2}f : d;
}}
""",
        helpers={"dx": dx},
    )
    dist_between = _helper(
        f"""
__device__ float {t}_dist_between(int i, int j) {{
    if (j < 0) return -1.0f;
    unsigned int m = $ctx.SLOTS.get(i)$;
    int p = $ctx.PTR.get(i)$;
    while (m) {{
        if ($ctx.ADJ.get(p)$ == j) return $ctx.link_dist(i, j, __ffs((int)m) - 1)$;
        m &= m - 1u;
        ++p;
    }}
    return -1.0f;
}}
""",
        helpers={"link_dist": link_dist},
    )
    neighbour_and_distance = _helper(
        f"""
__device__ void {t}_neighbour_and_distance(int i, int k, int* j_out, float* d_out) {{
    int j = $ctx.neighbour(i, k)$;
    float d = -1.0f;
    if (j != -1) d = $ctx.link_dist(i, j, k)$;
    *j_out = j;
    *d_out = d;
}}
""",
        helpers={"neighbour": neighbour, "link_dist": link_dist},
    )

    group.compose("n", n_cells)
    group.compose("n_leaves", n_leaves)
    group.compose("level", level)
    group.compose("reverse", reverse)
    group.compose("neighbour_raw", neighbour_raw)
    group.compose("neighbour", neighbour)
    group.compose("nbr_begin", nbr_begin)
    group.compose("nbr_end", nbr_end)
    group.compose("nbr_at", nbr_at)
    group.compose("is_active", is_active)
    group.compose("nodata", nodata_fn)
    group.compose("can_out", can_out)
    group.compose("dx", dx)
    group.compose("area", area)
    group.compose("cx", cx)
    group.compose("cy", cy)
    group.compose("dist", dist)
    group.compose("width", width)
    group.compose("flow_width", flow_width)
    group.compose("dist_between_nodes", dist_between)
    group.compose("neighbour_and_distance", neighbour_and_distance)
