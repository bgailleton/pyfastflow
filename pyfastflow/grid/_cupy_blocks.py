"""CUDA grid-helper templates for CuPy."""

import math

from ..core import freeze_helper as _helper, new_uid

_SQRT2 = math.sqrt(2.0)


def build_group(group, *, topology, boundary, nodata, outlet):
    """Compose the selected CuPy grid helpers onto ``group``."""
    t = f"pf{new_uid()}"
    d8 = topology == "D8"

    row = _helper(f"__device__ int {t}_row(int i) {{ return i / $ctx.NX.get(0)$; }}")
    col = _helper(f"__device__ int {t}_col(int i) {{ return i % $ctx.NX.get(0)$; }}")
    index = _helper(f"__device__ int {t}_index(int row, int col) {{ return row * $ctx.NX.get(0)$ + col; }}")

    # ``k`` varies at runtime, so keep the direction lookup in constant memory.
    if d8:
        delta = _helper(
            f"""
__constant__ int {t}_DELTA_DR[8] = {{-1, -1, -1, 0, 0, 1, 1, 1}};
__constant__ int {t}_DELTA_DC[8] = {{-1, 0, 1, -1, 1, -1, 0, 1}};
__device__ void {t}_delta(int k, int* dr, int* dc) {{
    *dr = {t}_DELTA_DR[k];
    *dc = {t}_DELTA_DC[k];
}}
"""
        )
    else:
        delta = _helper(
            f"""
__constant__ int {t}_DELTA_DR[4] = {{-1, 0, 0, 1}};
__constant__ int {t}_DELTA_DC[4] = {{0, -1, 1, 0}};
__device__ void {t}_delta(int k, int* dr, int* dc) {{
    *dr = {t}_DELTA_DR[k];
    *dc = {t}_DELTA_DC[k];
}}
"""
        )

    row_wrap = (
        _helper(f"__device__ int {t}_row_wrap(int row) {{ return row; }}")
        if boundary != "periodic_NS"
        else _helper(
            f"""
__device__ int {t}_row_wrap(int row) {{
    int r = row;
    if (r < 0) r += $ctx.NY.get(0)$;
    else if (r >= $ctx.NY.get(0)$) r -= $ctx.NY.get(0)$;
    return r;
}}
"""
        )
    )
    col_wrap = (
        _helper(f"__device__ int {t}_col_wrap(int col) {{ return col; }}")
        if boundary != "periodic_EW"
        else _helper(
            f"""
__device__ int {t}_col_wrap(int col) {{
    int c = col;
    if (c < 0) c += $ctx.NX.get(0)$;
    else if (c >= $ctx.NX.get(0)$) c -= $ctx.NX.get(0)$;
    return c;
}}
"""
        )
    )
    wrap = _helper(
        f"""
__device__ void {t}_wrap(int row, int col, int* wrow, int* wcol) {{
    *wrow = $ctx.row_wrap(row)$;
    *wcol = $ctx.col_wrap(col)$;
}}
""",
        helpers={"row_wrap": row_wrap, "col_wrap": col_wrap},
    )

    row_edge_ok = (
        _helper(f"__device__ int {t}_row_edge_ok(int row) {{ return 1; }}")
        if boundary == "periodic_NS"
        else _helper(f"__device__ int {t}_row_edge_ok(int row) {{ return (row >= 0 && row < $ctx.NY.get(0)$) ? 1 : 0; }}")
    )
    col_edge_ok = (
        _helper(f"__device__ int {t}_col_edge_ok(int col) {{ return 1; }}")
        if boundary == "periodic_EW"
        else _helper(f"__device__ int {t}_col_edge_ok(int col) {{ return (col >= 0 && col < $ctx.NX.get(0)$) ? 1 : 0; }}")
    )

    source_ok = (
        _helper(f"__device__ int {t}_source_ok(int i) {{ return ($ctx.NODATA_MASK.get(i)$ == 1) ? 0 : 1; }}")
        if nodata
        else _helper(f"__device__ int {t}_source_ok(int i) {{ return 1; }}")
    )

    move_allowed = _helper(
        f"""
__device__ int {t}_move_allowed(int i, int k) {{
    int row = $ctx.row(i)$;
    int col = $ctx.col(i)$;
    int dr, dc;
    $ctx.delta(k, &dr, &dc)$;
    int a = $ctx.row_edge_ok(row + dr)$;
    int b = $ctx.col_edge_ok(col + dc)$;
    int c = $ctx.source_ok(i)$;
    return a * b * c;
}}
""",
        helpers={
            "row": row, "col": col, "delta": delta,
            "row_edge_ok": row_edge_ok, "col_edge_ok": col_edge_ok, "source_ok": source_ok,
        },
    )

    if nodata:
        valid = _helper(
            f"""
__device__ int {t}_valid(int j) {{
    if (j < 0 || j >= $ctx.NX.get(0)$ * $ctx.NY.get(0)$) return 0;
    return ($ctx.NODATA_MASK.get(j)$ == 1) ? 0 : 1;
}}
"""
        )
    else:
        valid = _helper(f"__device__ int {t}_valid(int j) {{ return (j >= 0 && j < $ctx.NX.get(0)$ * $ctx.NY.get(0)$) ? 1 : 0; }}")

    neighbour_raw = _helper(
        f"""
__device__ int {t}_neighbour_raw(int i, int k) {{
    int row = $ctx.row(i)$;
    int col = $ctx.col(i)$;
    int dr, dc;
    $ctx.delta(k, &dr, &dc)$;
    int wrow, wcol;
    $ctx.wrap(row + dr, col + dc, &wrow, &wcol)$;
    return $ctx.index(wrow, wcol)$;
}}
""",
        helpers={"row": row, "col": col, "delta": delta, "wrap": wrap, "index": index},
    )

    neighbour = _helper(
        f"""
__device__ int {t}_neighbour(int i, int k) {{
    int j = -1;
    if ($ctx.move_allowed(i, k)$ == 1) {{
        int cand = $ctx.neighbour_raw(i, k)$;
        if ($ctx.valid(cand)$ == 1) j = cand;
    }}
    return j;
}}
""",
        helpers={"move_allowed": move_allowed, "neighbour_raw": neighbour_raw, "valid": valid},
    )

    is_active = (
        _helper(f"__device__ int {t}_is_active(int i) {{ return ($ctx.NODATA_MASK.get(i)$ == 1) ? 0 : 1; }}")
        if nodata
        else _helper(f"__device__ int {t}_is_active(int i) {{ return 1; }}")
    )
    nodata_fn = _helper(
        f"__device__ int {t}_nodata(int i) {{ return 1 - $ctx.is_active(i)$; }}",
        helpers={"is_active": is_active},
    )

    row_is_edge = (
        _helper(f"__device__ int {t}_row_is_edge(int row) {{ return 0; }}")
        if boundary == "periodic_NS"
        else _helper(f"__device__ int {t}_row_is_edge(int row) {{ return (row == 0 || row == $ctx.NY.get(0)$ - 1) ? 1 : 0; }}")
    )
    col_is_edge = (
        _helper(f"__device__ int {t}_col_is_edge(int col) {{ return 0; }}")
        if boundary == "periodic_EW"
        else _helper(f"__device__ int {t}_col_is_edge(int col) {{ return (col == 0 || col == $ctx.NX.get(0)$ - 1) ? 1 : 0; }}")
    )
    is_on_edge = _helper(
        f"""
__device__ int {t}_is_on_edge(int i) {{
    int row = $ctx.row(i)$;
    int col = $ctx.col(i)$;
    return ($ctx.row_is_edge(row)$ == 1 || $ctx.col_is_edge(col)$ == 1) ? 1 : 0;
}}
""",
        helpers={"row": row, "col": col, "row_is_edge": row_is_edge, "col_is_edge": col_is_edge},
    )

    row_edge_code = (
        _helper(f"__device__ int {t}_row_edge_code(int row) {{ return -1; }}")
        if boundary == "periodic_NS"
        else _helper(
            f"""
__device__ int {t}_row_edge_code(int row) {{
    if (row == 0) return 0;
    if (row == $ctx.NY.get(0)$ - 1) return 3;
    return -1;
}}
"""
        )
    )
    col_edge_code = (
        _helper(f"__device__ int {t}_col_edge_code(int col) {{ return -1; }}")
        if boundary == "periodic_EW"
        else _helper(
            f"""
__device__ int {t}_col_edge_code(int col) {{
    if (col == 0) return 1;
    if (col == $ctx.NX.get(0)$ - 1) return 2;
    return -1;
}}
"""
        )
    )
    which_edge = _helper(
        f"""
__device__ int {t}_which_edge(int i) {{
    int row = $ctx.row(i)$;
    int col = $ctx.col(i)$;
    int code = $ctx.row_edge_code(row)$;
    if (code == -1) code = $ctx.col_edge_code(col)$;
    return code;
}}
""",
        helpers={"row": row, "col": col, "row_edge_code": row_edge_code, "col_edge_code": col_edge_code},
    )

    if outlet == "mask":
        can_out = _helper(f"__device__ int {t}_can_out(int i) {{ return ($ctx.OUTLET_MASK.get(i)$ == 1) ? 1 : 0; }}")
    else:
        can_out = _helper(
            f"__device__ int {t}_can_out(int i) {{ return $ctx.is_on_edge(i)$; }}",
            helpers={"is_on_edge": is_on_edge},
        )

    if d8:
        dist_from_k = _helper(
            f"""
__device__ float {t}_dist_from_k(int k) {{
    float d = $ctx.DX.get(0)$;
    if (k == 0 || k == 2 || k == 5 || k == 7) d = $ctx.DX.get(0)$ * {_SQRT2}f;
    return d;
}}
"""
        )
    else:
        dist_from_k = _helper(f"__device__ float {t}_dist_from_k(int k) {{ return $ctx.DX.get(0)$; }}")

    row_dist = (
        _helper(f"__device__ int {t}_row_dist(int raw) {{ return abs(raw); }}")
        if boundary != "periodic_NS"
        else _helper(
            f"""
__device__ int {t}_row_dist(int raw) {{
    int d = abs(raw);
    int n = $ctx.NY.get(0)$;
    return d < (n - d) ? d : (n - d);
}}
"""
        )
    )
    col_dist = (
        _helper(f"__device__ int {t}_col_dist(int raw) {{ return abs(raw); }}")
        if boundary != "periodic_EW"
        else _helper(
            f"""
__device__ int {t}_col_dist(int raw) {{
    int d = abs(raw);
    int n = $ctx.NX.get(0)$;
    return d < (n - d) ? d : (n - d);
}}
"""
        )
    )

    diag_line = f"        else if (dr == 1 && dc == 1) out = $ctx.DX.get(0)$ * {_SQRT2}f;\n" if d8 else ""
    dist_between = _helper(
        f"""
__device__ float {t}_dist_between(int i, int j) {{
    float out = -1.0f;
    if (j >= 0) {{
        int ri = $ctx.row(i)$;
        int ci = $ctx.col(i)$;
        int rj = $ctx.row(j)$;
        int cj = $ctx.col(j)$;
        int dr = $ctx.row_dist(rj - ri)$;
        int dc = $ctx.col_dist(cj - ci)$;
        if (dr == 0 && dc == 1) out = $ctx.DX.get(0)$;
        else if (dr == 1 && dc == 0) out = $ctx.DX.get(0)$;
{diag_line}    }}
    return out;
}}
""",
        helpers={"row": row, "col": col, "row_dist": row_dist, "col_dist": col_dist},
    )

    neighbour_and_distance = _helper(
        f"""
__device__ void {t}_neighbour_and_distance(int i, int k, int* j_out, float* d_out) {{
    int j = $ctx.neighbour(i, k)$;
    float d = -1.0f;
    if (j != -1) d = $ctx.dist_from_k(k)$;
    *j_out = j;
    *d_out = d;
}}
""",
        helpers={"neighbour": neighbour, "dist_from_k": dist_from_k},
    )

    group.compose("neighbour", neighbour)
    group.compose("neighbour_raw", neighbour_raw)
    group.compose("nodata", nodata_fn)
    group.compose("is_active", is_active)
    group.compose("can_out", can_out)
    group.compose("dist_from_k", dist_from_k)
    group.compose("dist_between_nodes", dist_between)
    group.compose("is_on_edge", is_on_edge)
    group.compose("which_edge", which_edge)
    group.compose("neighbour_and_distance", neighbour_and_distance)
