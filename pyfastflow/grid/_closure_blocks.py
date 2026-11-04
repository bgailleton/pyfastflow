"""Python grid-helper templates for Taichi and Quadrants."""

from ..core import freeze_helper as _helper

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _row_tmpl(ctx, i):
    return i // ctx.NX.get(0)


def _col_tmpl(ctx, i):
    return i % ctx.NX.get(0)


def _index_tmpl(ctx, row, col):
    return row * ctx.NX.get(0) + col


# ---------------------------------------------------------------------------
# topology: _delta(k) - runtime if-ladder, k is per-call device data
# ---------------------------------------------------------------------------


def _delta_d4_tmpl(ctx, k):
    dr = 0
    dc = 0
    if k == 0:
        dr, dc = -1, 0
    elif k == 1:
        dr, dc = 0, -1
    elif k == 2:
        dr, dc = 0, 1
    else:
        dr, dc = 1, 0
    return dr, dc


def _delta_d8_tmpl(ctx, k):
    dr = 0
    dc = 0
    if k == 0:
        dr, dc = -1, -1
    elif k == 1:
        dr, dc = -1, 0
    elif k == 2:
        dr, dc = -1, 1
    elif k == 3:
        dr, dc = 0, -1
    elif k == 4:
        dr, dc = 0, 1
    elif k == 5:
        dr, dc = 1, -1
    elif k == 6:
        dr, dc = 1, 0
    else:
        dr, dc = 1, 1
    return dr, dc


# ---------------------------------------------------------------------------
# per-axis wrap (boundary): identity or periodic, chosen per axis
# ---------------------------------------------------------------------------


def _row_wrap_identity_tmpl(ctx, row):
    return row


def _row_wrap_periodic_tmpl(ctx, row):
    r = row
    if r < 0:
        r += ctx.NY.get(0)
    elif r >= ctx.NY.get(0):
        r -= ctx.NY.get(0)
    return r


def _col_wrap_identity_tmpl(ctx, col):
    return col


def _col_wrap_periodic_tmpl(ctx, col):
    c = col
    if c < 0:
        c += ctx.NX.get(0)
    elif c >= ctx.NX.get(0):
        c -= ctx.NX.get(0)
    return c


def _wrap_tmpl(ctx, row, col):
    return ctx._ROWWRAP(row), ctx._COLWRAP(col)


# ---------------------------------------------------------------------------
# per-axis edge gate (boundary): is a candidate coordinate still in range on
# that axis - a periodic axis never blocks, since it wraps instead.
# ---------------------------------------------------------------------------


def _row_edge_ok_bounded_tmpl(ctx, row):
    ok = 0
    if row >= 0 and row < ctx.NY.get(0):
        ok = 1
    return ok


def _row_edge_ok_periodic_tmpl(ctx, row):
    return 1


def _col_edge_ok_bounded_tmpl(ctx, col):
    ok = 0
    if col >= 0 and col < ctx.NX.get(0):
        ok = 1
    return ok


def _col_edge_ok_periodic_tmpl(ctx, col):
    return 1


# ---------------------------------------------------------------------------
# source-cell nodata gate: on/off
# ---------------------------------------------------------------------------


def _source_ok_always_tmpl(ctx, i):
    return 1


def _source_ok_nodata_tmpl(ctx, i):
    ok = 1
    if ctx.NODATA_MASK.get(i) == 1:
        ok = 0
    return ok


# ---------------------------------------------------------------------------
# _move_allowed(i, k): per-axis edge gate x source-cell nodata gate
# ---------------------------------------------------------------------------


def _move_allowed_tmpl(ctx, i, k):
    row = ctx._ROW(i)
    col = ctx._COL(i)
    dr, dc = ctx._DELTA(k)
    return ctx._ROWEDGEOK(row + dr) * ctx._COLEDGEOK(col + dc) * ctx._SOURCEOK(i)


# ---------------------------------------------------------------------------
# _valid(j): in range, and (nodata ? active : true)
# ---------------------------------------------------------------------------


def _valid_no_nodata_tmpl(ctx, j):
    ok = 0
    if j >= 0 and j < ctx.NX.get(0) * ctx.NY.get(0):
        ok = 1
    return ok


def _valid_nodata_tmpl(ctx, j):
    ok = 0
    if j >= 0 and j < ctx.NX.get(0) * ctx.NY.get(0):
        ok = 1
        if ctx.NODATA_MASK.get(j) == 1:
            ok = 0
    return ok


# ---------------------------------------------------------------------------
# public: neighbour / neighbour_raw
# ---------------------------------------------------------------------------


def _neighbour_raw_tmpl(ctx, i, k):
    row = ctx._ROW(i)
    col = ctx._COL(i)
    dr, dc = ctx._DELTA(k)
    wrow, wcol = ctx._WRAP(row + dr, col + dc)
    return ctx._INDEX(wrow, wcol)


def _neighbour_tmpl(ctx, i, k):
    j = -1
    if ctx._MOVEALLOWED(i, k) == 1:
        cand = ctx._NEIGHBOURRAW(i, k)
        if ctx._VALID(cand) == 1:
            j = cand
    return j


# ---------------------------------------------------------------------------
# public: is_active / nodata
# ---------------------------------------------------------------------------


def _is_active_always_tmpl(ctx, i):
    return 1


def _is_active_mask_tmpl(ctx, i):
    ok = 1
    if ctx.NODATA_MASK.get(i) == 1:
        ok = 0
    return ok


def _nodata_tmpl(ctx, i):
    return 1 - ctx._ISACTIVE(i)


# ---------------------------------------------------------------------------
# public: is_on_edge / which_edge (per-axis, mirrors _wrap/_edge_ok)
# ---------------------------------------------------------------------------


def _row_is_edge_active_tmpl(ctx, row):
    e = 0
    if row == 0 or row == ctx.NY.get(0) - 1:
        e = 1
    return e


def _row_is_edge_periodic_tmpl(ctx, row):
    return 0


def _col_is_edge_active_tmpl(ctx, col):
    e = 0
    if col == 0 or col == ctx.NX.get(0) - 1:
        e = 1
    return e


def _col_is_edge_periodic_tmpl(ctx, col):
    return 0


def _is_on_edge_tmpl(ctx, i):
    row = ctx._ROW(i)
    col = ctx._COL(i)
    e = 0
    if ctx._ROWISEDGE(row) == 1 or ctx._COLISEDGE(col) == 1:
        e = 1
    return e


def _row_edge_code_active_tmpl(ctx, row):
    code = -1
    if row == 0:
        code = 0
    elif row == ctx.NY.get(0) - 1:
        code = 3
    return code


def _row_edge_code_periodic_tmpl(ctx, row):
    return -1


def _col_edge_code_active_tmpl(ctx, col):
    code = -1
    if col == 0:
        code = 1
    elif col == ctx.NX.get(0) - 1:
        code = 2
    return code


def _col_edge_code_periodic_tmpl(ctx, col):
    return -1


def _which_edge_tmpl(ctx, i):
    row = ctx._ROW(i)
    col = ctx._COL(i)
    code = ctx._ROWEDGECODE(row)
    if code == -1:
        code = ctx._COLEDGECODE(col)
    return code


# ---------------------------------------------------------------------------
# public: can_out
# ---------------------------------------------------------------------------


def _can_out_mask_tmpl(ctx, i):
    out = 0
    if ctx.OUTLET_MASK.get(i) == 1:
        out = 1
    return out


def _can_out_edge_tmpl(ctx, i):
    return ctx._ISONEDGE(i)


# ---------------------------------------------------------------------------
# public: n / dx / area / reverse / dist / width / flow_width / dist_between_nodes
# ---------------------------------------------------------------------------


def _n_tmpl(ctx):
    return ctx.NX.get(0) * ctx.NY.get(0)


def _dx_tmpl(ctx, i):
    return ctx.DX.get(i)


def _area_tmpl(ctx, i):
    d = ctx.DX.get(i)
    return d * d


def _reverse_tmpl(ctx, k):
    return ctx.N_NEIGHBOURS.get(0) - 1 - k


def _dist_d4_tmpl(ctx, i, k):
    return ctx.DX.get(i)


_SQRT2 = 1.4142135623730951


def _dist_d8_tmpl(ctx, i, k):
    d = ctx.DX.get(i)
    if k == 0 or k == 2 or k == 5 or k == 7:
        d = d * _SQRT2
    return d


def _width_tmpl(ctx, i, k):
    return ctx._DIST(i, k)


def _row_dist_normal_tmpl(ctx, raw):
    return abs(raw)


def _row_dist_periodic_tmpl(ctx, raw):
    d = abs(raw)
    return min(d, ctx.NY.get(0) - d)


def _col_dist_normal_tmpl(ctx, raw):
    return abs(raw)


def _col_dist_periodic_tmpl(ctx, raw):
    d = abs(raw)
    return min(d, ctx.NX.get(0) - d)


def _dist_between_d4_tmpl(ctx, i, j):
    out = -1.0
    if j >= 0:
        dr = ctx._ROWDIST(ctx._ROW(j) - ctx._ROW(i))
        dc = ctx._COLDIST(ctx._COL(j) - ctx._COL(i))
        if dr == 0 and dc == 1:
            out = ctx.DX.get(i)
        elif dr == 1 and dc == 0:
            out = ctx.DX.get(i)
    return out


def _dist_between_d8_tmpl(ctx, i, j):
    out = -1.0
    if j >= 0:
        dr = ctx._ROWDIST(ctx._ROW(j) - ctx._ROW(i))
        dc = ctx._COLDIST(ctx._COL(j) - ctx._COL(i))
        if dr == 0 and dc == 1:
            out = ctx.DX.get(i)
        elif dr == 1 and dc == 0:
            out = ctx.DX.get(i)
        elif dr == 1 and dc == 1:
            out = ctx.DX.get(i) * _SQRT2
    return out


# ---------------------------------------------------------------------------
# public: neighbour_and_distance
# ---------------------------------------------------------------------------


def _neighbour_and_distance_tmpl(ctx, i, k):
    j = ctx._NEIGHBOUR(i, k)
    d = -1.0
    if j != -1:
        d = ctx._DIST(i, k)
    return j, d


def build_group(group, *, topology, boundary, nodata, outlet):
    """Compose the selected closure-backend grid helpers onto ``group``."""
    periodic_x = boundary in ("periodic_EW", "periodic_EW_NS")
    periodic_y = boundary in ("periodic_NS", "periodic_EW_NS")
    d8 = topology == "D8"

    row = _helper(_row_tmpl, params=["NX"])
    col = _helper(_col_tmpl, params=["NX"])
    index = _helper(_index_tmpl, params=["NX"])

    delta = _helper(_delta_d8_tmpl if d8 else _delta_d4_tmpl)

    row_wrap = _helper(
        _row_wrap_periodic_tmpl if periodic_y else _row_wrap_identity_tmpl,
        params=["NY"] if periodic_y else [],
    )
    col_wrap = _helper(
        _col_wrap_periodic_tmpl if periodic_x else _col_wrap_identity_tmpl,
        params=["NX"] if periodic_x else [],
    )
    wrap = _helper(_wrap_tmpl, helpers={"_ROWWRAP": row_wrap, "_COLWRAP": col_wrap})

    row_edge_ok = _helper(
        _row_edge_ok_periodic_tmpl if periodic_y else _row_edge_ok_bounded_tmpl,
        params=[] if periodic_y else ["NY"],
    )
    col_edge_ok = _helper(
        _col_edge_ok_periodic_tmpl if periodic_x else _col_edge_ok_bounded_tmpl,
        params=[] if periodic_x else ["NX"],
    )

    source_ok = (
        _helper(_source_ok_nodata_tmpl, params=["NODATA_MASK"]) if nodata else _helper(_source_ok_always_tmpl)
    )

    move_allowed = _helper(
        _move_allowed_tmpl,
        helpers={
            "_ROW": row,
            "_COL": col,
            "_DELTA": delta,
            "_ROWEDGEOK": row_edge_ok,
            "_COLEDGEOK": col_edge_ok,
            "_SOURCEOK": source_ok,
        },
    )

    valid_params = ["NX", "NY"] + (["NODATA_MASK"] if nodata else [])
    valid = _helper(_valid_nodata_tmpl if nodata else _valid_no_nodata_tmpl, params=valid_params)

    neighbour_raw = _helper(
        _neighbour_raw_tmpl,
        helpers={"_ROW": row, "_COL": col, "_DELTA": delta, "_WRAP": wrap, "_INDEX": index},
    )
    neighbour = _helper(
        _neighbour_tmpl,
        helpers={"_MOVEALLOWED": move_allowed, "_NEIGHBOURRAW": neighbour_raw, "_VALID": valid},
    )

    is_active = _helper(_is_active_mask_tmpl, params=["NODATA_MASK"]) if nodata else _helper(_is_active_always_tmpl)
    nodata_fn = _helper(_nodata_tmpl, helpers={"_ISACTIVE": is_active})

    row_is_edge = _helper(
        _row_is_edge_periodic_tmpl if periodic_y else _row_is_edge_active_tmpl,
        params=[] if periodic_y else ["NY"],
    )
    col_is_edge = _helper(
        _col_is_edge_periodic_tmpl if periodic_x else _col_is_edge_active_tmpl,
        params=[] if periodic_x else ["NX"],
    )
    is_on_edge = _helper(
        _is_on_edge_tmpl,
        helpers={"_ROW": row, "_COL": col, "_ROWISEDGE": row_is_edge, "_COLISEDGE": col_is_edge},
    )

    row_edge_code = _helper(
        _row_edge_code_periodic_tmpl if periodic_y else _row_edge_code_active_tmpl,
        params=[] if periodic_y else ["NY"],
    )
    col_edge_code = _helper(
        _col_edge_code_periodic_tmpl if periodic_x else _col_edge_code_active_tmpl,
        params=[] if periodic_x else ["NX"],
    )
    which_edge = _helper(
        _which_edge_tmpl,
        helpers={"_ROW": row, "_COL": col, "_ROWEDGECODE": row_edge_code, "_COLEDGECODE": col_edge_code},
    )

    if outlet == "mask":
        can_out = _helper(_can_out_mask_tmpl, params=["OUTLET_MASK"])
    else:
        can_out = _helper(_can_out_edge_tmpl, helpers={"_ISONEDGE": is_on_edge})

    n_cells = _helper(_n_tmpl, params=["NX", "NY"])
    dx = _helper(_dx_tmpl, params=["DX"])
    area = _helper(_area_tmpl, params=["DX"])
    reverse = _helper(_reverse_tmpl, params=["N_NEIGHBOURS"])
    dist = _helper(_dist_d8_tmpl if d8 else _dist_d4_tmpl, params=["DX"])
    width = _helper(_width_tmpl, helpers={"_DIST": dist})
    flow_width = _helper(_width_tmpl, helpers={"_DIST": dist})

    row_dist = _helper(
        _row_dist_periodic_tmpl if periodic_y else _row_dist_normal_tmpl,
        params=["NY"] if periodic_y else [],
    )
    col_dist = _helper(
        _col_dist_periodic_tmpl if periodic_x else _col_dist_normal_tmpl,
        params=["NX"] if periodic_x else [],
    )
    dist_between = _helper(
        _dist_between_d8_tmpl if d8 else _dist_between_d4_tmpl,
        params=["DX"],
        helpers={"_ROW": row, "_COL": col, "_ROWDIST": row_dist, "_COLDIST": col_dist},
    )

    neighbour_and_distance = _helper(
        _neighbour_and_distance_tmpl,
        helpers={"_NEIGHBOUR": neighbour, "_DIST": dist},
    )

    group.compose("neighbour", neighbour)
    group.compose("neighbour_raw", neighbour_raw)
    group.compose("nodata", nodata_fn)
    group.compose("is_active", is_active)
    group.compose("can_out", can_out)
    group.compose("n", n_cells)
    group.compose("dx", dx)
    group.compose("area", area)
    group.compose("dist", dist)
    group.compose("width", width)
    group.compose("flow_width", flow_width)
    group.compose("reverse", reverse)
    group.compose("dist_between_nodes", dist_between)
    group.compose("is_on_edge", is_on_edge)
    group.compose("which_edge", which_edge)
    group.compose("neighbour_and_distance", neighbour_and_distance)
