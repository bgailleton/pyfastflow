"""CUDA templates for depression handling."""

from ..core import HelperBuilder, KernelBuilder, RoutineBuilder, new_uid


def build_atomic_min_ll():
    """
    atomicMin over a signed 64-bit cell via a CAS loop - CUDA has no native
    atomicMin for signed long long (only int and unsigned long long), and
    the bitpacked saddle/outlet values need signed comparison to match
    Taichi/Quadrants' `atomic_min` over an i64 field.

    Returns
    -------
    HelperBuilder

    """
    t = f"pd{new_uid()}"
    return HelperBuilder(
        f"""
__device__ long long {t}_atomic_min_ll(long long* addr, long long val) {{
    long long old = *addr, assumed;
    do {{
        assumed = old;
        if (assumed <= val) break;
        old = (long long)atomicCAS((unsigned long long*)addr, (unsigned long long)assumed, (unsigned long long)val);
    }} while (assumed != old);
    return old;
}}
"""
    ).freeze()


def build_copy_field(*, n_flat: int):
    """
    dst[i] = src[i] over a whole n_flat int32 buffer - see
    _closure_depressions.py's build_copy_field.

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    KernelBuilder

    """
    t = f"pd{new_uid()}"
    return (
        KernelBuilder(
            f"""
__global__ void {t}_copy_field(const int* src, int* dst) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    dst[i] = src[i];
}}
""", domain=n_flat).freeze()
    )


def build_basin_id_init(*, grid, n_flat: int):
    """
    bid[i] = 0 on a can_out node, i+1 otherwise. Data arg (bid,). Composes
    its own `grid` occurrence.

    Parameters
    ----------
    grid : FrozenGroup
    n_flat : int

    Returns
    -------
    KernelBuilder

    """
    t = f"pbi{new_uid()}"
    return (
        KernelBuilder(
            f"""
__global__ void {t}_basin_id_init(int* bid) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    bid[i] = $ctx.grid.can_out(i)$ ? 0 : (i + 1);
}}
""", domain=n_flat)
        .compose("grid", grid)
        .freeze()
    )


def build_propagate_basin_iter(*, n_flat: int):
    """One pointer-jump step over `rec_jump`. Data arg (rec_jump,).

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    KernelBuilder
    """
    t = f"pbi{new_uid()}"
    return (
        KernelBuilder(
            f"""
__global__ void {t}_propagate_basin_iter(int* rec_jump) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (rec_jump[i] != rec_jump[rec_jump[i]]) {{
        rec_jump[i] = rec_jump[rec_jump[i]];
    }}
}}
""", domain=n_flat).freeze()
    )


def build_propagate_basin_final(*, n_flat: int):
    """bid[i] = bid[root(i)]. Data args (bid, rec_jump).

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    KernelBuilder
    """
    t = f"pbf{new_uid()}"
    return (
        KernelBuilder(
            f"""
__global__ void {t}_propagate_basin_final(int* bid, const int* rec_jump) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    bid[i] = bid[rec_jump[i]];
}}
""", domain=n_flat).freeze()
    )


def build_basin_labelling_vanilla(*, grid, copy_field, n_flat: int, logn: int):
    """
    RoutineBuilder (routine) for vanilla basin labelling - see
    _closure_depressions.py's own (identical step sequence and unroll
    choice). Every step here is already one launch (no cross-loop splitting
    needed for this variant - it has no per-round cross-thread ordering
    dependency other kernels here need split for).

    Parameters
    ----------
    grid : FrozenGroup
    copy_field : KernelBuilder
    n_flat : int
    logn : int

    Returns
    -------
    tuple[RoutineBuilder, dict]

    """
    basin_id_init = build_basin_id_init(grid=grid, n_flat=n_flat)
    propagate_basin_iter = build_propagate_basin_iter(n_flat=n_flat)
    propagate_basin_final = build_propagate_basin_final(n_flat=n_flat)

    kernels = {
        "basin_id_init": basin_id_init,
        "propagate_basin_iter": propagate_basin_iter,
        "propagate_basin_final": propagate_basin_final,
    }

    rb = RoutineBuilder()
    rb.step("basin_id_init", basin_id_init)
    rb.step("copy_rec_to_recjump", copy_field)
    for k in range(logn + 1):
        rb.step(f"propagate_iter_{k}", propagate_basin_iter)
    rb.step("propagate_basin_final", propagate_basin_final)

    return rb, kernels


def build_label_from_route(*, grid, n_flat: int):
    """
    bid[i] = 0 if the node basin_route reaches can output, else root + 1 - the
    carried-route basin labelling. Data args (bid, basin_route); composes
    `grid` for can_out. See _closure_depressions.py's build_label_from_route.

    """
    t = f"lfr{new_uid()}"
    return (
        KernelBuilder(
            f"""
__global__ void {t}_label_from_route(int* bid, const int* basin_route) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int root = basin_route[i];
    bid[i] = $ctx.grid.can_out(root)$ ? 0 : (root + 1);
}}
""", domain=n_flat)
        .compose("grid", grid)
        .freeze()
    )


def build_basin_labelling_route(*, grid, n_flat: int, logn: int):
    """
    RoutineBuilder for basin labelling from the carried `basin_route`: logn+1
    unrolled contractions of basin_route, then label_from_route. See
    _closure_depressions.py's build_basin_labelling_route.

    Composed step names: "contract_0".."contract_{logn}", "label_from_route".
    Data addresses: "contract_K.rec_jump" (all bound to basin_route),
    "label_from_route.bid"/".basin_route".

    """
    propagate_basin_iter = build_propagate_basin_iter(n_flat=n_flat)
    label_from_route = build_label_from_route(grid=grid, n_flat=n_flat)

    kernels = {"propagate_basin_iter": propagate_basin_iter, "label_from_route": label_from_route}

    rb = RoutineBuilder()
    for k in range(logn + 1):
        rb.step(f"contract_{k}", propagate_basin_iter)
    rb.step("label_from_route", label_from_route)

    return rb, kernels


def build_merge_basin_route(*, bitpack, n_flat: int):
    """
    basin_route[pit] = outlet node, folding every kept basin into its receiver
    (basin id = pit + 1). Data args (outlet, basin_route); composes `bitpack`
    for unpack_index. See _closure_depressions.py's build_merge_basin_route.

    """
    t = f"mbr{new_uid()}"
    return (
        KernelBuilder(
            f"""
__global__ void {t}_merge_basin_route(const long long* outlet, int* basin_route) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    if (i == 0 || outlet[i] == invalid) return;
    int p_rcv = $ctx.bitpack.unpack_index(outlet[i])$;
    basin_route[i - 1] = p_rcv;
}}
""", domain=n_flat)
        .compose("bitpack", bitpack)
        .freeze()
    )


def build_basin_labelling_optimized(*, grid, n_flat: int):
    """
    RoutineBuilder (routine) for optimized basin labelling - the closure
    backends' single label_basins_walk launch split into three real
    launches (copy, path-halving, bid finalize), since the path-halving
    phase needs every thread's copy to have landed first, and the finalize
    phase needs every thread's path-halving to have converged first.

    Parameters
    ----------
    grid : FrozenGroup
    n_flat : int

    Returns
    -------
    tuple[RoutineBuilder, dict]

    """
    t = f"pbo{new_uid()}"

    walk_copy = (
        KernelBuilder(
            f"""
__global__ void {t}_walk_copy(const int* rec, int* rec_jump) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    rec_jump[i] = rec[i];
}}
""", domain=n_flat).freeze()
    )
    walk_halving = (
        KernelBuilder(
            f"""
__global__ void {t}_walk_halving(int* rec_jump) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int guard = 0;
    while (rec_jump[i] != rec_jump[rec_jump[i]] && guard < {n_flat}) {{
        rec_jump[i] = rec_jump[rec_jump[i]];
        guard++;
    }}
}}
""", domain=n_flat).freeze()
    )
    walk_finalize = (
        KernelBuilder(
            f"""
__global__ void {t}_walk_finalize(const int* rec_jump, int* bid) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int root = rec_jump[i];
    bid[i] = $ctx.grid.can_out(root)$ ? 0 : root + 1;
}}
""", domain=n_flat)
        .compose("grid", grid)
        .freeze()
    )

    kernels = {"walk_copy": walk_copy, "walk_halving": walk_halving, "walk_finalize": walk_finalize}

    rb = RoutineBuilder()
    rb.step("walk_copy", walk_copy)
    rb.step("walk_halving", walk_halving)
    rb.step("walk_finalize", walk_finalize)

    return rb, kernels


def build_saddlesort(*, grid, bitpack, n_flat: int):
    """
    RoutineBuilder (routine) for the six saddlesort passes - see
    _closure_depressions.py's build_saddlesort for the step sequence.
    `bitpack` is the FrozenGroup ops.make_bitpack_group returns
    (`$ctx.bitpack.pack(...)$`/`.unpack_value`/`.unpack_index`); each
    KernelBuilder composes its own occurrence, only where it actually calls
    one of the three. `atomic_min_ll` (build_atomic_min_ll) is composed onto
    the two sites that need it.

    Parameters
    ----------
    grid : FrozenGroup
    bitpack : FrozenGroup
    n_flat : int

    Returns
    -------
    tuple[RoutineBuilder, dict]

    """
    atomic_min_ll = build_atomic_min_ll()
    t = f"pss{new_uid()}"

    border_zprime = (
        KernelBuilder(
            f"""
__global__ void {t}_border_zprime(const int* bid, const float* z, float* z_prime, unsigned char* is_border) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if ($ctx.grid.can_out(i)$) {{
        z_prime[i] = z[i];
        return;
    }}
    is_border[i] = 0;
    z_prime[i] = 1e9f;
    float zn = 1e9f;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(i, k)$;
        if (j != -1 && bid[j] != bid[i]) {{
            is_border[i] = 1;
            zn = fminf(zn, z[j]);
        }}
    }}
    if (is_border[i]) {{
        z_prime[i] = fmaxf(z[i], zn);
    }}
}}
""", domain=n_flat)
        .compose("grid", grid)
        .freeze()
    )
    init_saddle_outlet = (
        KernelBuilder(
            f"""
__global__ void {t}_init_saddle_outlet(long long* basin_saddle, long long* outlet, int* basin_saddlenode, int* b_rcv) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    basin_saddle[i] = invalid;
    outlet[i] = invalid;
    basin_saddlenode[i] = -1;
    b_rcv[i] = 0;
}}
""", domain=n_flat)
        .compose("bitpack", bitpack)
        .freeze()
    )
    atomic_min_saddle = (
        KernelBuilder(
            f"""
__global__ void {t}_atomic_min_saddle(const int* bid, const unsigned char* is_border, const float* z_prime, long long* basin_saddle) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (!is_border[i]) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    int tbid = bid[i];
    long long res = invalid;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(i, k)$;
        if (j != -1 && bid[j] != tbid) {{
            long long candidate = $ctx.bitpack.pack(z_prime[i], bid[j])$;
            res = (candidate < res) ? candidate : res;
        }}
    }}
    if (res != invalid) {{
        $ctx.atomic_min_ll(&basin_saddle[tbid], res)$;
    }}
}}
""", domain=n_flat)
        .compose("grid", grid).compose("bitpack", bitpack).compose("atomic_min_ll", atomic_min_ll)
        .freeze()
    )
    find_saddlenode = (
        KernelBuilder(
            f"""
__global__ void {t}_find_saddlenode(const int* bid, const unsigned char* is_border, const float* z_prime,
                                     const long long* basin_saddle, int* basin_saddlenode) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (!is_border[i] || bid[i] == 0) return;
    long long packed = basin_saddle[bid[i]];
    float target_z = $ctx.bitpack.unpack_value(packed)$;
    int target_b = $ctx.bitpack.unpack_index(packed)$;
    int is_here = 0;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(i, k)$;
        if (j != -1 && bid[j] == target_b && z_prime[i] == target_z) {{
            is_here = 1;
        }}
    }}
    if (is_here) {{
        basin_saddlenode[bid[i]] = i;
    }}
}}
""", domain=n_flat)
        .compose("grid", grid).compose("bitpack", bitpack)
        .freeze()
    )
    atomic_min_outlet = (
        KernelBuilder(
            f"""
__global__ void {t}_atomic_min_outlet(const int* bid, const long long* basin_saddle, const int* basin_saddlenode,
                                       const float* z, long long* outlet, int* b_rcv) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    if (i == 0 || basin_saddle[i] == invalid) return;
    int node = basin_saddlenode[i];
    float best_z = 1e9f;
    int best_b = 2147483647;
    int rec_out = -1;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(node, k)$;
        if (j != -1 && bid[j] != i) {{
            float cz = fmaxf(z[node], z[j]);
            int bj = bid[j];
            if (cz < best_z || (cz == best_z && bj < best_b)) {{
                best_z = cz;
                best_b = bj;
                rec_out = j;
            }}
        }}
    }}
    if (rec_out > -1) {{
        outlet[i] = $ctx.bitpack.pack(best_z, rec_out)$;
        b_rcv[i] = bid[rec_out];
    }}
}}
""", domain=n_flat)
        .compose("grid", grid).compose("bitpack", bitpack)
        .freeze()
    )
    set_keep = (
        KernelBuilder(
            f"""
__global__ void {t}_set_keep(const int* bid, const int* b_rcv, long long* outlet, long long* basin_saddle, int* basin_saddlenode) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    if (i == 0 || outlet[i] == invalid) return;
    int brd = b_rcv[i];
    if (brd != 0 && b_rcv[brd] == i && brd > i) {{
        outlet[i] = invalid;
        basin_saddle[i] = invalid;
        basin_saddlenode[i] = -1;
    }}
}}
""", domain=n_flat)
        .compose("bitpack", bitpack)
        .freeze()
    )

    kernels = {
        "border_zprime": border_zprime,
        "init_saddle_outlet": init_saddle_outlet,
        "atomic_min_saddle": atomic_min_saddle,
        "find_saddlenode": find_saddlenode,
        "atomic_min_outlet": atomic_min_outlet,
        "break_cycle": set_keep,
    }

    rb = RoutineBuilder()
    rb.step("border_zprime", border_zprime)
    rb.step("init_saddle_outlet", init_saddle_outlet)
    rb.step("atomic_min_saddle", atomic_min_saddle)
    rb.step("find_saddlenode", find_saddlenode)
    rb.step("atomic_min_outlet", atomic_min_outlet)
    rb.step("break_cycle", set_keep)

    return rb, kernels


def build_reroute_carve_vanilla(*, bitpack, copy_field, n_flat: int, logn: int):
    """
    RoutineBuilder (routine) for carve+vanilla reroute - see
    _closure_depressions.py's build_reroute_carve_vanilla for the buffer
    roles (`rec_jump` here is finalise's original, unjumped snapshot, not
    the pointer-jumped result - same note applies). init_reroute_carve,
    iteration_reroute_carve and finalise_reroute_carve are each further
    split into several real launches here (no grid-wide barrier inside one
    `__global__`); the closure backends keep each as one kernel.

    Parameters
    ----------
    bitpack : FrozenGroup
    copy_field : KernelBuilder
    n_flat : int
    logn : int

    Returns
    -------
    tuple[RoutineBuilder, dict]

    """
    t = f"prc{new_uid()}"

    init_reset_tag = (
        KernelBuilder(
            f"""
__global__ void {t}_init_reset_tag(unsigned char* tag) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    tag[i] = 0;
}}
""", domain=n_flat).freeze()
    )
    init_scatter_tag = (
        KernelBuilder(
            f"""
__global__ void {t}_init_scatter_tag(unsigned char* tag, const int* saddlenode) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (saddlenode[i] != -1) {{
        tag[saddlenode[i]] = 1;
    }}
}}
""", domain=n_flat).freeze()
    )
    init_copy_tag_alt = (
        KernelBuilder(
            f"""
__global__ void {t}_init_copy_tag_alt(const unsigned char* tag, unsigned char* tag_alt) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    tag_alt[i] = tag[i];
}}
""", domain=n_flat).freeze()
    )
    iter_build_work = (
        KernelBuilder(
            f"""
__global__ void {t}_iter_build_work(const unsigned char* tag, unsigned char* tag_alt, const int* rec,
                                     int* rec_work, const int* bid) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (bid[i] == 0) return;
    if (tag[i] && rec[i] != i) {{
        tag_alt[rec[i]] = 1;
    }}
    rec_work[i] = rec[i];
}}
""", domain=n_flat).freeze()
    )
    iter_jump = (
        KernelBuilder(
            f"""
__global__ void {t}_iter_jump(unsigned char* tag, const unsigned char* tag_alt, int* rec,
                               const int* rec_work, const int* bid) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (bid[i] == 0) return;
    if (rec_work[i] != i) {{
        rec[i] = rec_work[rec_work[i]];
    }}
    tag[i] = tag_alt[i];
}}
""", domain=n_flat).freeze()
    )
    finalise_reset_rec = (
        KernelBuilder(
            f"""
__global__ void {t}_finalise_reset_rec(int* rec, const int* rec_orig) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    rec[i] = rec_orig[i];
}}
""", domain=n_flat).freeze()
    )
    finalise_reverse = (
        KernelBuilder(
            f"""
__global__ void {t}_finalise_reverse(int* rec, const int* rec_orig, const unsigned char* tag, unsigned char* rerouted) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int ro = rec_orig[i];
    if (tag[ro] && tag[i] && i != ro) {{
        rec[ro] = i;
        rerouted[ro] = 1;
    }}
}}
""", domain=n_flat).freeze()
    )
    finalise_outlet = (
        KernelBuilder(
            f"""
__global__ void {t}_finalise_outlet(int* rec, const long long* outlet, const int* saddlenode, unsigned char* rerouted) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    if (outlet[i] != invalid) {{
        int node = $ctx.bitpack.unpack_index(outlet[i])$;
        rec[saddlenode[i]] = node;
        rerouted[saddlenode[i]] = 1;
    }}
}}
""", domain=n_flat)
        .compose("bitpack", bitpack)
        .freeze()
    )

    kernels = {
        "init_reset_tag": init_reset_tag,
        "init_scatter_tag": init_scatter_tag,
        "init_copy_tag_alt": init_copy_tag_alt,
        "iter_build_work": iter_build_work,
        "iter_jump": iter_jump,
        "finalise_reset_rec": finalise_reset_rec,
        "finalise_reverse": finalise_reverse,
        "finalise_outlet": finalise_outlet,
    }

    rb = RoutineBuilder()
    rb.step("init_reset_tag", init_reset_tag)
    rb.step("init_scatter_tag", init_scatter_tag)
    rb.step("init_copy_tag_alt", init_copy_tag_alt)
    rb.step("copy_recwork_to_rec", copy_field)
    rb.step("copy_recwork_to_recjump", copy_field)
    for k in range(logn + 1):
        rb.step(f"iter_build_work_{k}", iter_build_work)
        rb.step(f"iter_jump_{k}", iter_jump)
    rb.step("finalise_reset_rec", finalise_reset_rec)
    rb.step("finalise_reverse", finalise_reverse)
    rb.step("finalise_outlet", finalise_outlet)
    rb.step("copy_rec_to_recwork", copy_field)

    return rb, kernels


def build_reroute_carve_optimized(*, bitpack, n_flat: int):
    """
    carve_basins_serial KernelBuilder - one launch, one serial thread per
    basin; see _closure_depressions.py's build_reroute_carve_optimized.
    Node-disjoint chains across basins mean no cross-thread dependency at
    all, so this needs no splitting the way the vanilla carve routine does.

    Parameters
    ----------
    bitpack : FrozenGroup
    n_flat : int

    Returns
    -------
    KernelBuilder

    """
    t = f"pco{new_uid()}"
    return (
        KernelBuilder(
            f"""
__global__ void {t}_carve_basins_serial(int* rec, const int* basin_saddlenode, const long long* outlet) {{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    int s = basin_saddlenode[b];
    if (s == -1 || outlet[b] == invalid) return;
    int out_node = $ctx.bitpack.unpack_index(outlet[b])$;
    int node = s;
    int nxt = rec[node];
    rec[node] = out_node;
    int guard = 0;
    while (nxt != node && guard < {n_flat}) {{
        int nnxt = rec[nxt];
        rec[nxt] = node;
        node = nxt;
        nxt = nnxt;
        guard++;
    }}
}}
""", domain=n_flat)
        .compose("bitpack", bitpack)
        .freeze()
    )


def build_reroute_jump(*, bitpack, n_flat: int):
    """
    RoutineBuilder (routine) for reroute_jump - split into a reset launch
    and the jump launch itself, since the jump phase writes
    `rerouted[i - 1]` from thread `i`, a cell a *different* thread's reset
    zeroed. The closure backends keep this as one two-loop kernel.

    Basin IDs are one-based, so the write targets ``rec[i - 1]`` rather than
    ``rec[i]``.

    Parameters
    ----------
    bitpack : FrozenGroup
    n_flat : int

    Returns
    -------
    tuple[RoutineBuilder, dict]

    """
    t = f"prj{new_uid()}"

    reset_rerouted = (
        KernelBuilder(
            f"""
__global__ void {t}_reset_rerouted(unsigned char* rerouted) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    rerouted[i] = 0;
}}
""", domain=n_flat).freeze()
    )
    jump = (
        KernelBuilder(
            f"""
__global__ void {t}_jump(int* rec, const long long* outlet, unsigned char* rerouted) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    if (outlet[i] != invalid) {{
        int rrec = $ctx.bitpack.unpack_index(outlet[i])$;
        rec[i - 1] = rrec;
        rerouted[i - 1] = 1;
    }}
}}
""", domain=n_flat)
        .compose("bitpack", bitpack)
        .freeze()
    )

    kernels = {"reset_rerouted": reset_rerouted, "jump": jump}

    rb = RoutineBuilder()
    rb.step("reset_rerouted", reset_rerouted)
    rb.step("jump", jump)

    return rb, kernels


def build_depression_counter(*, grid, n_flat: int):
    """
    depression_counter KernelBuilder, data args (rec, ndep) - `ndep` is
    `ndep_p.handle().array`, passed positionally same as `rec` (a Parameter
    reached only through `$...$` get() spans is registered read-only in the
    constant block, so atomicAdd into it needs the raw pointer as an
    ordinary DATA argument instead). The caller must reset `ndep_p` to 0
    (`.set(0)`) before each launch. Composes its own `grid` occurrence.

    Parameters
    ----------
    grid : FrozenGroup
    n_flat : int

    Returns
    -------
    KernelBuilder

    """
    t = f"pdc{new_uid()}"
    return (
        KernelBuilder(
            f"""
__global__ void {t}_depression_counter(const int* rec, int* ndep) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (rec[i] == i && !($ctx.grid.can_out(i)$) && !($ctx.grid.nodata(i)$)) {{
        atomicAdd(ndep, 1);
    }}
}}
""", domain=n_flat)
        .compose("grid", grid)
        .freeze()
    )
