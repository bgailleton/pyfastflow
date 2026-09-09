"""
cupy-only MFD topology construction for make_graphflood's kind="vanilla_mfd" -
computes the per-node downslope receiver bitmask (`dirs`), normalized
slope-proportional weights (`mfd_w`) and indegree ../flow's persistent_mfd
accumulation (_cupy_mfd_accum.py) needs to run over an already-built MFD
graph, from a filled elevation surface. Unported anywhere else in the
package before this - ../../CLAUDE.md's own state notes flag this as the
one piece "still entirely unported": legacy's only MFD code
(pyfastflow/flow/flow_mfd_kernels.py) is a completely different Jacobi
power-iteration scheme with dense per-node routing_weights and no bitmask/
indegree at all, so this is new work, not a port.

cupy-only, no closure-backend equivalent, for the same structural reason
_cupy_mfd_accum.py itself is cupy-only: persistent_mfd's grid-wide barrier
needs raw CUDA primitives no closure-backend kernel model expresses - the
topology this module builds only exists to feed that accumulation kernel,
so there is no reason for a closure-backend variant to exist independently.

`filled` is the caller-supplied elevation surface this operates on -
make_graphflood's kind="vanilla_mfd" always passes the fill_reconstruct
surface (the resolved, monotonic z+h fill), never bare z or z+h directly:
MFD's per-node weight split needs every node to have somewhere to send
water, which an unresolved depression does not guarantee. `dist` is the
companion per-cell perturbation ../graphflood/_cupy_reconstruct_epsilon.py
accumulates along the `parent` chain - a strictly-toward-the-outlet, ULP-
scaled tie-break carrier. The slope helper's own second additive term (its
`h` argument) is exactly where it belongs:
`slope(filled[i], dist[i], filled[j], dist[j], k)` computes
`((filled[i] - filled[j]) + (dist[i] - dist[j])) / d` - the real relief
drop plus the perturbation drop, in one call. Inside a resolved depression
the `filled` drop is exactly 0 for every neighbour pair (a real lake
surface IS flat), and the `dist` drop - evaluated at its own ~1e-7
magnitude, never folded up into `filled`'s magnitude where it would round
away - is what gives every flat cell a downslope edge toward the outlet.
On genuine relief the `dist` drop is negligible against the real drop, so
it perturbs neither the direction set nor the weights there.

A can_out node gets `dirs[i] = 0` (no outgoing directions at all, mask
never set) and an all-zero `mfd_w` row - the same "this is where routing
stops" role `rec[i] = i` plays for SFD receivers, just expressed as "sends
nowhere" instead of "sends to itself" (persistent_mfd's own frontier/
indegree walk has no self-loop concept to exploit the SFD convention with).
A nodata node gets the identical `dirs[i] = 0` / all-zero `mfd_w` treatment
(via `ctx.grid.nodata(i)` in the same guard): its `filled` is a raster-sweep
sentinel far above the live terrain, so without the guard it would compute a
large positive slope to every live neighbour and route the spurious source
_cupy_mfd_accum.py's q_init seeds it with (which q_init itself gates to 0)
into the live ring. Live cells never route _into_ a nodata cell -
`ctx.grid.neighbour(i, k)` already returns -1 for a nodata target.

Author: B.G (08/2026)
"""

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

    Author: B.G (08/2026)
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
