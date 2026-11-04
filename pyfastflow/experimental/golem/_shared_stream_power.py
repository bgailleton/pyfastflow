"""Linear shared stream-power coefficients and GOLEM layer bookkeeping."""

from pyfastflow.core import KernelBuilder, SequenceBuilder, new_uid
from pyfastflow.flow import make_sfd_linear_decline
from pyfastflow.flow._program_cupy import _cupy_only, _grid_leaf_plan


def shared_stream_power_factory(be, bundles, config):
    """Build one implicit, mass-conserving ``n=1`` shared-power step.

    ``detachment_erodibility`` is Kd and ``transport_erodibility`` is Kt.
    The solver finds net surface change before this program assigns erosion
    to the sediment layer and then bedrock. Both use the same Kd for now.
    """
    _cupy_only(be)
    if config["local_minima"] == "cordonnier_jump":
        raise ValueError("shared_stream_power requires local SFD receiver links")
    n = config["nx"] * config["ny"]
    nk = 8 if config["topology"] == "D8" else 4
    tag = f"golem_shared_{new_uid()}"

    coefficients = KernelBuilder(f'''
extern "C" __global__ void {tag}_coefficients(
        const float* area, float* phi, float* psi) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    float a = fmaxf(area[i], 0.0f);
    float kd = fmaxf($ctx.KD.get(i)$, 0.0f);
    float kt = fmaxf($ctx.KT.get(i)$, 1.0e-12f);
    if (!$ctx.grid.is_active(i)$ || a == 0.0f) {{
        phi[i] = psi[i] = 0.0f;
    }} else {{
        phi[i] = kd * powf(a, fmaxf($ctx.MEXP.get(i)$, 0.0f));
        psi[i] = kd / (kt * a);
    }}
}}''', domain=n).compose("grid", bundles["grid"]).freeze()

    solve = make_sfd_linear_decline(
        be, bundles["grid"], n_flat=n, n_neighbours=nk,
    )

    reset_export = KernelBuilder(f'''
extern "C" __global__ void {tag}_reset_export(float* exported) {{
    exported[0] = 0.0f;
}}''', domain=1, block=1).freeze()

    phase = KernelBuilder(f'''
extern "C" __global__ void {tag}_phase(
        const float* z, const float* old_z, float* q,
        float* sediment_thickness,
        float* rock_erosion, float* sediment_entrainment,
        float* source, float* deposition, float* exported) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    rock_erosion[i] = sediment_entrainment[i] = 0.0f;
    source[i] = deposition[i] = 0.0f;
    if (!$ctx.grid.is_active(i)$) {{ q[i] = 0.0f; return; }}
    if ($ctx.grid.can_out(i)$) {{
        atomicAdd(exported, fmaxf(q[i], 0.0f));
        return;
    }}
    float dt = fmaxf($ctx.DT.get(i)$, 0.0f);
    if (dt == 0.0f) return;
    float dz = z[i] - old_z[i];
    float h = fmaxf(sediment_thickness[i], 0.0f);
    float area = $ctx.grid.area(i)$;
    if (dz < 0.0f) {{
        float removed = -dz;
        float sed_removed = fminf(h, removed);
        sediment_thickness[i] = h - sed_removed;
        sediment_entrainment[i] = sed_removed / dt;
        rock_erosion[i] = (removed - sed_removed) / dt;
        source[i] = removed * area / dt;
    }} else {{
        sediment_thickness[i] = h + dz;
        deposition[i] = dz / dt;
    }}
}}''', domain=n).compose("grid", bundles["grid"]).freeze()

    sb = SequenceBuilder()
    for name, kernel in (("coefficients", coefficients), ("solve", solve),
                         ("reset_export", reset_export), ("phase", phase)):
        sb.add(name, kernel).step(name)
    return sb.freeze()


def shared_stream_power_plan(frozen, _be):
    """Bind the direct solver, reusing rate outputs until the phase step."""
    return _grid_leaf_plan(frozen, {
        "area": "drainage_area", "phi": "shared_phi", "psi": "shared_psi",
        "KD": "detachment_erodibility", "KT": "transport_erodibility",
        "MEXP": "m", "DT": "dt", "rec": "rec", "z": "z",
        "old_z": "shared_old_z", "remaining": "shared_remaining",
        "sum_q0": "sediment_source_rate", "sum_qp": "deposition_rate",
        "q0": "rock_erosion_rate", "qp": "sediment_entrainment_rate",
        "q": "sediment_flux",
        "queue": "shared_queue",
        "count": "shared_count", "sediment_thickness": "sediment_thickness",
        "rock_erosion": "rock_erosion_rate",
        "sediment_entrainment": "sediment_entrainment_rate",
        "source": "sediment_source_rate", "deposition": "deposition_rate",
        "exported": "sediment_export_rate",
    })


__all__ = ["shared_stream_power_factory", "shared_stream_power_plan"]
