"""SPACE-style SFD sediment entrainment, routing, and deposition."""

import math

from pyfastflow.core import KernelBuilder, SequenceBuilder, new_uid
from pyfastflow.flow._program_cupy import _cupy_only, _grid_leaf_plan


def sediment_factory(be, bundles, config):
    """Build one explicit SPACE entrainment and transport-length step.

    Each pointer-jump pass pushes a cell's current flux to its receiver while
    carrying the survival product over its compressed path.  The final flux
    at every node includes its local and upstream incoming sediment.
    """
    _cupy_only(be)
    n = int(config["nx"]) * int(config["ny"])
    rounds = math.ceil(math.log2(max(1, n))) + 2
    if rounds % 2:
        rounds += 1
    tag = f"golem_sed_{new_uid()}"

    entrain = KernelBuilder(
        f'''extern "C" __global__ void {tag}_entrain(
                float* z, float* sediment_thickness, const int* rec,
                const float* area, float* source, float* survival,
                float* rock_erosion, float* sediment_entrainment) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            source[i] = survival[i] = rock_erosion[i] = sediment_entrainment[i] = 0.0f;
            if (!$ctx.grid.is_active(i)$ || $ctx.grid.can_out(i)$) return;
            int r = rec[i];
            if (r < 0 || r >= {n} || r == i || !$ctx.grid.is_active(r)$) return;
            float length = fmaxf($ctx.grid.dist_between_nodes(i, r)$, 1.0e-12f);
            float slope = fmaxf((z[i] - z[r]) / length,
                                 fmaxf($ctx.SLOPE_FLOOR.get(i)$, 0.0f));
            float stream_power = powf(fmaxf(area[i], 0.0f),
                                       fmaxf($ctx.MEXP.get(i)$, 0.0f))
                               * powf(slope, fmaxf($ctx.NEXP.get(i)$, 1.0e-4f));
            float dt = fmaxf($ctx.DT.get(i)$, 0.0f);
            float h = fmaxf(sediment_thickness[i], 0.0f);
            float cover_scale = fmaxf($ctx.COVER_SCALE.get(i)$, 1.0e-12f);
            float exposed_rock = expf(-h / cover_scale);
            float erode_sediment = fmaxf($ctx.SEDIMENT_ERODIBILITY.get(i)$, 0.0f)
                                  * stream_power * (1.0f - exposed_rock);
            erode_sediment = dt > 0.0f ? fminf(erode_sediment, h / dt) : 0.0f;
            float erode_rock = fmaxf($ctx.ROCK_ERODIBILITY.get(i)$, 0.0f)
                              * stream_power * exposed_rock;
            if (!isfinite(erode_sediment) || erode_sediment < 0.0f) erode_sediment = 0.0f;
            if (!isfinite(erode_rock) || erode_rock < 0.0f) erode_rock = 0.0f;
            sediment_thickness[i] = fmaxf(0.0f, h - dt * erode_sediment);
            z[i] -= dt * (erode_rock + erode_sediment);
            rock_erosion[i] = erode_rock;
            sediment_entrainment[i] = erode_sediment;
            source[i] = (erode_rock + erode_sediment)
                      * $ctx.grid.dx(i)$ * $ctx.grid.dx(i)$;
            float transport_length = $ctx.TRANSPORT_LENGTH.get(i)$;
            survival[i] = transport_length > 0.0f
                ? expf(-length / transport_length) : 0.0f;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()
    initialize = KernelBuilder(
        f'''extern "C" __global__ void {tag}_initialize(
                const float* source, const float* survival, const int* rec,
                float* q, float* path, int* rec_work) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            q[i] = source[i];
            path[i] = survival[i];
            rec_work[i] = rec[i];
        }}''', domain=n,
    ).freeze()

    def copy(name):
        return KernelBuilder(
            f'''extern "C" __global__ void {tag}_{name}(
                    const float* q_current, float* q_next,
                    const float* path_current, float* path_next) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < {n}) {{
                    q_next[i] = q_current[i];
                    path_next[i] = path_current[i];
                }}
            }}''', domain=n,
        ).freeze()

    def jump(name):
        return KernelBuilder(
            f'''extern "C" __global__ void {tag}_{name}(
                    const int* rec_current, int* rec_next,
                    const float* q_current, float* q_next,
                    const float* path_current, float* path_next) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= {n}) return;
                int parent = rec_current[i];
                rec_next[i] = i;
                if (parent < 0 || parent >= {n} || parent == i) return;
                float flux = q_current[i];
                if (flux != 0.0f) atomicAdd(&q_next[parent], path_current[i] * flux);
                int grandparent = rec_current[parent];
                if (grandparent >= 0 && grandparent < {n} && grandparent != parent) {{
                    rec_next[i] = grandparent;
                    path_next[i] = path_current[i] * path_current[parent];
                }}
            }}''', domain=n,
        ).freeze()

    copy_a, copy_b = copy("copy_a"), copy("copy_b")
    jump_a, jump_b = jump("jump_a"), jump("jump_b")
    reset_export = KernelBuilder(
        f'''extern "C" __global__ void {tag}_reset_export(float* exported) {{
            exported[0] = 0.0f;
        }}''', domain=1, block=1,
    ).freeze()
    settle = KernelBuilder(
        f'''extern "C" __global__ void {tag}_settle(
                float* z, float* sediment_thickness, const float* survival,
                float* q, float* deposition, float* exported) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            deposition[i] = 0.0f;
            if (!$ctx.grid.is_active(i)$) {{ q[i] = 0.0f; return; }}
            float incoming = fmaxf(q[i], 0.0f);
            if ($ctx.grid.can_out(i)$) {{
                atomicAdd(exported, incoming);
                q[i] = incoming;
                return;
            }}
            float retained = fminf(fmaxf(survival[i], 0.0f), 1.0f);
            float deposited = incoming * (1.0f - retained);
            float dt = fmaxf($ctx.DT.get(i)$, 0.0f);
            float area = $ctx.grid.area(i)$;
            deposition[i] = deposited / area;
            sediment_thickness[i] += dt * deposition[i];
            z[i] += dt * deposition[i];
            q[i] = incoming * retained;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    sequence = SequenceBuilder()
    for name, node in (
        ("entrain", entrain), ("initialize", initialize),
        ("copy_a", copy_a), ("jump_a", jump_a),
        ("copy_b", copy_b), ("jump_b", jump_b),
        ("reset_export", reset_export), ("settle", settle),
    ):
        sequence.add(name, node)
    sequence.step("entrain").step("initialize")
    for _ in range(rounds // 2):
        sequence.step("copy_a").step("jump_a").step("copy_b").step("jump_b")
    sequence.step("reset_export").step("settle")
    return sequence.freeze()


def sediment_plan(frozen, _be):
    """Bind the sediment transport sequence and its work buffers."""
    plan = _grid_leaf_plan(frozen, {
        "z": "z", "sediment_thickness": "sediment_thickness", "rec": "rec",
        "area": "drainage_area", "source": "sediment_source_rate",
        "survival": "sediment_survival", "rock_erosion": "rock_erosion_rate",
        "sediment_entrainment": "sediment_entrainment_rate",
        "ROCK_ERODIBILITY": "rock_erodibility",
        "SEDIMENT_ERODIBILITY": "sediment_erodibility", "COVER_SCALE": "cover_scale",
        "TRANSPORT_LENGTH": "transport_length", "MEXP": "m", "NEXP": "n",
        "SLOPE_FLOOR": "slope_floor", "DT": "dt",
        "q": "sediment_flux", "path": "sediment_path_a",
        "rec_work": "sediment_rec_a", "q_current": "sediment_flux",
        "q_next": "sediment_q_alt", "path_current": "sediment_path_a",
        "path_next": "sediment_path_b", "rec_current": "sediment_rec_a",
        "rec_next": "sediment_rec_b", "exported": "sediment_export_rate",
        "deposition": "deposition_rate",
    })
    plan.update({
        "copy_b.q_current": "sediment_q_alt", "copy_b.q_next": "sediment_flux",
        "copy_b.path_current": "sediment_path_b", "copy_b.path_next": "sediment_path_a",
        "jump_b.rec_current": "sediment_rec_b", "jump_b.rec_next": "sediment_rec_a",
        "jump_b.q_current": "sediment_q_alt", "jump_b.q_next": "sediment_flux",
        "jump_b.path_current": "sediment_path_b", "jump_b.path_next": "sediment_path_a",
    })
    return plan


__all__ = ["sediment_factory", "sediment_plan"]
