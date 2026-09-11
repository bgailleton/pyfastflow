"""Reusable CuPy single- and multi-direction hillshading Program."""

from pyfastflow.core import Backend, KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder


def _cupy_only(be: Backend) -> None:
    if be.name != "cupy":
        raise ValueError(f"experimental HillshadeProgram is cupy-only, got {be.name!r}")


def _shade_factory(multidirectional: bool):
    def factory(be, _bundles, config):
        _cupy_only(be)
        nx, ny = int(config["nx"]), int(config["ny"])
        n = nx * ny
        dx = float(config["dx"])
        altitude = float(config["altitude"])
        azimuth = float(config["azimuth"])
        z_factor = float(config["z_factor"])
        nlights = 4 if multidirectional else 1
        source = f'''
__device__ float shade_one(float gx, float gy, float altitude, float azimuth) {{
    const float deg = 0.017453292519943295f;
    float slope = atan2f(sqrtf(gx * gx + gy * gy), 1.0f);
    float aspect = 0.0f;
    if (gx != 0.0f || gy != 0.0f) {{
        aspect = 1.5707963267948966f - atan2f(gy, gx);
        if (aspect < 0.0f) aspect += 6.283185307179586f;
    }}
    float zenith = 1.5707963267948966f - altitude * deg;
    float value = cosf(zenith) * cosf(slope)
        + sinf(zenith) * sinf(slope) * cosf(azimuth * deg - aspect);
    return fmaxf(0.0f, fminf(1.0f, value));
}}

extern "C" __global__ void terrain_shade(const float* z, float* image) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    int row = i / {nx};
    int col = i - row * {nx};
    int left = col > 0 ? i - 1 : i;
    int right = col + 1 < {nx} ? i + 1 : i;
    int top = row > 0 ? i - {nx} : i;
    int bottom = row + 1 < {ny} ? i + {nx} : i;
    float gx = (z[right] - z[left]) * (float)({z_factor:.9g}) / (2.0f * (float)({dx:.9g}));
    float gy = (z[bottom] - z[top]) * (float)({z_factor:.9g}) / (2.0f * (float)({dx:.9g}));
    float value = 0.0f;
    #pragma unroll
    for (int light = 0; light < {nlights}; light++)
        value += shade_one(gx, gy, (float)({altitude:.9g}), (float)({azimuth:.9g}) + 90.0f * light);
    image[i] = value / {float(nlights):.1f}f;
}}
'''
        return KernelBuilder(source, domain=n).freeze()

    return factory


def build_hillshade_program() -> type:
    """Build a CuPy shading Program with single and four-light modes."""
    b = ProgramBuilder("HillshadeProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx")
    b.config("method", choices=("hillshade", "multishade"), default="multishade")
    b.config("dx", default=1.0)
    b.config("azimuth", default=315.0)
    b.config("altitude", default=45.0)
    b.config("z_factor", default=1.0)
    b.data("z", "f32", (Dim("ny"), Dim("nx")), role="input", shape_source=True)
    b.data("image", "f32", (Dim("ny"), Dim("nx")), role="output")
    binding = {"z": "z", "image": "image"}
    b.add("render_hillshade", _shade_factory(False), bind=binding)
    b.add("render_multishade", _shade_factory(True), bind=binding)
    b.dispatch("render", on="method", cases={
        "hillshade": "render_hillshade",
        "multishade": "render_multishade",
    })
    return b.freeze()


HillshadeProgram = build_hillshade_program()

__all__ = ["HillshadeProgram", "build_hillshade_program"]
