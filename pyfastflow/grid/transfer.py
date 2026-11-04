"""GPU restriction and prolongation between grids differing by a factor of two."""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder


def build_grid_transfer_program():
    b = ProgramBuilder("GridTransferProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx")
    b.config("boundary", choices=("normal", "periodic_EW", "periodic_NS",
                                  "periodic_EW_NS"),
             default="normal")
    b.config("jitter", default=0.25)
    b.config("seed", default=0)
    coarse = (Dim("ny"), Dim("nx"))
    fine = (2 * Dim("ny"), 2 * Dim("nx"))
    b.data("coarse", "f32", coarse, role="input")
    b.data("fine", "f32", fine, role="output")
    b.data("coarse_mask", "u8", coarse, role="input")
    b.data("fine_mask", "u8", fine, role="output")

    def restrict(_be, _bundles, config):
        nx, ny = config["nx"], config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void grid_restrict_2x(
    const float* fine, float* coarse) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {nx * ny}) return;
    int y = i / {nx}, x = i % {nx};
    int j = (2 * y) * {2 * nx} + 2 * x;
    coarse[i] = 0.25f * (fine[j] + fine[j + 1]
                + fine[j + {2 * nx}] + fine[j + {2 * nx} + 1]);
}}''', domain=nx * ny).freeze()

    def prolong(_be, _bundles, config):
        nx, ny = config["nx"], config["ny"]
        periodic_x = config["boundary"] in ("periodic_EW", "periodic_EW_NS")
        periodic_y = config["boundary"] in ("periodic_NS", "periodic_EW_NS")
        jitter = float(config["jitter"])
        seed = int(config["seed"])
        if not 0.0 <= jitter <= 0.5:
            raise ValueError("jitter must be between 0 and 0.5 fine-grid cells")
        return KernelBuilder(f'''
__device__ unsigned int grid_mix(unsigned int x) {{
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}}
extern "C" __global__ void grid_prolong_2x(
    const float* coarse, float* fine) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {4 * nx * ny}) return;
    int y = i / {2 * nx}, x = i % {2 * nx};
    unsigned int hx = grid_mix((unsigned int)i ^ {seed & 0xffffffff}u);
    unsigned int hy = grid_mix(hx ^ 0x9e3779b9u);
    float jx = ((float)(hx & 65535u) / 65535.0f - 0.5f) * {2 * jitter:.9e}f;
    float jy = ((float)(hy & 65535u) / 65535.0f - 0.5f) * {2 * jitter:.9e}f;
    float xf = 0.5f * ((float)x + 0.5f + jx) - 0.5f;
    float yf = 0.5f * ((float)y + 0.5f + jy) - 0.5f;
    if (!{int(periodic_x)}) xf = fminf(fmaxf(xf, 0.0f), {nx - 1}.0f);
    if (!{int(periodic_y)}) yf = fminf(fmaxf(yf, 0.0f), {ny - 1}.0f);
    int x0 = (int)floorf(xf), y0 = (int)floorf(yf);
    float tx = xf - x0, ty = yf - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    if ({int(periodic_x)}) {{
        if (x0 < 0) x0 += {nx};
        if (x1 >= {nx}) x1 -= {nx};
    }} else {{
        x0 = max(0, min(x0, {nx - 1}));
        x1 = max(0, min(x1, {nx - 1}));
    }}
    if ({int(periodic_y)}) {{
        if (y0 < 0) y0 += {ny};
        if (y1 >= {ny}) y1 -= {ny};
    }} else {{
        y0 = max(0, min(y0, {ny - 1}));
        y1 = max(0, min(y1, {ny - 1}));
    }}
    float a = coarse[y0 * {nx} + x0];
    float b = coarse[y0 * {nx} + x1];
    float c = coarse[y1 * {nx} + x0];
    float d = coarse[y1 * {nx} + x1];
    fine[i] = (1.0f - ty) * ((1.0f - tx) * a + tx * b)
            + ty * ((1.0f - tx) * c + tx * d);
}}''', domain=4 * nx * ny).freeze()

    def prolong_mask(_be, _bundles, config):
        nx, ny = config["nx"], config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void grid_prolong_mask_2x(
    const unsigned char* coarse_mask, unsigned char* fine_mask) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {4 * nx * ny}) return;
    int y = i / {2 * nx}, x = i % {2 * nx};
    fine_mask[i] = coarse_mask[(y / 2) * {nx} + x / 2];
}}''', domain=4 * nx * ny).freeze()

    b.add("restrict", restrict, bind={"fine": "fine", "coarse": "coarse"})
    b.add("prolong", prolong, bind={"coarse": "coarse", "fine": "fine"})
    b.add("prolong_mask", prolong_mask, bind={
        "coarse_mask": "coarse_mask", "fine_mask": "fine_mask",
    })
    return b.freeze()


GridTransferProgram = build_grid_transfer_program()

__all__ = ["GridTransferProgram", "build_grid_transfer_program"]
