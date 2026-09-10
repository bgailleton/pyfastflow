"""A reusable CuPy Perlin-noise :class:`Program`.

Example
-------
>>> noise = PerlinNoiseProgram(Backend.from_name("cupy"), nx=2048, ny=2048)
>>> noise.generate()
>>> terrain = noise.noise.array       # a cupy.ndarray, no host copy
>>> noise.close()

The device result can be adopted by ``SFDFlowProgram.z`` before closing this
Program, allowing the two Programs to compose without a host round trip.
"""

from pyfastflow.core import Backend, KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.noise import make_noise_group, make_noise_parameters


def _cupy_only(be: Backend) -> None:
    if be.name != "cupy":
        raise ValueError(f"experimental PerlinNoiseProgram is cupy-only, got {be.name!r}")


def build_perlin_noise_program() -> type:
    """Build the configurable Perlin-noise Program class."""
    b = ProgramBuilder("PerlinNoiseProgram")
    b.dim("ny").dim("nx")
    b.config("ny")
    b.config("nx")
    b.config("amplitude", default=300.0)
    b.config("frequency", default=5.0)
    b.config("octaves", default=6)
    b.config("persistence", default=0.5)
    b.config("seed", default=42)
    b.data("noise", "f32", (Dim("ny"), Dim("nx")), role="output")

    def noise_structure(be, **_):
        _cupy_only(be)
        return make_noise_group(be, kind="perlin")

    def noise_params(be, pool, *, nx, ny, amplitude, frequency, octaves, persistence, seed):
        _cupy_only(be)
        return make_noise_parameters(
            be, pool, kind="perlin", amplitude=amplitude, frequency=frequency,
            octaves=octaves, persistence=persistence, seed=seed,
        ) | {
            "NX": be.ParameterCls("NOISE_NX", dtype="i32", mode="const", value=nx, pool=pool),
            "NY": be.ParameterCls("NOISE_NY", dtype="i32", mode="const", value=ny, pool=pool),
        }

    selected = ("amplitude", "frequency", "octaves", "persistence", "seed")
    b.bundle("perlin", noise_structure, noise_params, dims=("nx", "ny"), config=selected)

    def generate_factory(be, bundles, config):
        _cupy_only(be)
        n = config["nx"] * config["ny"]
        source = f'''extern "C" __global__ void generate_perlin(float* noise) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) noise[i] = $ctx.perlin.at(i)$;
        }}'''
        return KernelBuilder(source, domain=n).compose("perlin", bundles["perlin"]).freeze()

    b.add("generate", generate_factory, bind={"perlin": "perlin", "noise": "noise"})
    return b.freeze()


PerlinNoiseProgram = build_perlin_noise_program()

__all__ = ["PerlinNoiseProgram", "build_perlin_noise_program"]
