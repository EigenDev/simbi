# =============================================================================
# test_mhd_nonideal_plumbing.py
#
# the config-to-kernel plumbing gate for the MHD non-ideal knobs: a run with
# resistivity (or viscosity) set MUST evolve differently from the ideal run of
# the same setup. this is deliberately a plumbing gate —
# the diffusion operators themselves are oracle-tested in rust (the mimetic
# adjoint identity, viscous-heating equivalence); what only an end-to-end run
# can catch is a builder chain that drops the config value on the floor, which
# makes every "resistive" or "viscous" MHD run silently ideal.
# =============================================================================
import glob
import os
import tempfile
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.simulation import runner
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

RES = 32
L = 0.5


class _MagnetizedVortex2D(SimbiProblem):
    """a smooth magnetized shear on the 2d cartesian grid: nonzero J and nonzero
    velocity gradients, so BOTH non-ideal operators have something to diffuse."""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0)]
    regime: Annotated[Regime, ProblemParam(Regime.NMHD)]
    resolution: Annotated[tuple[int, int], ProblemParam((RES, RES))]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(-L, L), (-L, L)])
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN)]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE)]
    boundary_conditions: Annotated[
        BoundaryCondition, ProblemParam(BoundaryCondition.PERIODIC)
    ]
    cfl_number: Annotated[float, ProblemParam(0.3)]
    end_time: Annotated[float, ProblemParam(0.1, checkpoint_safe=True)]
    nu: Annotated[float, ProblemParam(0.0, cli=True, description="kinematic viscosity")]
    eta: Annotated[float, ProblemParam(0.0, cli=True, description="ohmic resistivity")]
    checkpoint_interval: Annotated[
        float, ProblemParam(1.0e30, cli=True, checkpoint_safe=True)
    ]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu

    @computed_field
    @property
    def resistivity(self) -> float:
        return self.eta
    data_directory: Annotated[
        Path, ProblemParam(Path("data/_test"), cli=True, checkpoint_safe=True)
    ]

    def initial_primitive_state(self) -> InitialStateType:
        nx, ny = self.resolution
        d = 2.0 * L / RES
        xc = lambda i: -L + (i + 0.5) * d
        xf = lambda i: -L + i * d
        two_pi = 2.0 * np.pi

        def gas_state() -> GasStateGenerator:
            for j in range(ny):
                for i in range(nx):
                    vx = 0.1 * np.sin(two_pi * xc(j) / (2 * L))
                    vy = 0.1 * np.sin(two_pi * xc(i) / (2 * L))
                    yield (1.0, vx, vy, 0.0, 1.0)

        # a_z(x, y) = cos(2 pi x / (2L)) + cos(2 pi y / (2L)) on the edges: the
        # discrete edge curl gives an exactly divergence-free in-plane loop field
        # with nonzero J_z.
        def a_z(x: float, y: float) -> float:
            return (np.cos(two_pi * x / (2 * L)) + np.cos(two_pi * y / (2 * L))) / two_pi

        def b_field(bn: str):
            def gen():
                if bn == "b1":
                    for j in range(ny):
                        for i in range(nx + 1):
                            yield (a_z(xf(i), xf(j + 1)) - a_z(xf(i), xf(j))) / d
                elif bn == "b2":
                    for j in range(ny + 1):
                        for i in range(nx):
                            yield -(a_z(xf(i + 1), xf(j)) - a_z(xf(i), xf(j))) / d
                else:
                    for _ in range(nx * ny):
                        yield 0.0

            return gen

        return (gas_state, b_field("b1"), b_field("b2"), b_field("b3"))


def _run(nu: float, eta: float) -> dict[str, np.ndarray]:
    d = tempfile.mkdtemp() + "/"
    p = _MagnetizedVortex2D(nu=nu, eta=eta, data_directory=Path(d))
    runner.run(p, compute_mode="cpu")
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, f"nonideal MHD run (nu={nu}, eta={eta}) crashed"
    out = {}
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - RES) // 2
        sl = slice(halo, halo + RES)
        for nm in ("rho", "v1", "b1"):
            out[nm] = prims[nm][sl, sl]
    return out


@needs_backend
def test_resistivity_from_config_genuinely_acts() -> None:
    ideal = _run(0.0, 0.0)
    resistive = _run(0.0, 0.05)
    db = np.abs(ideal["b1"] - resistive["b1"]).max()
    bscale = np.abs(ideal["b1"]).max()
    assert bscale > 1e-6, "the field never developed; the comparison is vacuous"
    assert db > 1e-6 * bscale, (
        f"resistivity = 0.05 left the field bit-near-identical to ideal ({db:e}); "
        "the config value is being dropped by the builder chain"
    )


@needs_backend
def test_viscosity_from_config_genuinely_acts() -> None:
    ideal = _run(0.0, 0.0)
    viscous = _run(0.05, 0.0)
    dv = np.abs(ideal["v1"] - viscous["v1"]).max()
    vscale = np.abs(ideal["v1"]).max()
    assert vscale > 1e-6, "the shear never developed; the comparison is vacuous"
    assert dv > 1e-6 * vscale, (
        f"viscosity = 0.05 left the velocity bit-near-identical to ideal ({dv:e}); "
        "the config value is being dropped by the builder chain"
    )
