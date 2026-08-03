# =============================================================================
# turbulence_decay.py
#
# decaying subsonic turbulence in a periodic box: a solenoidal (divergence-free)
# velocity field seeded at a target rms mach number, no gravity, no forcing.
# the kinetic-energy decay rate measures the scheme's effective numerical
# dissipation at a given resolution — comparing reconstructions at fixed
# resolution (and one reconstruction across resolutions) separates scheme
# dissipation from grid resolution.
# usage:
#  simbi run turbulence_decay --reconstruction plm --resolution 64,64,64
#  simbi run turbulence_decay --reconstruction ppm --resolution 64,64,64
#  simbi run turbulence_decay --reconstruction plm --resolution 128,128,128
# =============================================================================
from pathlib import Path
from typing import Annotated

import numpy as np

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.types.typing import GasStateGenerator, InitialStateType

# the seed only breaks symmetry; the saturated statistics are seed-independent,
# and a FIXED seed makes any two runs differ only in the numerics under test.
SEED = 20260803


class TurbulenceDecay(SimbiProblem):
    """decaying solenoidal subsonic turbulence in a triply periodic box."""

    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    mach_rms: Annotated[
        float,
        ProblemParam(
            0.7,
            cli=True,
            description="initial rms mach number of the solenoidal field",
        ),
    ]
    k_min: Annotated[
        int, ProblemParam(1, description="lowest seeded wavenumber (box units)")
    ]
    k_max: Annotated[
        int, ProblemParam(4, description="highest seeded wavenumber (box units)")
    ]

    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((64, 64, 64), cli=True, description="zones per axis"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)],
            description="domain boundaries",
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]
    boundary_conditions: Annotated[
        BoundaryCondition,
        ProblemParam(
            BoundaryCondition.PERIODIC, description="boundary conditions"
        ),
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="riemann solver")
    ]

    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/turbulence_decay"),
            cli=True,
            checkpoint_safe=True,
            description="output data directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            4.0,
            cli=True,
            checkpoint_safe=True,
            description="end time (~2.8 eddy turnovers at mach 0.7)",
        ),
    ]
    checkpoint_interval: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            checkpoint_safe=True,
            description="checkpoint cadence for the kinetic-energy series",
        ),
    ]

    def _solenoidal_velocity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """band-limited divergence-free velocity at the target rms mach.

        v = curl A with A drawn as random fourier modes restricted to
        k_min <= |k| <= k_max: the curl is divergence-free by construction and
        the band limit keeps the seed resolved at every resolution compared.
        """
        n = self.resolution[0]
        rng = np.random.default_rng(SEED)
        # hermitian-symmetric white noise via a real-space draw, then band mask
        a_hat = [np.fft.rfftn(rng.standard_normal((n, n, n))) for _ in range(3)]
        k1 = np.fft.fftfreq(n, d=1.0 / n)
        kx, ky, kz = np.meshgrid(
            k1, k1, np.fft.rfftfreq(n, d=1.0 / n) * n, indexing="ij"
        )
        kmag = np.sqrt(kx**2 + ky**2 + kz**2)
        band = (kmag >= self.k_min) & (kmag <= self.k_max)
        for a in a_hat:
            a *= band
        # v = curl A in fourier space: v_hat = i k x A_hat
        two_pi_i = 2.0j * np.pi
        vx = np.fft.irfftn(two_pi_i * (ky * a_hat[2] - kz * a_hat[1]), s=(n, n, n))
        vy = np.fft.irfftn(two_pi_i * (kz * a_hat[0] - kx * a_hat[2]), s=(n, n, n))
        vz = np.fft.irfftn(two_pi_i * (kx * a_hat[1] - ky * a_hat[0]), s=(n, n, n))
        cs = 1.0  # rho0 = 1, p0 = 1/gamma -> cs = sqrt(gamma p / rho) = 1
        v_rms = np.sqrt(np.mean(vx**2 + vy**2 + vz**2))
        scale = self.mach_rms * cs / v_rms
        return vx * scale, vy * scale, vz * scale

    def initial_primitive_state(self) -> InitialStateType:
        """uniform gas carrying the solenoidal velocity seed."""

        def gas_state() -> GasStateGenerator:
            nx, ny, nz = self.resolution
            assert nx == ny == nz, "the fourier seed assumes a cubic grid"
            vx, vy, vz = self._solenoidal_velocity()
            rho0 = 1.0
            p0 = 1.0 / self.adiabatic_index  # cs = 1 at rho0 = 1
            for kk in range(nz):
                for jj in range(ny):
                    for ii in range(nx):
                        yield (
                            rho0,
                            vx[ii, jj, kk],
                            vy[ii, jj, kk],
                            vz[ii, jj, kk],
                            p0,
                        )

        return gas_state
