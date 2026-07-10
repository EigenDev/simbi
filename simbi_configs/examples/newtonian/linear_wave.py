# =============================================================================
# linear_wave.py
#
# the athenak table-1 benchmark, reproduced like-for-like.
#
# stone et al., "AthenaK: A Performance-Portable Version of the Athena++ AMR
# Framework", section 5.1: performance is measured with the newtonian hydro
# solver running a LINEAR WAVE convergence test on a UNIFORM mesh, on CPUs using
# a SINGLE 128^3 MeshBlock and all available cores. their reported figures:
#
#   Apple M1 pro (8 cores):   hydro 34 Mzc/s,  MHD 11 Mzc/s
#   Intel Xeon Gold 6326 (32 cores):  hydro 63,  MHD 33,  SR-hydro 16
#
# this config is the ONLY simbi configuration directly comparable to those
# numbers: cartesian, uniform, periodic, newtonian hydro, no sources, no
# immersed bodies, no mesh refinement, no curvilinear geometry. any of those
# additions costs a multiple (their own table 2 shows hydro 63 -> SR-hydro 16 on
# one machine), so a spherical/AMR/body problem is NOT a valid comparison point.
#
# physics: a right-going linearized sound wave on a uniform background. with
# rho0 = 1 and p0 = 1/gamma the sound speed is cs = sqrt(gamma p0 / rho0) = 1, so
# the isentropic eigenvector collapses to a single profile s(x) = A sin(kx):
#
#   rho = rho0 + s        (drho = s)
#   vx  =        s        (dv = cs drho / rho0 = s)
#   p   = p0   + s        (dp = cs^2 drho = s)
#
# amplitude 1e-6 keeps it in the linear regime: smooth, shock-free, no limiter
# activation, no floors. at cs = 1 on a unit box the wave completes exactly one
# crossing at t = 1, so the final state should equal the initial state to
# truncation error -- a free correctness check on the benchmark itself.
#
# throughput (MZCS) is cell-updates per wall-second and is INDEPENDENT of the cfl
# number; cfl only sets how many steps a given end_time takes.
#
# the problem is RANK-GENERIC: `dim` follows len(resolution), and the wave always
# travels along x1, so 1d / 2d / 3d run the IDENTICAL physics and solver. that makes
# throughput comparable ACROSS ranks -- comparing a 3d wave against a 2d shear
# instability measures the problem, not the dimension. at equal cell counts:
#
#   1d: --resolution 262144
#   2d: --resolution 512 512
#   3d: --resolution 64 64 64
#
# all three are 262144 cells, all above the host cover's WHOLE_BELOW_CELLS cutoff.
# per-cell work should rise with rank (flux directions D, conserved fields 2 + D),
# so MZCS should FALL from 1d -> 3d. if it does not, the bottleneck is a fixed
# per-cell cost rather than memory traffic.
#
# usage:
#   simbi run linear_wave                       # 128^3, hllc -- the athenak comparison
#   simbi run linear_wave --solver hlle
#   simbi run linear_wave --resolution 64 64 64 --end-time 0.25   # quick pass
#   SYMBI_PROFILE=1 simbi run linear_wave       # per-phase wall-time breakdown
#   SYMBI_BLOCK=off simbi run linear_wave       # a/b the host cache-tile cover
#   SYMBI_BLOCK=32  simbi run linear_wave
# =============================================================================
from pathlib import Path
from typing import Annotated

import numpy as np

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.types.typing import GasStateGenerator, InitialStateType


class LinearWave(SimbiProblem):
    """right-going linear sound wave on a uniform 3d cartesian mesh."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index", group="physics")
    ]
    amplitude: Annotated[
        float,
        ProblemParam(
            1.0e-6,
            cli=True,
            description="wave amplitude; small enough to stay linear",
            group="physics",
        ),
    ]

    # domain -- a single uniform 128^3 block, matching the athenak cpu benchmark.
    # len(resolution) selects the rank; the wave always travels along x1.
    resolution: Annotated[
        tuple[int, ...],
        ProblemParam(
            (128, 128, 128), cli=True, description="zones per axis; len selects the rank", group="grid"
        ),
    ]
    bounds: Annotated[
        list[tuple[float, float]] | None,
        ProblemParam(
            None,
            description="unit box per axis; auto-filled from the rank",
            group="grid",
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system", group="grid"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime", group="physics")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="grid spacing in x1", group="grid"),
    ]

    # numerics
    boundary_conditions: Annotated[
        BoundaryCondition,
        ProblemParam(BoundaryCondition.PERIODIC, description="boundary conditions", group="numerics"),
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="riemann solver", group="numerics")
    ]
    # 0.1 (the base default) would triple the step count for the same end_time. the
    # measured MZCS is unaffected -- only the wall-clock of a full crossing is.
    cfl_number: Annotated[
        float, ProblemParam(0.4, cli=True, description="cfl number", group="numerics")
    ]

    # simulation control
    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/linear_wave"),
            cli=True,
            checkpoint_safe=True,
            description="output data directory",
            group="output",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            1.0,
            cli=True,
            checkpoint_safe=True,
            description="one full wave crossing at cs = 1",
            group="output",
        ),
    ]
    # a 128^3 checkpoint is ~84 MB; the base default (0.1) writes ten of them over a
    # crossing and the i/o would land inside the timed region. only the final state.
    checkpoint_interval: Annotated[
        float,
        ProblemParam(
            1.0e9,
            cli=True,
            checkpoint_safe=True,
            description="effectively disabled; the benchmark measures compute",
            group="output",
        ),
    ]

    def setup(self) -> None:
        """unit box on every gridded axis; the rank follows len(resolution)."""
        super().setup()
        if self.bounds is None:
            self.bounds = [(0.0, 1.0)] * len(self.resolution)

    def initial_primitive_state(self) -> InitialStateType:
        """right-going sound wave: rho = 1 + s, v1 = s, p = 1/gamma + s, s = A sin(k x1)."""

        def gas_state() -> GasStateGenerator:
            res = self.resolution
            dim = len(res)
            nx = res[0]
            x0, x1 = self.bounds[0]
            gamma = float(self.adiabatic_index)
            # cs = sqrt(gamma * p0 / rho0) = 1 with this normalization, which is what
            # collapses the sound-wave eigenvector to one shared profile.
            rho0, p0 = 1.0, 1.0 / gamma
            amp = float(self.amplitude)

            dx = (x1 - x0) / nx
            k = 2.0 * np.pi / (x1 - x0)

            # the wave depends on x1 ALONE. evaluate it once per x1-column (nx values) and
            # re-emit the row: a per-cell sin() at 128^3 is 2.1M transcendentals and would
            # dominate startup before a single step runs.
            xc = x0 + (np.arange(nx) + 0.5) * dx
            s = amp * np.sin(k * xc)
            # gas tuple is (rho, v1..v_dim, p); only the x1 velocity is perturbed.
            zeros = (0.0,) * (dim - 1)
            row = [
                (float(rho0 + si), float(si), *zeros, float(p0 + si)) for si in s
            ]

            # the transverse axes just repeat the row; iterate them as one flat count so
            # the generator is rank-generic (innermost axis is x1, matching the loader).
            transverse = 1
            for n in res[1:]:
                transverse *= n
            for _ in range(transverse):
                yield from row

        return gas_state
