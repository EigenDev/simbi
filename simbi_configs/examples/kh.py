# =============================================================================
# kh.py
#
# kelvin-helmholtz instability in newtonian fluid.
# 2d shear flow problem with density contrast.
# =============================================================================
from pathlib import Path

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

# constants for initial conditions
SEED = 12345
rng = np.random.default_rng(SEED)
PEEK_TO_PEEK = 0.01


class KelvinHelmholtz(SimbiProblem):
    """kelvin-helmholtz instability in newtonian fluid."""

    # physics
    adiabatic_index: float = ProblemParam(
        5.0 / 3.0, description="adiabatic index"
    )
    rhoL: float = ProblemParam(2.0, description="density in the central layer")
    rhoR: float = ProblemParam(1.0, description="density in the outer regions")
    vxT: float = ProblemParam(
        0.5, description="x-velocity in the central layer"
    )
    vxB: float = ProblemParam(
        -0.5, description="x-velocity in the outer regions"
    )
    pL: float = ProblemParam(2.5, description="pressure in the central layer")
    pR: float = ProblemParam(2.5, description="pressure in the outer regions")

    # domain
    resolution: tuple[int, int] = ProblemParam(
        (256, 256), cli=True, description="number of zones in x and y"
    )
    bounds: list[tuple[float, float]] = ProblemParam(
        [(-0.5, 0.5), (-0.5, 0.5)], description="domain boundaries"
    )
    coord_system: CoordSystem = ProblemParam(
        CoordSystem.CARTESIAN, description="coordinate system"
    )
    regime: Regime = ProblemParam(
        Regime.NEWTONIAN, description="physics regime"
    )
    x1_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="grid spacing in x1 direction"
    )

    # numerics
    boundary_conditions: BoundaryCondition = ProblemParam(
        BoundaryCondition.PERIODIC, description="boundary conditions"
    )
    solver: Solver = ProblemParam(Solver.HLLC, description="numerical solver")

    # simulation control
    data_directory: Path = ProblemParam(
        Path("data/kh_config"),
        cli=True,
        checkpoint_safe=True,
        description="output data directory",
    )
    end_time: float = ProblemParam(
        20.0, cli=True, checkpoint_safe=True, description="end time"
    )

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for kelvin-helmholtz instability."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            ymin, ymax = self.bounds[1]

            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny

            for jj in range(ny):
                y = ymin + jj * dy
                for ii in range(nx):
                    vx_noise = PEEK_TO_PEEK * np.sin(2 * np.pi * rng.normal())
                    vy_noise = PEEK_TO_PEEK * np.sin(2 * np.pi * rng.normal())

                    if abs(y) < 0.25:
                        rho = self.rhoL
                        vx = self.vxT + vx_noise
                        vy = 0.0 + vy_noise
                        p = self.pL
                    else:
                        rho = self.rhoR
                        vx = self.vxB + vx_noise
                        vy = 0.0 + vy_noise
                        p = self.pR

                    yield (rho, vx, vy, p)

        return gas_state
