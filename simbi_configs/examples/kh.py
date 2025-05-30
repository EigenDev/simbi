from pathlib import Path
import numpy as np

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import (
    CoordSystem,
    Regime,
    CellSpacing,
    Solver,
    BoundaryCondition,
)
from simbi.core.types.typing import GasStateGenerator, InitialStateType

# Constants for initial conditions
SEED = 12345
rng = np.random.default_rng(SEED)
PEEK_TO_PEEK = 0.01


class KelvinHelmholtz(SimbiBaseConfig):
    """
    Kelvin Helmholtz problem in Newtonian Fluid
    """

    # Configuration parameters
    resolution: tuple[int, int] = SimbiField(
        (256, 256), description="Number of zones in x and y dimensions"
    )

    # Physical parameters
    rhoL: float = SimbiField(2.0, description="Density in the central layer")
    rhoR: float = SimbiField(1.0, description="Density in the outer regions")
    vxT: float = SimbiField(0.5, description="x-velocity in the central layer")
    vxB: float = SimbiField(-0.5, description="x-velocity in the outer regions")
    pL: float = SimbiField(2.5, description="Pressure in the central layer")
    pR: float = SimbiField(2.5, description="Pressure in the outer regions")

    bounds: list[tuple[float, float]] = SimbiField(
        [(-0.5, 0.5), (-0.5, 0.5)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.CLASSICAL, description="Physics regime")

    adiabatic_index: float = SimbiField(5.0 / 3.0, description="Adiabatic index")

    # Optional customizations with non-default values
    boundary_conditions: BoundaryCondition = SimbiField(
        BoundaryCondition.PERIODIC, description="Boundary conditions"
    )

    solver: Solver = SimbiField(Solver.HLLC, description="Numerical solver")

    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    data_directory: Path = SimbiField(
        Path("data/kh_config"), description="Output data directory"
    )

    end_time: float = SimbiField(20.0, description="End time for the simulation")

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for Kelvin-Helmholtz instability.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            ymin, ymax = self.bounds[1]

            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny

            for j in range(ny):
                y = ymin + j * dy
                for i in range(nx):
                    # Add random perturbations to velocity
                    vx_noise = PEEK_TO_PEEK * np.sin(2 * np.pi * rng.normal())
                    vy_noise = PEEK_TO_PEEK * np.sin(2 * np.pi * rng.normal())

                    # Set properties based on y-position (shear layer)
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
