import math
from functools import partial

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import (
    CoordSystem,
    BoundaryCondition,
    Regime,
    CellSpacing,
    Solver,
)
from simbi.core.types.typing import (
    InitialStateType,
    GasStateGenerator,
    StaggeredBFieldGenerator,
)

# Constants for the blast wave setup
XMIN = -6.0
XMAX = 6.0
P_EXP = 1.0
RHO_EXP = 0.1
R_EXP = 0.08
R_STOP = 1.0


class MagneticBomb(SimbiBaseConfig):
    """The Magnetic Bomb
    Launch a cylindrical relativistic magnetized blast wave
    """

    # Configuration parameters
    rho0: float = SimbiField(1.0e-4, description="Density scale")
    p0: float = SimbiField(3.0e-5, description="Pressure scale")
    b0: float = SimbiField(0.1, description="Magnetic field scale")

    # Required fields from SimbiBaseConfig
    resolution: tuple[int, int] = SimbiField((256, 256), description="Grid resolution")

    bounds: list[tuple[float, float]] = SimbiField(
        [(XMIN, XMAX), (XMIN, XMAX)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.SRMHD, description="Physics regime")

    adiabatic_index: float = SimbiField(4.0 / 3.0, description="Adiabatic index")

    # Optional customizations with non-default values
    solver: Solver = SimbiField(Solver.HLLE, description="Numerical solver")

    boundary_conditions: list[BoundaryCondition] = SimbiField(
        [BoundaryCondition.OUTFLOW], description="Boundary conditions"
    )

    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    start_time: float = SimbiField(0.0, description="Simulation start time")

    end_time: float = SimbiField(4.0, description="Simulation end time")

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for magnetic blast wave.

        Returns:
            Tuple of generator functions for gas state and B-fields
        """

        # Gas state generator for density, velocity, and pressure
        def gas_state() -> GasStateGenerator:
            ni, nj = self.resolution
            nk = 1
            xbounds = self.bounds[0]
            ybounds = self.bounds[1]
            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj

            rho_amb = float(self.rho0)
            pre_amb = float(self.p0)
            pslope = (P_EXP - pre_amb) / (R_STOP - R_EXP)
            rhoslope = (RHO_EXP - rho_amb) / (R_STOP - R_EXP)

            for k in range(nk):
                for j in range(nj):
                    y = ybounds[0] + (j + 0.5) * dy
                    for i in range(ni):
                        x = xbounds[0] + (i + 0.5) * dx
                        r = math.sqrt(x**2 + y**2)

                        if r < R_EXP:
                            # Inside explosion region
                            yield (RHO_EXP, 0.0, 0.0, 0.0, P_EXP)
                        elif r > R_EXP and r < R_STOP:
                            # Transition region
                            yield (
                                RHO_EXP - rhoslope * (r - R_EXP),
                                0.0,
                                0.0,
                                0.0,
                                P_EXP - pslope * (r - R_EXP),
                            )
                        else:
                            # Ambient region
                            yield (rho_amb, 0.0, 0.0, 0.0, pre_amb)

        # B-field generator function
        def b_field(bn: str) -> StaggeredBFieldGenerator:
            """Generate B-field component values"""
            ni, nj = self.resolution
            nk = 1

            # Different grid sizes for different components due to staggering
            for k in range(nk + (bn == "bz")):
                for j in range(nj + (bn == "by")):
                    for i in range(ni + (bn == "bx")):
                        if bn == "bx":
                            yield float(self.b0)
                        else:
                            yield 0.0

        # Create partial functions for each B-field component
        bx_gen = partial(b_field, "bx")
        by_gen = partial(b_field, "by")
        bz_gen = partial(b_field, "bz")

        # Return tuple of generator functions
        return (gas_state, bx_gen, by_gen, bz_gen)
