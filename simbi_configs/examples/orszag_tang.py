import math
from functools import partial

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import (
    BoundaryCondition,
    CoordSystem,
    Regime,
    CellSpacing,
    Solver,
)
from simbi.core.types.typing import (
    InitialStateType,
    GasStateGenerator,
    StaggeredBFieldGenerator,
)
from pydantic import computed_field
from typing import Any

# Domain constants
XMIN = 0.0
XMAX = 1.0


class OrszagTang(SimbiBaseConfig):
    """The Orszag-Tang vortex test case"""

    # Configuration parameters
    v0: float = SimbiField(0.5, description="Velocity scale")
    b0: float = SimbiField(1.0, description="Magnetic field scale")

    # Required fields from SimbiBaseConfig
    resolution: tuple[int, int, int] = SimbiField(
        (256, 256, 1), description="Grid resolution"
    )

    bounds: list[tuple[float, float]] = SimbiField(
        [(XMIN, XMAX), (XMIN, XMAX)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.SRMHD, description="Physics regime")

    adiabatic_index: float = SimbiField(5.0 / 3.0, description="Adiabatic index")

    # Optional customizations with non-default values
    solver: Solver = SimbiField(Solver.HLLE, description="Numerical solver")

    boundary_conditions: list[BoundaryCondition] = SimbiField(
        [BoundaryCondition.PERIODIC], description="Boundary conditions"
    )

    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    start_time: float = SimbiField(0.0, description="Simulation start time")

    end_time: float = SimbiField(0.0, description="Simulation end time")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Calculate sound speed parameter
        self._cs = (self.adiabatic_index - 1.0) / self.adiabatic_index

        # Update end_time if needed
        if self.end_time == 0.0:
            self.end_time = (XMAX - XMIN) / self._cs

    @computed_field
    @property
    def cs(self) -> float:
        """Sound speed parameter"""
        return self._cs

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for Orszag-Tang vortex.

        Returns:
            Tuple of generator functions for gas state and B-fields
        """

        def gas_state() -> GasStateGenerator:
            """Generate gas state variables"""
            ni, nj, nk = self.resolution
            xbounds = self.bounds[0]
            ybounds = self.bounds[1]

            # Set up initial conditions
            p0 = float(self.adiabatic_index)
            rho0 = float(self.adiabatic_index) ** 2
            v0 = self.v0

            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj

            for k in range(nk):
                for j in range(nj):
                    y = ybounds[0] + (j + 0.5) * dy  # Cell center
                    for i in range(ni):
                        x = xbounds[0] + (i + 0.5) * dx  # Cell center

                        # Velocity field for vortex
                        vx = -v0 * math.sin(2.0 * math.pi * y)
                        vy = +v0 * math.sin(2.0 * math.pi * x)

                        yield (rho0, vx, vy, 0.0, p0)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            """Generate B-field component values"""
            ni, nj, nk = self.resolution
            xbounds = self.bounds[0]
            ybounds = self.bounds[1]

            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj
            b0 = self.b0

            # Different grid sizes for different components due to staggering
            for k in range(nk + (bn == "bz")):
                for j in range(nj + (bn == "by")):
                    y = ybounds[0] + j * dy
                    for i in range(ni + (bn == "bx")):
                        x = xbounds[0] + i * dx

                        # Different B-field formula for each component
                        if bn == "bx":
                            yield -b0 * math.sin(2.0 * math.pi * y)
                        elif bn == "by":
                            yield +b0 * math.sin(4.0 * math.pi * x)
                        else:
                            yield 0.0

        # Create partial functions for each B-field component
        bx_gen = partial(b_field, "bx")
        by_gen = partial(b_field, "by")
        bz_gen = partial(b_field, "bz")

        # Return tuple of all generator functions
        return (gas_state, bx_gen, by_gen, bz_gen)
