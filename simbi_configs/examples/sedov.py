import math
from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import (
    BoundaryCondition,
    CoordSystem,
    Regime,
    CellSpacing,
    Solver,
)
from simbi.core.types.typing import GasStateGenerator, InitialStateType
from simbi import compute_num_polar_zones
from typing import Any

# Constants
RHO_AMB = 1.0
T_AMB = 1e-10
NU = 3.0


class SedovTaylor(SimbiBaseConfig):
    """The Sedov Taylor Problem
    Sedov-Taylor Explosion on a 2D Spherical logarithmic mesh with variable zones per decade in radius
    """

    # Configuration parameters
    e0: float = SimbiField(1.0, description="Energy scale")
    rho0: float = SimbiField(1.0, description="Density scale")
    rinit: float = SimbiField(0.1, description="Initial grid radius")
    rend: float = SimbiField(1.0, description="Radial extent")
    k: float = SimbiField(0.0, description="Density power law exponent")
    zpd: int = SimbiField(1024, description="Number of radial zones per decade")
    full_sphere: bool = SimbiField(
        False, description="Flag for full sphere computation"
    )

    # Required fields from SimbiBaseConfig - will be set during __init__
    resolution: tuple[int, int] = SimbiField(
        (0, 0), description="Grid resolution (calculated)"
    )

    bounds: list[tuple[float, float]] = SimbiField(
        [(0.0, 0.0), (0.0, 0.0)], description="Domain boundaries (calculated)"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.SPHERICAL, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.CLASSICAL, description="Physics regime")

    adiabatic_index: float = SimbiField(5.0 / 3.0, description="Adiabatic index")

    # Optional customizations
    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LOG, description="Grid spacing in radial direction"
    )

    boundary_conditions: list[BoundaryCondition] = SimbiField(
        [
            BoundaryCondition.REFLECTING,
            BoundaryCondition.OUTFLOW,
            BoundaryCondition.REFLECTING,
            BoundaryCondition.REFLECTING,
        ],
        description="Boundary conditions",
    )

    solver: Solver = SimbiField(Solver.HLLC, description="Numerical solver")

    default_start_time: float = SimbiField(0.0, description="Simulation start time")

    default_end_time: float = SimbiField(1.0, description="Simulation end time")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Calculate number of radial zones based on zones per decade
        ndec = math.log10(self.rend / self.rinit)
        self._nr = round(self.zpd * ndec)

        # Set theta boundaries based on full_sphere flag
        self._theta_min = 0
        self._theta_max = math.pi if self.full_sphere else 0.5 * math.pi

        # Calculate number of polar zones
        self._npolar = compute_num_polar_zones(
            rmin=float(self.rinit),
            rmax=float(self.rend),
            nr=self._nr,
            theta_bounds=(self._theta_min, self._theta_max),
            zpd=int(self.zpd),
        )

        # Update resolution and bounds fields
        self.resolution = (self._nr, self._npolar)
        self.bounds = [(self.rinit, self.rend), (self._theta_min, self._theta_max)]

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for Sedov-Taylor explosion.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            nr, npolar = self.resolution
            explosion_radius = self.rinit * 1.5
            dlogr = math.log10(self.rend / self.rinit) / nr

            for j in range(npolar):
                for i in range(nr):
                    # Logarithmic radial grid
                    r = self.rinit * 10 ** (i * dlogr)

                    # Density with power law profile
                    rho = RHO_AMB * r ** (-self.k)

                    # Pressure inside vs. outside the explosion region
                    if r <= explosion_radius:
                        # Energy deposition inside explosion radius
                        pre = (self.adiabatic_index - 1.0) * (
                            3.0 * self.e0 / (NU + 1) / math.pi / explosion_radius**NU
                        )
                    else:
                        # Ambient conditions outside explosion radius
                        pre = T_AMB * rho

                    yield (rho, 0.0, 0.0, pre)

        return gas_state
