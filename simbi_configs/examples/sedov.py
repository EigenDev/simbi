# =============================================================================
# sedov.py
#
# sedov-taylor explosion on a 2d spherical logarithmic mesh.
# variable zones per decade in radius.
# =============================================================================
import math
from typing import Any

from simbi import ProblemParam, SimbiProblem, compute_num_polar_zones
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.types.typing import GasStateGenerator, InitialStateType

# constants
RHO_AMB = 1.0
T_AMB = 1e-10
NU = 3.0


class SedovTaylor(SimbiProblem):
    """sedov-taylor explosion on 2d spherical logarithmic mesh."""

    # physics
    adiabatic_index: float = ProblemParam(
        5.0 / 3.0, description="adiabatic index"
    )
    e0: float = ProblemParam(1.0, cli=True, description="energy scale")
    rho0: float = ProblemParam(1.0, description="density scale")
    k: float = ProblemParam(
        0.0, cli=True, description="density power law exponent"
    )

    # domain parameters
    rinit: float = ProblemParam(
        0.1, cli=True, description="initial grid radius"
    )
    rend: float = ProblemParam(1.0, cli=True, description="radial extent")
    zpd: int = ProblemParam(
        128, cli=True, description="radial zones per decade"
    )
    full_sphere: bool = ProblemParam(
        False, cli=True, description="flag for full sphere computation"
    )

    # domain (computed during init)
    resolution: tuple[int, int] = ProblemParam(
        (0, 0), description="grid resolution (calculated)"
    )
    bounds: list[tuple[float, float]] = ProblemParam(
        [(0.0, 0.0), (0.0, 0.0)], description="domain boundaries (calculated)"
    )
    coord_system: CoordSystem = ProblemParam(
        CoordSystem.SPHERICAL, description="coordinate system"
    )
    regime: Regime = ProblemParam(
        Regime.NEWTONIAN, description="physics regime"
    )

    # numerics
    x1_spacing: CellSpacing = ProblemParam(
        CellSpacing.LOG, description="grid spacing in radial direction"
    )
    boundary_conditions: list[BoundaryCondition] = ProblemParam(
        [
            BoundaryCondition.REFLECTING,
            BoundaryCondition.OUTFLOW,
            BoundaryCondition.REFLECTING,
            BoundaryCondition.REFLECTING,
        ],
        description="boundary conditions",
    )
    solver: Solver = ProblemParam(Solver.HLLC, description="numerical solver")

    # simulation control
    start_time: float = ProblemParam(0.0, description="simulation start time")
    end_time: float = ProblemParam(
        1.0, cli=True, checkpoint_safe=True, description="simulation end time"
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # calculate number of radial zones based on zones per decade
        ndec = math.log10(self.rend / self.rinit)
        nr = round(self.zpd * ndec)

        # set theta boundaries based on full_sphere flag
        theta_min = 0
        theta_max = math.pi if self.full_sphere else 0.5 * math.pi

        # calculate number of polar zones
        npolar = compute_num_polar_zones(
            rmin=float(self.rinit),
            rmax=float(self.rend),
            nr=nr,
            theta_bounds=(theta_min, theta_max),
            zpd=int(self.zpd),
        )

        # update resolution and bounds fields
        object.__setattr__(self, "resolution", (nr, npolar))
        object.__setattr__(
            self,
            "bounds",
            [
                (self.rinit, self.rend),
                (theta_min, theta_max),
            ],
        )

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for sedov-taylor explosion."""

        def gas_state() -> GasStateGenerator:
            nr, npolar = self.resolution
            explosion_radius = self.rinit * 1.5
            dlogr = math.log10(self.rend / self.rinit) / nr

            for jj in range(npolar):
                for ii in range(nr):
                    r = self.rinit * 10 ** (ii * dlogr)
                    rho = RHO_AMB * r ** (-self.k)

                    if r <= explosion_radius:
                        pre = (self.adiabatic_index - 1.0) * (
                            3.0
                            * self.e0
                            / (NU + 1)
                            / math.pi
                            / explosion_radius**NU
                        )
                    else:
                        pre = T_AMB * rho

                    yield (rho, 0.0, 0.0, pre)

        return gas_state
