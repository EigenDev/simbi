# =============================================================================
# sedov.py
#
# sedov-taylor explosion on a 2d spherical logarithmic mesh.
# variable zones per decade in radius.
# =============================================================================
import math
from typing import Annotated

from pydantic import model_validator

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
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    e0: Annotated[
        float, ProblemParam(1.0, cli=True, description="energy scale")
    ]
    rho0: Annotated[float, ProblemParam(1.0, description="density scale")]
    k: Annotated[
        float,
        ProblemParam(0.0, cli=True, description="density power law exponent"),
    ]

    # domain parameters
    rinit: Annotated[
        float, ProblemParam(0.1, cli=True, description="initial grid radius")
    ]
    rend: Annotated[
        float, ProblemParam(1.0, cli=True, description="radial extent")
    ]
    zpd: Annotated[
        int, ProblemParam(128, cli=True, description="radial zones per decade")
    ]
    full_sphere: Annotated[
        bool,
        ProblemParam(
            False, cli=True, description="flag for full sphere computation"
        ),
    ]

    # domain (computed during init)
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((0, 0), description="grid resolution (calculated)"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(0.0, 0.0), (0.0, 0.0)],
            description="domain boundaries (calculated)",
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.SPHERICAL, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]

    # numerics
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(
            CellSpacing.LOG, description="grid spacing in radial direction"
        ),
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [
                BoundaryCondition.REFLECTING,
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.REFLECTING,
                BoundaryCondition.REFLECTING,
            ],
            description="boundary conditions",
        ),
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="numerical solver")
    ]

    # simulation control
    start_time: Annotated[
        float, ProblemParam(0.0, description="simulation start time")
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            1.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    @model_validator(mode="after")
    def compute_defaults(self) -> "SedovTaylor":
        """compute resolution and bounds based on input parameters."""
        # neither field can default to None (their types are a zone tuple and an
        # extent list), so the zero placeholders — (0, 0) zones, zero-extent
        # bounds — mark "derive from rinit/rend/zpd/full_sphere". a zero zone
        # count or a zero-extent axis is never a runnable grid.
        resolution_unset = self.resolution is None or not all(self.resolution)
        bounds_unset = self.bounds is None or all(lo == hi for lo, hi in self.bounds)
        if bounds_unset:
            theta_max = math.pi if self.full_sphere else 0.5 * math.pi
            self.bounds = [
                (self.rinit, self.rend),
                (0.0, theta_max),
            ]
        if resolution_unset:
            # radial zone count from zones per decade; polar zone count matched
            # to the radial spacing over the (possibly user-set) theta extent.
            ndec = math.log10(self.rend / self.rinit)
            nr = round(self.zpd * ndec)
            npolar = compute_num_polar_zones(
                rmin=float(self.rinit),
                rmax=float(self.rend),
                nr=nr,
                theta_bounds=tuple(self.bounds[1]),
                zpd=int(self.zpd),
            )
            self.resolution = (nr, npolar)

        return self

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
