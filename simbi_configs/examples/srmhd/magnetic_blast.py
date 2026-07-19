# =============================================================================
# magnetic_blast.py
#
# cylindrical relativistic magnetized blast wave.
# =============================================================================
import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)

# constants for the blast wave setup
XMIN = -6.0
XMAX = 6.0
P_EXP = 1.0
RHO_EXP = 0.1
R_EXP = 0.08
R_STOP = 1.0


class MagneticBomb(SimbiProblem):
    """cylindrical relativistic magnetized blast wave."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index")
    ]
    rho0: Annotated[
        float, ProblemParam(1.0e-4, cli=True, description="density scale")
    ]
    p0: Annotated[
        float, ProblemParam(3.0e-5, cli=True, description="pressure scale")
    ]
    b0: Annotated[
        float, ProblemParam(0.1, cli=True, description="magnetic field scale")
    ]

    # domain
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((256, 256), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(XMIN, XMAX), (XMIN, XMAX)], description="domain boundaries"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.RMHD, description="physics regime")
    ]

    # numerics
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLE, description="numerical solver")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.OUTFLOW], description="boundary conditions"
        ),
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(
            CellSpacing.LINEAR, description="grid spacing in x1 direction"
        ),
    ]

    # simulation control
    start_time: Annotated[
        float, ProblemParam(0.0, description="simulation start time")
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            4.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for magnetic blast wave."""

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

            for kk in range(nk):
                for jj in range(nj):
                    y = ybounds[0] + (jj + 0.5) * dy
                    for ii in range(ni):
                        x = xbounds[0] + (ii + 0.5) * dx
                        r = math.sqrt(x**2 + y**2)

                        if r < R_EXP:
                            yield (RHO_EXP, 0.0, 0.0, 0.0, P_EXP)
                        elif r > R_EXP and r < R_STOP:
                            yield (
                                RHO_EXP - rhoslope * (r - R_EXP),
                                0.0,
                                0.0,
                                0.0,
                                P_EXP - pslope * (r - R_EXP),
                            )
                        else:
                            yield (rho_amb, 0.0, 0.0, 0.0, pre_amb)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj = self.resolution
            nk = 1

            for kk in range(nk + (bn == "bz")):
                for jj in range(nj + (bn == "by")):
                    for ii in range(ni + (bn == "bx")):
                        if bn == "bx":
                            yield float(self.b0)
                        else:
                            yield 0.0

        bx_gen = partial(b_field, "bx")
        by_gen = partial(b_field, "by")
        bz_gen = partial(b_field, "bz")

        return (gas_state, bx_gen, by_gen, bz_gen)
