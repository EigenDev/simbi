# =============================================================================
# nmhd_orszag_tang.py
#
# the orszag-tang vortex in NEWTONIAN ideal mhd, run with the 5-wave HLLD solver.
# the algebraic nmhd c2p cannot fail in the current sheets the way the relativistic
# inversion can, so it is the robustness payoff. genuine 2.5d (spatial d=2, vector
# dof=3): resolution (nx, ny, 1).
# =============================================================================
import math
from functools import partial
from typing import Annotated

from pydantic import model_validator

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

# domain constants
XMIN = 0.0
XMAX = 1.0


class NewtonianOrszagTang(SimbiProblem):
    """the orszag-tang vortex in newtonian ideal mhd (hlld)."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    v0: Annotated[
        float, ProblemParam(1.0, cli=True, description="velocity scale")
    ]
    b0: Annotated[
        float,
        ProblemParam(
            1.0 / math.sqrt(4.0 * math.pi),
            cli=True,
            description="magnetic field scale",
        ),
    ]

    # domain
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((256, 256, 1), cli=True, description="grid resolution"),
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
        Regime, ProblemParam(Regime.NMHD, description="physics regime")
    ]

    # numerics
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLD, description="numerical solver")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.PERIODIC], description="boundary conditions"
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
            1.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """canonical orszag-tang: rho=gamma^2, p=gamma, v0-scaled vortex."""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            xbounds, ybounds = self.bounds[0], self.bounds[1]
            p0 = float(self.adiabatic_index)
            rho0 = float(self.adiabatic_index) ** 2
            v0 = self.v0
            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj

            for kk in range(nk):
                for jj in range(nj):
                    y = ybounds[0] + (jj + 0.5) * dy
                    for ii in range(ni):
                        x = xbounds[0] + (ii + 0.5) * dx
                        vx = -v0 * math.sin(2.0 * math.pi * y)
                        vy = +v0 * math.sin(2.0 * math.pi * x)
                        yield (rho0, vx, vy, 0.0, p0)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj, nk = self.resolution
            xbounds, ybounds = self.bounds[0], self.bounds[1]
            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj
            b0 = self.b0

            for kk in range(nk + (bn == "bz")):
                for jj in range(nj + (bn == "by")):
                    y = ybounds[0] + jj * dy
                    for ii in range(ni + (bn == "bx")):
                        x = xbounds[0] + ii * dx
                        if bn == "bx":
                            yield -b0 * math.sin(2.0 * math.pi * y)
                        elif bn == "by":
                            yield +b0 * math.sin(4.0 * math.pi * x)
                        else:
                            yield 0.0

        return (
            gas_state,
            partial(b_field, "bx"),
            partial(b_field, "by"),
            partial(b_field, "bz"),
        )
