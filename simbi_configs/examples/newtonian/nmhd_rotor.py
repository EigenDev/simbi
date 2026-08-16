# =============================================================================
# nmhd_rotor.py
#
# the magnetized rotor test (Balsara & Spicer 1999; Toth 2000 test 1) in newtonian
# ideal mhd, HLLD by default. a dense disk spins inside a uniform Bx field; the
# rotation winds the field into torsional alfven waves — a low-beta robustness +
# div(B) stress test. genuine 2.5d (spatial d=2, vector dof=3).
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

# domain + rotor constants (Toth test 1)
XMIN = 0.0
XMAX = 1.0
XC = 0.5
R0 = 0.1
R1 = 0.115
V0 = 2.0


def _rotor_state(x: float, y: float) -> tuple[float, float, float]:
    """(rho, vx, vy) at (x, y); p=1, vz=0 elsewhere."""
    dx, dy = x - XC, y - XC
    r = math.sqrt(dx * dx + dy * dy)
    if r < R0:
        return (10.0, -V0 * dy / R0, V0 * dx / R0)
    if r < R1:
        f = (R1 - r) / (R1 - R0)
        return (1.0 + 9.0 * f, -f * V0 * dy / r, f * V0 * dx / r)
    return (1.0, 0.0, 0.0)


class NewtonianRotor(SimbiProblem):
    """the magnetized rotor in newtonian ideal mhd (hlld)."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(1.4, description="adiabatic index")
    ]
    p0: Annotated[float, ProblemParam(1.0, description="ambient pressure")]
    b0: Annotated[
        float,
        ProblemParam(
            5.0 / math.sqrt(4.0 * math.pi),
            cli=True,
            description="uniform Bx field strength",
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
            0.15,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """spinning disk in a uniform Bx field."""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            xbounds, ybounds = self.bounds[0], self.bounds[1]
            p0 = self.p0
            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj

            for kk in range(nk):
                for jj in range(nj):
                    y = ybounds[0] + (jj + 0.5) * dy
                    for ii in range(ni):
                        x = xbounds[0] + (ii + 0.5) * dx
                        rho, vx, vy = _rotor_state(x, y)
                        yield (rho, vx, vy, 0.0, p0)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj, nk = self.resolution
            b0 = self.b0
            # uniform Bx on the x-faces; By, Bz are zero.
            for kk in range(nk + (bn == "bz")):
                for jj in range(nj + (bn == "by")):
                    for ii in range(ni + (bn == "bx")):
                        yield b0 if bn == "bx" else 0.0

        return (
            gas_state,
            partial(b_field, "bx"),
            partial(b_field, "by"),
            partial(b_field, "bz"),
        )
