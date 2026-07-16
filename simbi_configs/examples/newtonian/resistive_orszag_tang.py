# =============================================================================
# resistive_orszag_tang.py
#
# the orszag-tang vortex (Orszag & Tang 1979) in NON-IDEAL newtonian mhd: the
# ideal setup plus a uniform ohmic resistivity eta, so the induction equation
# carries a diffusive emf eta*J. resistivity is the physical control on the
# central current sheet: as the two field null-lines are driven together the
# sheet thins to the grid scale, and eta sets the reconnection rate + caps the
# peak current density instead of letting numerical dissipation decide it.
#
# the diagnostic is the magnetic energy: ideal mhd (eta=0) conserves it up to the
# scheme's numerical diffusion; eta>0 makes it decay monotonically as the sheets
# reconnect, and that lost magnetic energy reappears as gas internal energy (the
# ohmic heating is exact + conservative in the constrained-transport step). run a
# small eta scan and the peak |J_z| and the field-energy decay rate track eta.
#
# genuine 2.5d (spatial d=2, vector dof=3): resolution (nx, ny, 1). hlld solver.
#
# usage:
#  simbi run resistive_orszag_tang.py --eta 0.0    # ideal reference
#  simbi run resistive_orszag_tang.py --eta 1.0e-3 # sheets smoothed, field decays
# =============================================================================
import math
from functools import partial
from typing import Annotated

from pydantic import computed_field

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

XMIN = 0.0
XMAX = 1.0


class ResistiveOrszagTang(SimbiProblem):
    """the orszag-tang vortex in resistive newtonian mhd (hlld + eta*J emf)."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    v0: Annotated[float, ProblemParam(1.0, cli=True, description="velocity scale")]
    b0: Annotated[
        float,
        ProblemParam(
            1.0 / math.sqrt(4.0 * math.pi),
            cli=True,
            description="magnetic field scale",
        ),
    ]
    eta: Annotated[
        float,
        ProblemParam(
            1.0e-3,
            cli=True,
            description="ohmic resistivity (diffusive emf eta*J). 0 = ideal mhd; "
            "the magnetic reynolds number is Rm = v0 L / eta = 1/eta here (L=v0=1)",
        ),
    ]

    # domain (128^2 keeps the vortex cheap on cpu; the sheet is resolved by t~0.5)
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((128, 128, 1), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(XMIN, XMAX), (XMIN, XMAX)], description="domain boundaries"),
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
        ProblemParam(CellSpacing.LINEAR, description="grid spacing in x1"),
    ]

    # control
    start_time: Annotated[
        float, ProblemParam(0.0, description="simulation start time")
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            1.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    @computed_field
    @property
    def resistivity(self) -> float:
        # the backend reads `resistivity`; expose it as the cli eta.
        return self.eta

    def initial_primitive_state(self) -> InitialStateType:
        """canonical newtonian orszag-tang: rho = 25/(36 pi), p = 5/(12 pi) so
        cs^2 = gamma p / rho = 1, with the v0-scaled vortex and the div-free
        staggered field. identical ideal ic to nmhd_orszag_tang.py; the only
        difference is the non-ideal emf that the resistivity switches on."""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            xbounds, ybounds = self.bounds[0], self.bounds[1]
            rho0 = 25.0 / (36.0 * math.pi)
            p0 = 5.0 / (12.0 * math.pi)
            v0 = self.v0
            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj

            for _kk in range(nk):
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

            for _kk in range(nk + (bn == "bz")):
                for jj in range(nj + (bn == "by")):
                    for ii in range(ni + (bn == "bx")):
                        if bn == "bx":
                            # bx on the x-face: transverse y at the cell center so
                            # the discrete field is symmetric about the domain center.
                            y = ybounds[0] + (jj + 0.5) * dy
                            yield -b0 * math.sin(2.0 * math.pi * y)
                        elif bn == "by":
                            # by on the y-face: transverse x at the cell center.
                            x = xbounds[0] + (ii + 0.5) * dx
                            yield +b0 * math.sin(4.0 * math.pi * x)
                        else:
                            yield 0.0

        return (
            gas_state,
            partial(b_field, "bx"),
            partial(b_field, "by"),
            partial(b_field, "bz"),
        )
