# =============================================================================
# imhd_orszag_tang.py
#
# the orszag-tang vortex in ISOTHERMAL ideal mhd (Mignone 2007), HLLD by default.
# the isothermal closure p = cs^2 rho drops the energy equation: the primitive is
# (rho, vx, vy, vz) with NO pressure slot. genuine 2.5d (spatial d=2, dof=3).
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

# domain constants
XMIN = 0.0
XMAX = 1.0


class IsothermalOrszagTang(SimbiProblem):
    """the orszag-tang vortex in isothermal ideal mhd (hlld)."""

    # physics — isothermal: sound speed cs replaces the adiabatic index. gamma
    # is unused by the isothermal eos (p = cs^2 rho) but the base model requires
    # the field; 1.0 keeps the isothermal classification consistent.
    sound_speed: Annotated[
        float, ProblemParam(1.0, cli=True, description="isothermal sound speed")
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
    rho0: Annotated[
        float, ProblemParam(1.0, description="ambient density")
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
        Regime, ProblemParam(Regime.IMHD, description="physics regime")
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
        """isothermal orszag-tang: the primitive has no pressure slot."""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            xbounds, ybounds = self.bounds[0], self.bounds[1]
            rho0 = self.rho0
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
                        # isothermal primitive: (rho, vx, vy, vz) — no pressure.
                        yield (rho0, vx, vy, 0.0)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj, nk = self.resolution
            xbounds, ybounds = self.bounds[0], self.bounds[1]
            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj
            b0 = self.b0

            for kk in range(nk + (bn == "bz")):
                for jj in range(nj + (bn == "by")):
                    for ii in range(ni + (bn == "bx")):
                        if bn == "bx":
                            # bx lives on the x-face: sample the TRANSVERSE y at
                            # the cell center (jj+0.5), so the discrete field is
                            # symmetric about the domain center. sampling y at the
                            # edge (jj*dy) breaks the OT 180-degree point symmetry
                            # and misaligns b against the cell-centered velocity.
                            y = ybounds[0] + (jj + 0.5) * dy
                            yield -b0 * math.sin(2.0 * math.pi * y)
                        elif bn == "by":
                            # by lives on the y-face: transverse x at the cell center.
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
