# =============================================================================
# refined_mhd_3d.py
#
# a 3d MHD run on a STATICALLY REFINED mesh — the orszag-tang vortex extruded
# along z (uniform in z), with one fine box covering the central vortex. the
# point is the constrained-transport invariant: the fine staggered B is seeded
# by DIVERGENCE-FREE prolongation of the coarse faces, so div(B)=0 holds on BOTH
# levels and across the coarse-fine boundary. mhd refinement is 3d-cartesian only.
# =============================================================================
import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)


class RefinedMhd3D(SimbiProblem):
    """3d (z-extruded) orszag-tang vortex with a refined central box."""

    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    v0: Annotated[float, ProblemParam(0.5, description="velocity scale")]
    b0: Annotated[float, ProblemParam(1.0, description="magnetic field scale")]

    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((16, 16, 16), cli=True, description="coarse (nx,ny,nz)"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="domain bounds"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.RMHD, description="physics regime")
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLD, description="numerical solver")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.PERIODIC], description="boundary conditions"
        ),
    ]

    # ---- one fine box over the central vortex (3d cartesian) ----
    refinement_enabled: Annotated[
        bool, ProblemParam(True, description="enable mesh refinement")
    ]
    refinement_max_levels: Annotated[
        int, ProblemParam(2, description="coarse + 1 fine")
    ]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam(
            [[0.25, 0.75, 0.25, 0.75, 0.0, 1.0]],
            description="x,y core [0.25,0.75]; full z",
        ),
    ]
    refinement_ratios: Annotated[
        list[int], ProblemParam([2], description="coarse->fine ratio")
    ]

    end_time: Annotated[
        float,
        ProblemParam(0.05, cli=True, checkpoint_safe=True, description="end"),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """orszag-tang in x-y, uniform along z; div-free staggered B."""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            (x0, x1), (y0, y1), (z0, z1) = self.bounds
            dx, dy = (x1 - x0) / ni, (y1 - y0) / nj
            p0 = float(self.adiabatic_index)
            rho0 = float(self.adiabatic_index) ** 2
            v0 = self.v0
            for _kk in range(nk):
                for jj in range(nj):
                    y = y0 + (jj + 0.5) * dy
                    for ii in range(ni):
                        x = x0 + (ii + 0.5) * dx
                        vx = -v0 * math.sin(2.0 * math.pi * y)
                        vy = +v0 * math.sin(2.0 * math.pi * x)
                        yield (rho0, vx, vy, 0.0, p0)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj, nk = self.resolution
            (x0, x1), (y0, y1), (z0, z1) = self.bounds
            dx, dy = (x1 - x0) / ni, (y1 - y0) / nj
            b0 = self.b0
            for _kk in range(nk + (bn == "bz")):
                for jj in range(nj + (bn == "by")):
                    y = y0 + jj * dy
                    for ii in range(ni + (bn == "bx")):
                        x = x0 + ii * dx
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
