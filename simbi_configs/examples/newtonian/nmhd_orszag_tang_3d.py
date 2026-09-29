# =============================================================================
# nmhd_orszag_tang_3d.py
#
# the newtonian orszag-tang vortex on a three-dimensional periodic box: the
# canonical x-y vortex with a z velocity v_z = 0.1 v0 sin(2 pi z) that carries the
# flow across the z axis. the staggered field is the planar orszag-tang field,
# B_x(y), B_y(x), B_z = 0, divergence-free on the faces. hlld gas, resolution
# (nx, ny, nz).
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


class NewtonianOrszagTang3D(SimbiProblem):
    """the orszag-tang vortex in newtonian ideal mhd on a 3d periodic box (hlld)."""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="adiabatic index")]
    v0: Annotated[float, ProblemParam(1.0, cli=True, description="velocity scale")]
    b0: Annotated[
        float,
        ProblemParam(1.0 / math.sqrt(4.0 * math.pi), cli=True, description="magnetic field scale"),
    ]
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((32, 32, 32), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.NMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLD, description="numerical solver")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.PERIODIC], description="boundary conditions"),
    ]
    end_time: Annotated[
        float,
        ProblemParam(0.5, cli=True, checkpoint_safe=True, description="simulation end time"),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """the athena newtonian orszag-tang state, rho = 25/(36 pi), p = 5/(12 pi), with the
        z velocity v_z = 0.1 v0 sin(2 pi z)"""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            (x0, x1), (y0, y1), (z0, z1) = self.bounds
            dx, dy, dz = (x1 - x0) / ni, (y1 - y0) / nj, (z1 - z0) / nk
            rho0 = 25.0 / (36.0 * math.pi)
            p0 = 5.0 / (12.0 * math.pi)
            v0 = self.v0
            for kk in range(nk):
                z = z0 + (kk + 0.5) * dz
                for jj in range(nj):
                    y = y0 + (jj + 0.5) * dy
                    for ii in range(ni):
                        x = x0 + (ii + 0.5) * dx
                        vx = -v0 * math.sin(2.0 * math.pi * y)
                        vy = +v0 * math.sin(2.0 * math.pi * x)
                        vz = 0.1 * v0 * math.sin(2.0 * math.pi * z)
                        yield (rho0, vx, vy, vz, p0)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj, nk = self.resolution
            (x0, x1), (y0, y1), _ = self.bounds
            dx, dy = (x1 - x0) / ni, (y1 - y0) / nj
            b0 = self.b0
            for _kk in range(nk + (bn == "bz")):
                for jj in range(nj + (bn == "by")):
                    for ii in range(ni + (bn == "bx")):
                        if bn == "bx":
                            # the x-face field sampled at the cell-centered y of its face
                            yield -b0 * math.sin(2.0 * math.pi * (y0 + (jj + 0.5) * dy))
                        elif bn == "by":
                            yield +b0 * math.sin(4.0 * math.pi * (x0 + (ii + 0.5) * dx))
                        else:
                            yield 0.0

        return (
            gas_state,
            partial(b_field, "bx"),
            partial(b_field, "by"),
            partial(b_field, "bz"),
        )
