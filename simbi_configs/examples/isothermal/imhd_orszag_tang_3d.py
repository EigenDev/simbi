# =============================================================================
# imhd_orszag_tang_3d.py
#
# the orszag-tang vortex in isothermal ideal mhd on a three-dimensional periodic
# box: the x-y vortex with a z velocity v_z = 0.1 v0 sin(2 pi z) that carries the
# flow across the z axis, under the planar orszag-tang face field B_x(y), B_y(x),
# B_z = 0, divergence-free on the faces. the isothermal closure p = cs^2 rho drops
# the energy equation. hlld, resolution (nx, ny, nz).
# =============================================================================
import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType, StaggeredBFieldGenerator


class IsothermalOrszagTang3D(SimbiProblem):
    """the orszag-tang vortex in isothermal ideal mhd on a 3d periodic box (hlld)."""

    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]
    v0: Annotated[float, ProblemParam(1.0, cli=True, description="velocity scale")]
    b0: Annotated[
        float,
        ProblemParam(1.0 / math.sqrt(4.0 * math.pi), cli=True, description="magnetic field scale"),
    ]
    resolution: Annotated[
        tuple[int, int, int], ProblemParam((32, 32, 32), cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.IMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLD, description="numerical solver")]
    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.PERIODIC], description="boundary conditions")
    ]
    end_time: Annotated[float, ProblemParam(0.5, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """unit density; the vortex with v_z = 0.1 v0 sin(2 pi z); no pressure slot"""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            (x0, x1), (y0, y1), (z0, z1) = self.bounds
            dx, dy, dz = (x1 - x0) / ni, (y1 - y0) / nj, (z1 - z0) / nk
            v0 = self.v0
            for kk in range(nk):
                z = z0 + (kk + 0.5) * dz
                for jj in range(nj):
                    y = y0 + (jj + 0.5) * dy
                    for ii in range(ni):
                        x = x0 + (ii + 0.5) * dx
                        yield (
                            1.0,
                            -v0 * math.sin(2.0 * math.pi * y),
                            v0 * math.sin(2.0 * math.pi * x),
                            0.1 * v0 * math.sin(2.0 * math.pi * z),
                        )

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj, nk = self.resolution
            (x0, x1), (y0, y1), _ = self.bounds
            dx, dy = (x1 - x0) / ni, (y1 - y0) / nj
            b0 = self.b0
            for _kk in range(nk + (bn == "bz")):
                for jj in range(nj + (bn == "by")):
                    for ii in range(ni + (bn == "bx")):
                        if bn == "bx":
                            yield -b0 * math.sin(2.0 * math.pi * (y0 + (jj + 0.5) * dy))
                        elif bn == "by":
                            yield b0 * math.sin(4.0 * math.pi * (x0 + (ii + 0.5) * dx))
                        else:
                            yield 0.0

        return gas_state, partial(b_field, "bx"), partial(b_field, "by"), partial(b_field, "bz")
