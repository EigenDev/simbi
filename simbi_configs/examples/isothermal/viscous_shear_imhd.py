# =============================================================================
# viscous_shear_imhd.py
#
# a viscous shear layer in isothermal ideal mhd on a periodic square, 2.5D: the
# transverse shear v_y = V0 sin(2 pi x) with a cross shear v_x = V0/2 sin(2 pi y),
# uniform density, under a uniform face field B_x = B0 (divergence-free) and a
# constant kinematic viscosity nu. the closure p = cs^2 rho drops the energy
# equation; the field couples the shear to alfven waves the viscosity damps.
# =============================================================================
import math
from functools import partial
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType, StaggeredBFieldGenerator


class ViscousShearImhd(SimbiProblem):
    """a viscous shear layer with a cross shear under a uniform field, isothermal mhd."""

    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]
    v0: Annotated[float, ProblemParam(0.1, cli=True, description="shear amplitude")]
    b0: Annotated[float, ProblemParam(0.1, cli=True, description="uniform B_x")]
    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]
    resolution: Annotated[
        tuple[int, int, int], ProblemParam((64, 64, 1), cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="domain boundaries")
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.IMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLD, description="numerical solver")]
    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.PERIODIC], description="boundary conditions")
    ]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="simulation end time")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, vx, vy, vz) = (1, V0/2 sin(2 pi y), V0 sin(2 pi x), 0) under B_x = B0"""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            (x0, x1), (y0, y1) = self.bounds
            dx, dy = (x1 - x0) / ni, (y1 - y0) / nj
            for _kk in range(nk):
                for jj in range(nj):
                    y = y0 + (jj + 0.5) * dy
                    for ii in range(ni):
                        x = x0 + (ii + 0.5) * dx
                        yield (1.0, 0.5 * self.v0 * math.sin(2.0 * math.pi * y), self.v0 * math.sin(2.0 * math.pi * x), 0.0)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj, nk = self.resolution
            for _kk in range(nk + (bn == "bz")):
                for _jj in range(nj + (bn == "by")):
                    for _ii in range(ni + (bn == "bx")):
                        yield self.b0 if bn == "bx" else 0.0

        return gas_state, partial(b_field, "bx"), partial(b_field, "by"), partial(b_field, "bz")
