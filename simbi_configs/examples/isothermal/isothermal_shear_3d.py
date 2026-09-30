# =============================================================================
# isothermal_shear_3d.py
#
# isothermal hydro on a three-dimensional periodic box: a density wave
# rho = 1 + 0.2 sin(2 pi x) carried by a shear flow along every axis,
# v = 0.3 (sin(2 pi y), sin(2 pi z), sin(2 pi x)), so each axis carries flux
# across the other two. the closure p = cs^2 rho drops the energy equation.
# =============================================================================
import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class IsothermalShear3D(SimbiProblem):
    """a density wave in a three-axis shear flow, isothermal, on a 3d periodic box."""

    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]
    resolution: Annotated[
        tuple[int, int, int], ProblemParam((32, 32, 32), cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")]
    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.PERIODIC], description="boundary conditions")
    ]
    end_time: Annotated[float, ProblemParam(0.5, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, vx, vy, vz) of the wave and the three-axis shear"""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            (x0, x1), (y0, y1), (z0, z1) = self.bounds
            dx, dy, dz = (x1 - x0) / ni, (y1 - y0) / nj, (z1 - z0) / nk
            for kk in range(nk):
                z = z0 + (kk + 0.5) * dz
                for jj in range(nj):
                    y = y0 + (jj + 0.5) * dy
                    for ii in range(ni):
                        x = x0 + (ii + 0.5) * dx
                        yield (
                            1.0 + 0.2 * math.sin(2.0 * math.pi * x),
                            0.3 * math.sin(2.0 * math.pi * y),
                            0.3 * math.sin(2.0 * math.pi * z),
                            0.3 * math.sin(2.0 * math.pi * x),
                        )

        return gas_state
