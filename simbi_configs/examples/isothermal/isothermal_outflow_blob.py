# =============================================================================
# isothermal_outflow_blob.py
#
# isothermal hydro with outflow walls: a gaussian overdensity of amplitude 0.5 and
# width 0.1 drifting at (0.3, 0.2) across the unit square toward the open walls.
# =============================================================================
import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class IsothermalOutflowBlob(SimbiProblem):
    """a drifting isothermal overdensity with outflow walls."""

    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]
    resolution: Annotated[tuple[int, int], ProblemParam((64, 64), cli=True, description="grid resolution")]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="domain boundaries")
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")]
    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.OUTFLOW], description="outflow walls")
    ]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, vx, vy) = (1 + 0.5 exp(-|x - c|^2 / 0.01), 0.3, 0.2), c = (0.5, 0.5)"""

        def gas_state() -> GasStateGenerator:
            ni, nj = self.resolution
            (x0, x1), (y0, y1) = self.bounds
            dx, dy = (x1 - x0) / ni, (y1 - y0) / nj
            for jj in range(nj):
                y = y0 + (jj + 0.5) * dy
                for ii in range(ni):
                    x = x0 + (ii + 0.5) * dx
                    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
                    yield (1.0 + 0.5 * math.exp(-r2 / 0.01), 0.3, 0.2)

        return gas_state
