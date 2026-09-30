# =============================================================================
# isothermal_channel.py
#
# isothermal hydro in a channel: periodic along x, reflecting walls at y = 0 and
# y = 1. a shear v_x = 0.3 sin(2 pi y) with a cross flow
# v_y = 0.05 sin(2 pi x) sin(pi y), which vanishes on the walls, over a density
# wave rho = 1 + 0.1 cos(2 pi x).
# =============================================================================
import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class IsothermalChannel(SimbiProblem):
    """an isothermal shear flow between reflecting walls, periodic along the channel."""

    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]
    resolution: Annotated[tuple[int, int], ProblemParam((64, 64), cli=True, description="grid resolution")]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="domain boundaries")
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.PERIODIC, BoundaryCondition.REFLECTING], description="x periodic, y walls"),
    ]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, vx, vy) of the density wave, the shear and the wall-bounded cross flow"""

        def gas_state() -> GasStateGenerator:
            ni, nj = self.resolution
            (x0, x1), (y0, y1) = self.bounds
            dx, dy = (x1 - x0) / ni, (y1 - y0) / nj
            for jj in range(nj):
                y = y0 + (jj + 0.5) * dy
                for ii in range(ni):
                    x = x0 + (ii + 0.5) * dx
                    yield (
                        1.0 + 0.1 * math.cos(2.0 * math.pi * x),
                        0.3 * math.sin(2.0 * math.pi * y),
                        0.05 * math.sin(2.0 * math.pi * x) * math.sin(math.pi * y),
                    )

        return gas_state
