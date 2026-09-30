# =============================================================================
# isothermal_gradient_box.py
#
# a dense pulse spinning inside an isothermal box whose walls prescribe
# gradients: a neumann wall on the x faces holds an outward normal derivative
# of the density and the velocity, and a robin wall on the y faces holds
# a*U + b*dU/dn = c for each. the closure p = cs^2 rho fixes the pressure at
# the ghost from the density, so the walls carry no pressure coefficient of
# their own. the pulse reaches every wall within a sound-crossing time.
# =============================================================================
import math
from typing import Annotated

from simbi import Neumann, ProblemParam, Robin, SimbiProblem
from simbi.types import CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class IsothermalGradientBox(SimbiProblem):
    """a spinning isothermal pulse in a box with neumann and robin walls."""

    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]
    spin: Annotated[float, ProblemParam(0.3, cli=True, description="solid-body rotation rate")]
    resolution: Annotated[tuple[int, int], ProblemParam((64, 64), cli=True, description="grid resolution")]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="domain boundaries")
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")]
    boundary_conditions: Annotated[
        list,
        ProblemParam(
            [
                Neumann(rho=0.2, velocity=(0.0, 0.1)),
                Robin(rho=(1.0, 0.2, 1.1), velocity=((1.0, 0.0, 0.0), (1.0, 0.1, 0.0))),
            ],
            description="boundary conditions [x, y]",
        ),
    ]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, vx, vy) = (1 + 0.5 g, -spin (y - 1/2), spin (x - 1/2)), g a gaussian at (0.4, 0.6)"""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            (xmin, xmax), (ymin, ymax) = self.bounds
            dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
            for jj in range(ny):
                y = ymin + (jj + 0.5) * dy
                for ii in range(nx):
                    x = xmin + (ii + 0.5) * dx
                    g = math.exp(-((x - 0.4) ** 2 + (y - 0.6) ** 2) / 0.02)
                    yield (1.0 + 0.5 * g, -self.spin * (y - 0.5), self.spin * (x - 0.5))

        return gas_state
