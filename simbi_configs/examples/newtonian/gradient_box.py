# =============================================================================
# gradient_box.py
#
# a dense, over-pressured pulse spinning inside a closed box whose walls
# prescribe gradients: a neumann wall on the x faces holds an outward normal
# derivative per primitive, and a robin wall on the y faces holds
# a*U + b*dU/dn = c per primitive. the pulse reaches every wall within a
# sound-crossing time, so the walls shape the flow from the first steps.
#
# usage:
#   simbi run simbi_configs/examples/newtonian/gradient_box.py --resolution 64,64
# =============================================================================

import math
from typing import Annotated

from simbi import Neumann, ProblemParam, Robin, SimbiProblem
from simbi.types import CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class GradientBox(SimbiProblem):
    """a spinning pulse in a box with neumann and robin walls."""

    resolution: Annotated[
        tuple[int, int],
        ProblemParam((64, 64), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]
    adiabatic_index: Annotated[
        float, ProblemParam(1.4, description="adiabatic index")
    ]
    boundary_conditions: Annotated[
        list,
        ProblemParam(
            [
                Neumann(rho=0.2, velocity=(0.0, 0.1), pressure=-0.3),
                Robin(
                    rho=(1.0, 0.2, 1.1),
                    velocity=((1.0, 0.0, 0.0), (1.0, 0.1, 0.0)),
                    pressure=(1.0, 0.3, 1.2),
                ),
            ],
            description="boundary conditions [x, y]",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            1.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]
    spin: Annotated[
        float, ProblemParam(0.3, cli=True, description="solid-body rotation rate")
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """a gaussian pulse off the box center in a solid-body rotation about the
        center. the primitive is (rho, vx, vy, p)."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            (xmin, xmax), (ymin, ymax) = self.bounds
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            for jj in range(ny):
                y = ymin + (jj + 0.5) * dy
                for ii in range(nx):
                    x = xmin + (ii + 0.5) * dx
                    g = math.exp(-((x - 0.4) ** 2 + (y - 0.6) ** 2) / 0.02)
                    vx = -self.spin * (y - 0.5)
                    vy = self.spin * (x - 0.5)
                    yield (1.0 + 0.5 * g, vx, vy, 1.0 + g)

        return gas_state
