# =============================================================================
# synge_blast.py
#
# a relativistic blast under the synge (taub-mathews) perfect gas: a hot disk of
# radius 0.1 at pressure 10 in a cold medium at pressure 0.01, both at unit rest
# density and at rest, on the unit square with outflow walls. the closure is
# parameter-free: the effective adiabatic index walks from 5/3 in the cold medium
# toward 4/3 in the hot core, so the blast carries both limits at once.
# =============================================================================
import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.input import Eos
from simbi.types.typing import GasStateGenerator, InitialStateType


class SyngeBlast(SimbiProblem):
    """a hot disk in a cold medium under the synge perfect gas."""

    resolution: Annotated[
        tuple[int, int],
        ProblemParam((128, 128), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    eos: Annotated[
        Eos, ProblemParam(Eos.SYNGE, cli=True, description="the synge perfect gas")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.OUTFLOW], description="boundary conditions"),
    ]
    end_time: Annotated[
        float,
        ProblemParam(0.2, cli=True, checkpoint_safe=True, description="simulation end time"),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """unit density at rest; pressure 10 inside the central disk of radius 0.1, 0.01 outside"""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            (x0, x1), (y0, y1) = self.bounds
            dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
            for jj in range(ny):
                y = y0 + (jj + 0.5) * dy
                for ii in range(nx):
                    x = x0 + (ii + 0.5) * dx
                    hot = math.hypot(x - 0.5, y - 0.5) < 0.1
                    yield (1.0, 0.0, 0.0, 10.0 if hot else 0.01)

        return gas_state
