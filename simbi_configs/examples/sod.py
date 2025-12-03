# =============================================================================
# sod.py
#
# sod's shock tube problem in 1d newtonian fluid.
# classic riemann problem with discontinuous initial conditions.
# =============================================================================
from typing import Annotated, Sequence

from simbi import ProblemParam, SimbiProblem
from simbi.types import CellSpacing, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class SodProblem(SimbiProblem):
    """sod's shock tube problem in 1d newtonian fluid."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic gas index")
    ]

    # domain
    resolution: Annotated[
        int, ProblemParam(100, cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        Sequence[Sequence[float]],
        ProblemParam([(0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(
            CellSpacing.LINEAR, description="grid spacing in x1 direction"
        ),
    ]

    # simulation control
    end_time: Annotated[
        float,
        ProblemParam(
            0.2,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]
    checkpoint_interval: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            checkpoint_safe=True,
            description="checkpoint interval",
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for sod shock tube."""

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            dx = (self.bounds[0][1] - self.bounds[0][0]) / nx
            for ii in range(nx):
                if ii * dx < 0.5:
                    yield (1.0, 0.0, 1.0)
                else:
                    yield (0.125, 0.0, 0.1)

        return gas_state
