# =============================================================================
# ram44.py
#
# shock with non-zero transverse velocity on one side in 2d.
# adapted from zhang and macfadyen (2006) section 4.4.
# =============================================================================
from dataclasses import dataclass
from typing import Annotated, Iterator

from simbi import ProblemParam, SimbiProblem
from simbi.types import CellSpacing, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


@dataclass
class SplitState:
    """state for split shock problem."""

    rho: float
    v1: float
    v2: float
    pre: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.v1
        yield self.v2
        yield self.pre

    def __len__(self) -> int:
        return 4


class Ram44(SimbiProblem):
    """shock with non-zero transverse velocity, zhang & macfadyen (2006) 4.4."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]

    # domain
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((400, 400), cli=True, description="grid resolution"),
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
        Regime, ProblemParam(Regime.RHD, description="physics regime")
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
            0.4,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for ram44 shock."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / nx

            left_state = SplitState(1.0, 0.0, 0.0, 1e3)
            right_state = SplitState(1.0, 0.0, 0.99, 1e-2)

            for jj in range(ny):
                for ii in range(nx):
                    x = xmin + (ii + 0.5) * dx

                    if x < 0.5 * xextent:
                        yield tuple(left_state)
                    else:
                        yield tuple(right_state)

        return gas_state
