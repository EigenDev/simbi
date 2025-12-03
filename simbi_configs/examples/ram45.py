# =============================================================================
# ram45.py
#
# 1d shock-heating problem in planar geometry.
# adapted from zhang and macfadyen (2006) section 4.5.
# =============================================================================
from dataclasses import dataclass
from typing import Annotated, Iterator

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


@dataclass(frozen=True)
class ShockHeatingState:
    """state for shock heating problem."""

    rho: float
    v1: float
    p: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.v1
        yield self.p


class Ram45(SimbiProblem):
    """1d shock-heating problem, zhang & macfadyen (2006) 4.5."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index")
    ]

    # domain
    resolution: Annotated[
        int, ProblemParam(100, cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.SRHD, description="physics regime")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(
            CellSpacing.LINEAR, description="grid spacing in x1 direction"
        ),
    ]

    # numerics
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.OUTFLOW, BoundaryCondition.REFLECTING],
            description="boundary conditions",
        ),
    ]

    # simulation control
    end_time: Annotated[
        float,
        ProblemParam(
            2.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for shock heating problem."""

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            state = ShockHeatingState(1.0, (1.0 - 1.0e-8), 1e-6)

            for ii in range(nx):
                yield tuple(state)

        return gas_state
