# =============================================================================
# mignone_bodo.py
#
# mignone & bodo (2005), relativistic test problems on 1d mesh.
# =============================================================================
from dataclasses import dataclass
from typing import Annotated, Iterator

from simbi import ProblemParam, SimbiProblem
from simbi.types import CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


@dataclass(frozen=True)
class ShockTubeState:
    rho: float
    v: float
    p: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.v
        yield self.p


class MignoneBodo(SimbiProblem):
    """mignone & bodo (2005), relativistic test problems on 1d mesh."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic gas index")
    ]
    problem: Annotated[
        int,
        ProblemParam(
            1, cli=True, description="test problem to compute (1 or 2)"
        ),
    ]

    # domain
    resolution: Annotated[
        int, ProblemParam(1000, cli=True, description="grid resolution")
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

    @property
    def problem_state(self) -> dict[int, tuple[ShockTubeState, ShockTubeState]]:
        return {
            1: (
                ShockTubeState(1.0, 0.0, 1.0),
                ShockTubeState(0.1, 0.0, 0.125),
            ),
            2: (
                ShockTubeState(1.0, -0.2, 0.4),
                ShockTubeState(1.0, +0.2, 0.4),
            ),
        }

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for mignone & bodo shock tube."""

        def gas_state() -> GasStateGenerator:
            ni = self.resolution
            xextent = self.bounds[0][1] - self.bounds[0][0]
            dx = xextent / ni
            for ii in range(ni):
                xi = self.bounds[0][0] + ii * dx
                if xi < 0.5 * xextent:
                    rho, v, p = self.problem_state[self.problem][0]
                else:
                    rho, v, p = self.problem_state[self.problem][1]
                yield (rho, v, p)

        return gas_state
