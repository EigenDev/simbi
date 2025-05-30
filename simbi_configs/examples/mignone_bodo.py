from simbi import SimbiBaseConfig, SimbiField
from simbi.core.types.input import CoordSystem, Regime
from simbi.typing import InitialStateType
from typing import Generator, Iterator, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class ShockTubeState:
    rho: float
    v: float
    p: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.v
        yield self.p


class MignoneBodo(SimbiBaseConfig):
    """
    Mignone & Bodo (2005), Relativistic Test Problems on 1D Mesh
    """

    resolution: int = SimbiField(1000, description="resolution of grid zones")
    adiabatic_index: float = SimbiField(4.0 / 3.0, description="Adiabatic gas index")
    problem: int = SimbiField(1, description="test problem to compute", choices=[1, 2])

    bounds: list[tuple[float, float]] = SimbiField(
        [(0.0, 1.0)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )
    regime: Regime = SimbiField(Regime.SRHD, description="Physics regime")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._problem_state = {
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
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni = self.resolution
            xextent = self.bounds[0][1] - self.bounds[0][0]
            dx = xextent / ni
            for i in range(ni):
                xi = self.bounds[0][0] + i * dx
                if xi < 0.5 * xextent:
                    rho, v, p = self._problem_state[self.problem][0]
                else:
                    rho, v, p = self._problem_state[self.problem][1]
                yield (rho, v, p)

        return gas_state
