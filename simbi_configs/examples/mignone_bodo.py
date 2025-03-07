from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.typing import InitialStateType
from typing import Sequence, Generator, Iterator, Any
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


class MignoneBodo(BaseConfig):
    """
    Mignone & Bodo (2005), Relativistic Test Problems on 1D Mesh
    """

    class config:
        nzones = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
        adiabatic_index = DynamicArg(
            "ad-gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type=float
        )
        problem = DynamicArg(
            "problem", 1, help="test problem to compute", var_type=int, choices=[1, 2]
        )

    def __init__(self) -> None:
        self.problem_state = {
            1: (
                ShockTubeState(1.0, 0.0, 1.0),
                ShockTubeState(0.1, 0.0, 0.125),
            ),
            2: (
                ShockTubeState(1.0, -2.0, 0.4),
                ShockTubeState(1.0, 2.0, 0.4),
            ),
        }

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni = self.resolution
            xextent = self.bounds[1] - self.bounds[0]
            dx = xextent / ni
            for i in range(ni):
                xi = self.bounds[0] + i * dx
                if xi < 0.5 * xextent:
                    rho, v, p = self.problem_state[int(self.config.problem)][0]
                else:
                    rho, v, p = self.problem_state[int(self.config.problem)][1]
                yield (rho, v, p)

        return gas_state

    @simbi_property
    def bounds(self) -> Sequence[float]:
        return (0.0, 1.0)

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> DynamicArg:
        return self.config.nzones

    @simbi_property
    def adiabatic_index(self) -> DynamicArg:
        return self.config.adiabatic_index

    @simbi_property
    def regime(self) -> str:
        return "srhd"
