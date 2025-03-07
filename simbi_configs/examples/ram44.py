from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.typing import InitialStateType
from typing import Sequence, Generator, Iterator
from dataclasses import dataclass


@dataclass
class SplitState:
    """State for split shock problem"""

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


class Ram44(BaseConfig):
    """
    Shock with non-zero transverse velocity on one side in 2D with 1 Partition
    This setup was adapted from Zhang and MacFadyen (2006) section 4.4
    """

    class config:
        nzones = DynamicArg("nzones", 400, help="number of grid zones", var_type=int)
        adiabatic_index = DynamicArg(
            "ad-gamma", 5.0 / 3.0, help="Adiabatic gas index", var_type=float
        )

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni, nj = self.resolution
            xextent = self.bounds[0][1] - self.bounds[0][0]
            dx = xextent / ni
            for j in range(nj):
                for i in range(ni):
                    x = self.bounds[0][0] + i * dx
                    if x < 0.5 * xextent:
                        yield tuple(SplitState(1.0, 0.0, 0.0, 1e3))
                    else:
                        yield tuple(SplitState(1.0, 0.0, 0.99, 1e-2))

        return gas_state

    @simbi_property
    def bounds(self) -> Sequence[Sequence[float]]:
        return ((0.0, 1.0), (0.0, 1.0))

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[DynamicArg]:
        return (self.config.nzones, self.config.nzones)

    @simbi_property
    def adiabatic_index(self) -> DynamicArg:
        return self.config.adiabatic_index

    @simbi_property
    def regime(self) -> str:
        return "srhd"

    @simbi_property
    def default_end_time(self) -> float:
        return 0.4
