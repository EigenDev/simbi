from simbi import BaseConfig, DynamicArg, simbi_property
from typing import Sequence, Generator
from dataclasses import dataclass


@dataclass
class ShockTubeState:
    """Dataclass for Shock Tube State"""

    rho: float
    v1: float
    v2: float
    p: float

    def __iter__(self) -> Generator:
        yield self.rho
        yield self.v1
        yield self.v2
        yield self.p


class SodProblemQuad(BaseConfig):
    """
    Sod's Shock Tube Problem in 2D Newtonian Fluid with 4 Partitions
    This setup was adapted from Zhang and MacFadyen (2006) section 4.8 pg. 11
    """

    class config:
        nzones = DynamicArg("nzones", 256, help="number of grid zones", var_type=int)
        adiabatic_index = DynamicArg(
            "ad-gamma", 5.0 / 3.0, help="Adiabatic gas index", var_type=float
        )

    @simbi_property
    def initial_primitive_state(self) -> Generator[tuple[float, ...], None, None]:

        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni, nj = self.resolution
            xextent = self.bounds[0][1] - self.bounds[0][0]
            yextent = self.bounds[1][1] - self.bounds[1][0]
            dx = xextent / ni
            dy = yextent / nj
            for j in range(nj):
                for i in range(ni):
                    x = self.bounds[0][0] + i * dx
                    y = self.bounds[1][0] + j * dy
                    if x < 0.5 * xextent:
                        if y < 0.5 * yextent:
                            yield tuple(ShockTubeState(0.5, 0.0, 0.0, 1.0))
                        else:
                            yield tuple(ShockTubeState(0.1, 0.9, 0.0, 1.0))
                    else:
                        if y < 0.5 * yextent:
                            yield tuple(ShockTubeState(0.1, 0.0, 0.9, 1.0))
                        else:
                            yield tuple(ShockTubeState(0.1, 0.0, 0.0, 0.01))

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
