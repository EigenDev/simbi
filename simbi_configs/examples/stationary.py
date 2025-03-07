from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.typing import InitialStateType
from typing import Sequence, Generator


class StationaryWaveHLL(BaseConfig):
    """
    Stationary Wave Test Problems in 1D Newtonian Fluid using HLL solver
    """

    class config:
        nzones = DynamicArg("nzones", 400, help="number of grid zones", var_type=int)
        adiabatic_index = DynamicArg(
            "ad-gamma", 5.0 / 3.0, help="Adiabatic gas index", var_type=float
        )

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni = self.resolution
            xextent = self.bounds[1] - self.bounds[0]
            dx = xextent / ni
            for i in range(ni):
                xi = self.bounds[0] + i * dx
                if xi < 0.5 * xextent:
                    yield (1.4, 0.0, 1.0)
                else:
                    yield (1.0, 0.0, 1.0)

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
        return "classical"

    @simbi_property
    def solver(self) -> str:
        return "hlle"

    @simbi_property
    def data_directory(self) -> str:
        return "data/stationary/hlle"


class StationaryWaveHLLC(StationaryWaveHLL):
    """
    Stationary Wave Test Problems in 1D Newtonian Fluid using HLLC Toro et al. (1992) solver
    """

    @simbi_property
    def solver(self) -> str:
        return "hllc"

    @simbi_property
    def data_directory(self) -> str:
        return "data/stationary/hllc"
