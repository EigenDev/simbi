from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.typing import InitialStateType
from typing import Sequence, Generator, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class ShockHeatingState:
    rho: float
    v1: float
    p: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.v1
        yield self.p


class Ram45(BaseConfig):
    """
    1D shock-heating problem in planar geometry
    This setup was adapted from Zhang and MacFadyen (2006) section 4.5
    """

    class config:
        nzones = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
        adiabatic_index = DynamicArg(
            "ad-gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type=float
        )

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni = self.resolution
            for i in range(ni):
                yield tuple(ShockHeatingState(1.0, (1.0 - 1.0e-5), 1e-6))

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

    @simbi_property
    def boundary_conditions(self) -> Sequence[str]:
        return ["outflow", "reflecting"]

    @simbi_property
    def default_end_time(self) -> float:
        return 2.0
