from simbi import BaseConfig, DynamicArg, simbi_property
from typing import Sequence, Generator
from dataclasses import dataclass


class Ram61(BaseConfig):
    """(Hard Test: Fails)
    Shock with non-zero transverse velocity on both sides in 2D with 1 Partition
    This setup was adapted from Zhang and MacFadyen (2006) section 6.1
    """

    class config:
        nzones = DynamicArg("nzones", 400, help="number of grid zones", var_type=int)
        adiabatic_index = DynamicArg(
            "ad-gamma", 5.0 / 3.0, help="Adiabatic gas index", var_type=float
        )

    @simbi_property
    def initial_primitive_state(self) -> Generator[tuple[float, ...], None, None]:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni, nj = self.resolution
            for j in range(nj):
                for i in range(ni):
                    if i < ni // 2:
                        yield (1.0, 0.0, 0.90, 1e3)
                    else:
                        yield (1.0, 0.0, 0.90, 1e-2)

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
