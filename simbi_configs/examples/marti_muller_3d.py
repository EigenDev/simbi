from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.typing import InitialStateType
from typing import Sequence, Generator


class MartiMuller3D(BaseConfig):
    """
    Marti & Muller (2003), Relativistic  Shock Tube Problem on 3D Mesh
    """

    class config:
        nzones = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
        adiabatic_index = DynamicArg(
            "ad-gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type=float
        )

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni, nj, nk = self.resolution
            xextent = self.bounds[0][1] - self.bounds[0][0]
            dx = xextent / ni
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        xi = self.bounds[0][0] + i * dx
                        if xi <= 0.5 * xextent:
                            yield (10.0, 0.0, 0.0, 0.0, 13.33)
                        else:
                            yield (1.0, 0.0, 0.0, 0.0, 1e-10)

        return gas_state

    @simbi_property
    def bounds(self) -> Sequence[Sequence[float]]:
        return ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[DynamicArg]:
        return (self.config.nzones, self.config.nzones, self.config.nzones)

    @simbi_property
    def adiabatic_index(self) -> DynamicArg:
        return self.config.adiabatic_index

    @simbi_property
    def regime(self) -> str:
        return "srhd"
