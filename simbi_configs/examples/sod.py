from simbi import BaseConfig, DynamicArg, simbi_property
from typing import Generator, Sequence


class SodProblem(BaseConfig):
    """
    Sod's Shock Tube Problem in 1D Newtonian Fluid
    """

    class config:
        nzones = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
        adiabatic_index = DynamicArg(
            "ad-gamma", 5.0 / 3.0, help="Adiabatic gas index", var_type=float
        )

    @simbi_property
    def initial_primitive_state(
        self,
    ) -> Generator[tuple[float, float, float], None, None]:
        """return initial primitive generator"""

        def _initial_state():
            nx = self.resolution
            dx = (self.bounds[1] - self.bounds[0]) / nx
            for i in range(nx):
                if i * dx < 0.5:
                    yield 1.0, 0.0, 1.0
                else:
                    yield 0.125, 0.0, 0.1

        return _initial_state

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
