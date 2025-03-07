import math
from simbi import BaseConfig, simbi_property, DynamicArg
from simbi.typing import InitialStateType
from typing import Sequence, Generator, Any
from functools import partial


XMIN = 0.0
XMAX = 2.0 * math.pi


class OrszagTang(BaseConfig):
    """The Orszag-Tang vortex test case"""

    class config:
        v0 = DynamicArg("v0", 0.5, help="velocity scale", var_type=float)
        b0 = DynamicArg("b0", 1.0, help="magnetic field scale", var_type=float)
        nzones = DynamicArg(
            "nzones", 256, help="number of zones in x and y", var_type=int
        )
        adiabatic_index = DynamicArg(
            "ad-gamma", 5.0 / 3.0, help="Adiabtic gas index", var_type=float
        )

    def __init__(self) -> None:
        self.cs: float = (
            self.config.adiabatic_index - 1.0
        ) / self.config.adiabatic_index

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni, nj, nk = self.resolution
            p0 = float(self.config.adiabatic_index)
            rho0 = float(self.config.adiabatic_index) ** 2
            v0 = self.config.v0
            xbounds = self.bounds[0]
            ybounds = self.bounds[1]
            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj
            for k in range(nk):
                for j in range(nj):
                    y = ybounds[0] + j * dy
                    for i in range(ni):
                        x = xbounds[0] + i * dx
                        vx = -v0 * math.sin(y)
                        vy = +v0 * math.sin(x)
                        yield (rho0, vx, vy, 0.0, p0)

        def b_field(bn: str) -> Generator[float, None, None]:
            ni, nj, nk = self.resolution
            xbounds = self.bounds[0]
            ybounds = self.bounds[1]
            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj
            b0 = self.config.b0
            for k in range(nk + (bn == "bz")):
                for j in range(nj + (bn == "by")):
                    y = ybounds[0] + j * dy
                    for i in range(ni + (bn == "bx")):
                        x = xbounds[0] + i * dx
                        if bn == "bx":
                            yield -b0 * math.sin(1.0 * y)
                        elif bn == "by":
                            yield +b0 * math.sin(2.0 * x)
                        else:
                            yield 0.0

        bx = partial(b_field, "bx")
        by = partial(b_field, "by")
        bz = partial(b_field, "bz")

        return gas_state, bx, by, bz

    @simbi_property
    def bounds(self) -> Sequence[Sequence[Any]]:
        return ((XMIN, XMAX), (XMIN, XMAX))

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[int | DynamicArg]:
        return (self.config.nzones, self.config.nzones, 1)

    @simbi_property
    def adiabatic_index(self) -> DynamicArg:
        return self.config.adiabatic_index

    @simbi_property
    def regime(self) -> str:
        return "srmhd"

    @simbi_property
    def default_start_time(self) -> float:
        return 0.0

    @simbi_property
    def default_end_time(self) -> float:
        return (XMAX - XMIN) / self.cs

    @simbi_property
    def solver(self) -> str:
        return "hlle"

    @simbi_property
    def boundary_conditions(self) -> Sequence[str]:
        return ["periodic"]
