import math
from simbi import BaseConfig, simbi_property, DynamicArg
from simbi.typing import InitialStateType
from typing import Sequence, Generator
from functools import partial

XMIN = -6.0
XMAX = 6.0
P_EXP = 1.0
RHO_EXP = 0.1
R_EXP = 0.08
R_STOP = 1.0


class MagneticBomb(BaseConfig):
    """The Magnetic Bomb
    Launch a cylindrical relativistic magnetized blast wave
    """

    class config:
        rho0 = DynamicArg("rho0", 1.0e-4, help="density scale", var_type=float)
        p0 = DynamicArg("p0", 3.0e-5, help="pressure scale", var_type=float)
        b0 = DynamicArg("b0", 0.1, help="magnetic field scale", var_type=float)
        nzones = DynamicArg(
            "nzones", 256, help="number of zones in x and y", var_type=int
        )
        adiabatic_index = DynamicArg(
            "ad-gamma", 4.0 / 3.0, help="Adiabtic gas index", var_type=float
        )

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> Generator[tuple[float, ...]]:
            ni, nj, nk = self.resolution
            xbounds = self.bounds[0]
            ybounds = self.bounds[1]
            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj
            rho_amb = self.config.rho0
            pre_amb = self.config.p0
            pslope = (P_EXP - pre_amb) / (R_STOP - R_EXP)
            rhoslope = (RHO_EXP - rho_amb) / (R_STOP - R_EXP)
            for k in range(nk):
                for j in range(nj):
                    y = ybounds[0] + j * dy
                    for i in range(ni):
                        x = xbounds[0] + i * dx
                        r = math.sqrt(x**2 + y**2)
                        if r < R_EXP:
                            yield (RHO_EXP, 0.0, 0.0, 0.0, P_EXP)
                        elif r > R_EXP and r < R_STOP:
                            yield (
                                RHO_EXP - rhoslope * (r - R_EXP),
                                0.0,
                                0.0,
                                0.0,
                                P_EXP - rhoslope * (r - R_EXP),
                            )
                        else:
                            yield (rho_amb, 0.0, 0.0, 0.0, pre_amb)

        def b_field(bn: str) -> Generator[float, None, None]:
            ni, nj, nk = self.resolution
            for k in range(nk + (bn == "bz")):
                for j in range(nj + (bn == "by")):
                    for i in range(ni + (bn == "bx")):
                        if bn == "bx":
                            yield self.config.b0
                        else:
                            yield 0.0

        bx = partial(b_field, "bx")
        by = partial(b_field, "by")
        bz = partial(b_field, "bz")

        return (gas_state, bx, by, bz)

    @simbi_property
    def bounds(self) -> Sequence[Sequence[float]]:
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
        return 4.0

    @simbi_property
    def solver(self) -> str:
        return "hlle"

    @simbi_property
    def boundary_conditions(self) -> Sequence[str]:
        return ["outflow"]
