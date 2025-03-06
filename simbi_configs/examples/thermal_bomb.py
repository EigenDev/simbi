import math
from simbi import (
    BaseConfig,
    simbi_property,
    DynamicArg,
    compute_num_polar_zones,
    find_nearest,
)
from typing import Sequence, Generator, Any


RHO_AMB = 1.0
T_AMB = 1e-10
NU = 3.0


class thermalBomb(BaseConfig):
    """The Thermal Bomb
    Launch a relativistic blast wave on a 2D Spherical Logarithmic mesh with variable zones per decade in radius
    """

    class config:
        e0 = DynamicArg("e0", 10.0, help="energy scale", var_type=float)
        rho0 = DynamicArg("rho0", 1.0, help="density scale", var_type=float)
        rinit = DynamicArg("rinit", 0.1, help="intial grid radius", var_type=float)
        rend = DynamicArg("rend", 1.0, help="radial extent", var_type=float)
        k = DynamicArg("k", 0.0, help="density power law k", var_type=float)
        full_sphere = DynamicArg(
            "full-sphere",
            False,
            help="flag for full_sphere computation",
            var_type=bool,
            action="store_true",
        )
        zpd = DynamicArg(
            "zpd", 1024, help="number of radial zones per decade", var_type=int
        )
        adiabatic_index = DynamicArg(
            "ad-gamma", 4.0 / 3.0, help="Adiabtic gas index", var_type=float
        )

    def __init__(self) -> None:
        ndec = math.log10(self.config.rend / self.config.rinit)
        self.nr = round(self.config.zpd * ndec)
        self.theta_min = 0
        self.theta_max = math.pi if self.config.full_sphere else 0.5 * math.pi
        self.npolar = compute_num_polar_zones(
            rmin=self.config.rinit,
            rmax=self.config.rend,
            nr=self.nr,
            theta_bounds=(self.theta_min, self.theta_max),
        )

    @simbi_property
    def initial_primitive_state(self) -> Generator[tuple[float, ...], None, None]:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni, nj = self.resolution
            explosion_radius = self.config.rinit * 1.5
            dlogr = math.log10(self.config.rend / self.config.rinit) / ni
            for j in range(nj):
                for i in range(ni):
                    r = self.config.rinit * 10 ** (i * dlogr)
                    rho = RHO_AMB * r ** (-self.config.k)
                    if r <= explosion_radius:
                        pre = (self.config.adiabatic_index - 1.0) * (
                            3.0
                            * self.config.e0
                            / (NU + 1)
                            / math.pi
                            / explosion_radius**NU
                        )
                    else:
                        pre = T_AMB * rho
                    yield (
                        rho,
                        0.0,
                        0.0,
                        pre,
                    )

        return gas_state

    @simbi_property
    def bounds(self) -> Sequence[Sequence[Any]]:
        return ((self.config.rinit, self.config.rend), (self.theta_min, self.theta_max))

    @simbi_property
    def x1_spacing(self) -> str:
        return "log"

    @simbi_property
    def coord_system(self) -> str:
        return "spherical"

    @simbi_property
    def resolution(self) -> Sequence[int]:
        return (self.nr, self.npolar)

    @simbi_property
    def adiabatic_index(self) -> DynamicArg:
        return self.config.adiabatic_index

    @simbi_property
    def regime(self) -> str:
        return "srhd"

    @simbi_property
    def default_start_time(self) -> float:
        return 0.0

    @simbi_property
    def default_end_time(self) -> float:
        return 1.0

    @simbi_property
    def solver(self) -> str:
        return "hllc"

    @simbi_property
    def boundary_conditions(self) -> Sequence[str]:
        return ["reflecting", "outflow", "outflow", "outflow"]
