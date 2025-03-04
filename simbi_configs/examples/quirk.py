import numpy as np
from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import *


class Quirk(BaseConfig):
    """
    Quirk's problem in Newtonian Fluid
    """

    nxpts = DynamicArg(
        "nxpts", 2400, help="Number of zones in x direction", var_type=int
    )
    nypts = DynamicArg("nypts", 20, help="Number of zones in y direction", var_type=int)
    low_mach = DynamicArg(
        "low_mach", False, help="Low Mach number problem", var_type=bool
    )

    xmin = +0.0
    xmax = +2400.0
    ymin = +0.0
    ymax = +20.0

    def __init__(self) -> None:

        if self.low_mach:
            # Mach 6 scale
            rhoL = 216.0 / 41.0
            vxL = (35.0 / 36.0) * np.sqrt(35)
            vyL = 0.0
            pL = 251.0 / 6.0
        else:
            # Mach 20 case
            rhoL = 160.0 / 27.0
            vxL = (133.0 / 8.0) * np.sqrt(1.4)
            vyL = 0.0
            pL = 466.5

        # static state
        rhoR = 1.0
        vxR = +0.0
        vyR = +0.0
        pR = 1.0

        x = np.linspace(self.xmin, self.xmax, self.nxpts.value)

        self.rho = np.zeros(shape=(self.nypts.value, self.nxpts.value))
        self.vx = np.zeros_like(self.rho)
        self.vy = np.zeros_like(self.rho)
        self.p = np.zeros_like(self.rho)

        self.rho[:, x <= 5] = rhoL
        self.rho[:, x > 5] = rhoR
        self.vx[:, x <= 5] = vxL
        self.vx[:, x > 5] = vxR
        self.vy[:, x <= 5] = vyL
        self.vy[:, x > 5] = vyR
        self.p[:, x <= 5] = pL
        self.p[:, x > 5] = pR

        # introduce artificial numerical noise to all
        # variables to break the symmetry
        np.random.seed(42)
        self.rho += 1e-6 * np.random.randn(*self.rho.shape)
        self.vx += 1e-6 * np.random.randn(*self.vx.shape)
        self.vy += 1e-6 * np.random.randn(*self.vy.shape)
        self.p += 1e-6 * np.random.randn(*self.p.shape)

    @simbi_property
    def initial_primitive_state(self) -> Sequence[NDArray[numpy_float]]:
        return (self.rho, self.vx, self.vy, self.p)

    @simbi_property
    def bounds(self) -> Sequence[Sequence[float]]:
        return ((self.xmin, self.xmax), (self.ymin, self.ymax))

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[Any]:
        return (self.nxpts, self.nypts)

    @simbi_property
    def adiabatic_index(self) -> float:
        return 5.0 / 3.0

    @simbi_property
    def regime(self) -> str:
        return "classical"

    @simbi_property
    def boundary_conditions(self) -> str:
        return "reflecting"

    @simbi_property
    def solver(self) -> str:
        return "hllc"

    @simbi_property
    def data_directory(self) -> str:
        return f"data/quirk/fix/{'low_mach' if self.low_mach else 'high_mach'}"

    @simbi_property
    def default_end_time(self) -> float:
        return 330 if self.low_mach else 100
