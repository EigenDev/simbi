import numpy as np
from simbi import BaseConfig, simbi_property, DynamicArg, compute_num_polar_zones

XMIN = -6.0
XMAX = 6.0
PEXP = 1.0
RHOEXP = 0.1
REXP = 0.08
RSTOP = 1.0


def find_nearest(arr: NDArray[np.float64], val: float) -> Tuple[Any, Any]:
    idx = np.argmin(np.abs(arr - val))
    return idx, arr[idx]


class magneticBomb(BaseConfig):
    """The Magnetic Bomb
    Launch a cylindrical relativistic magnetized blast wave
    """

    # Dynamic Args to be fed to argparse
    rho0 = DynamicArg("rho0", 1.0e-4, help="density scale", var_type=float)
    p0 = DynamicArg("p0", 3.0e-5, help="pressure scale", var_type=float)
    b0 = DynamicArg("b0", 0.1, help="magnetic field scale", var_type=float)
    nzones = DynamicArg("nzones", 256, help="number of zones in x and y", var_type=int)
    adiabatic_index = DynamicArg(
        "ad-gamma", 4.0 / 3.0, help="Adiabtic gas index", var_type=float
    )

    def __init__(self) -> None:
        bx_shape = (1, self.nzones.value, self.nzones.value + 1)
        by_shape = (1, self.nzones.value + 1, self.nzones.value)
        bz_shape = (2, self.nzones.value, self.nzones.value)
        nzones = int(self.nzones)
        x1 = np.linspace(XMIN, XMAX, nzones)
        x2 = np.linspace(XMIN, XMAX, nzones)
        self.rho = np.ones((1, nzones, nzones), float) * self.rho0
        self.p = np.ones_like(self.rho) * self.p0
        self.v1 = np.zeros_like(self.rho)
        self.v2 = self.v1.copy()
        self.v3 = self.v1.copy()
        self.bvec = np.array(
            [np.ones(bx_shape) * self.b0, np.zeros(by_shape), np.zeros(bz_shape)],
            dtype=object,
        )

        xx, yy = np.meshgrid(x1, x2)
        rr = np.sqrt(xx**2 + yy**2)
        exp_reg = rr < REXP
        pslope = (PEXP - self.p0) / (RSTOP - REXP)
        rhoslope = (RHOEXP - self.rho0) / (RSTOP - REXP)
        self.p[:, exp_reg] = PEXP
        self.rho[:, exp_reg] = RHOEXP
        rbound = (rr > REXP) & (rr < RSTOP)
        self.p[:, rbound] = PEXP - pslope * (rr[rbound] - REXP)
        self.rho[:, rbound] = RHOEXP - rhoslope * (rr[rbound] - REXP)

    @simbi_property
    def initial_primitive_state(self) -> Sequence[NDArray[np.float64]]:
        return (
            self.rho,
            self.v1,
            self.v2,
            self.v3,
            self.p,
            self.bvec[0],
            self.bvec[1],
            self.bvec[2],
        )

    @simbi_property
    def bounds(self) -> Sequence[Sequence[Any]]:
        return ((XMIN, XMAX), (XMIN, XMAX), (0, 1))

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[int | DynamicArg]:
        return (self.nzones, self.nzones, 1)

    @simbi_property
    def adiabatic_index(self) -> DynamicArg:
        return self.adiabatic_index

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
