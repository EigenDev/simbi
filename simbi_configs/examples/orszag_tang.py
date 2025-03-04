import numpy as np
from simbi import BaseConfig, simbi_property, DynamicArg


XMIN = 0.0
XMAX = 2.0 * np.pi


class OrszagTang(BaseConfig):
    """The Orszag-Tang vortex test case"""

    # Dynamic Args to be fed to argparse
    v0 = DynamicArg("v0", 0.5, help="velocity scale", var_type=float)
    b0 = DynamicArg("b0", 1.0, help="magnetic field scale", var_type=float)
    nzones = DynamicArg("nzones", 256, help="number of zones in x and y", var_type=int)
    adiabatic_index = DynamicArg(
        "ad-gamma", 5.0 / 3.0, help="Adiabtic gas index", var_type=float
    )

    def __init__(self) -> None:
        p0 = self.adiabatic_index
        rho0 = self.adiabatic_index**2
        nzones = int(self.nzones)
        bx_shape = (1, nzones, nzones + 1)
        by_shape = (1, nzones + 1, nzones)
        bz_shape = (2, nzones, nzones)
        x1 = np.linspace(XMIN, XMAX, nzones)
        x2 = np.linspace(XMIN, XMAX, nzones)
        x2 = x2[:, None]
        self.rho = np.ones((1, nzones, nzones), float) * rho0
        self.p = np.ones_like(self.rho) * p0
        self.v1 = self.v0 * np.ones_like(self.rho) * (-np.sin(x2))
        self.v2 = self.v0 * np.ones_like(self.rho) * (+np.sin(x1))
        self.v3 = np.zeros_like(self.rho)
        self.bvec = np.array(
            [
                self.b0 * np.ones(bx_shape) * (-np.sin(1.0 * x2)),
                self.b0 * np.ones(by_shape) * (+np.sin(2.0 * x1)),
                np.zeros(bz_shape),
            ],
            dtype=object,
        )

        self.cs: float = (self.adiabatic_index - 1.0) / self.adiabatic_index

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
        return (XMAX - XMIN) / self.cs

    @simbi_property
    def solver(self) -> str:
        return "hlle"

    @simbi_property
    def boundary_conditions(self) -> Sequence[str]:
        return ["periodic"]
