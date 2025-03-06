import numpy as np
from simbi import BaseConfig, DynamicArg, simbi_property
from typing import Any, Sequence, Generator

SEED = 12345
rng = np.random.default_rng(SEED)
PEEK_TO_PEEK = 0.01


class KelvinHelmholtz(BaseConfig):
    """
    Kelvin Helmholtz problem in Newtonian Fluid
    """

    class config:
        npts = DynamicArg(
            "npts", 256, help="Number of zones in x and y dimensions", var_type=int
        )

    xmin = -0.5
    xmax = 0.5
    ymin = -0.5
    ymax = 0.5
    rhoL = 2.0
    vxT = 0.5
    pL = 2.5
    rhoR = 1.0
    vxB = -0.5
    pR = 2.5

    @simbi_property
    def initial_primitive_state(self) -> Generator[tuple[float, ...], None, None]:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            dy = (self.ymax - self.ymin) / self.config.npts
            for j in range(self.config.npts):
                y = self.ymin + j * dy
                for i in range(self.config.npts):
                    vx_noise = PEEK_TO_PEEK * np.sin(2 * np.pi * rng.normal())
                    vy_noise = PEEK_TO_PEEK * np.sin(2 * np.pi * rng.normal())
                    if abs(y) < 0.25:
                        rho = self.rhoL
                        vx = self.vxT + vx_noise
                        vy = 0.0 + vy_noise
                        p = self.pL
                    else:
                        rho = self.rhoR
                        vx = self.vxB + vx_noise
                        vy = 0.0 + vy_noise
                        p = self.pR
                    yield rho, vx, vy, p

        return gas_state

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
        return (self.config.npts, self.config.npts)

    @simbi_property
    def adiabatic_index(self) -> float:
        return 5.0 / 3.0

    @simbi_property
    def regime(self) -> str:
        return "classical"

    @simbi_property
    def boundary_conditions(self) -> str:
        return "periodic"

    @simbi_property
    def solver(self) -> str:
        return "hllc"

    @simbi_property
    def data_directory(self) -> str:
        return "data/kh_config"
