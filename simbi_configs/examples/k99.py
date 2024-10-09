from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import *
import numpy as np

XMIN = -2.0
XMAX = +2.0
XMEM = 0.5 * (XMIN + XMAX)

def beta(u: list[float]) -> NDArray[numpy_float]:
    unp: NDArray[numpy_float] = np.asarray(u)
    gamma: float = (1.0 + unp.dot(unp))**(0.5)
    beta: NDArray[numpy_float] = unp / gamma
    return beta

class MagneticShockTube(BaseConfig):
    """
    Komissarov (1999), 1D SRMHD test problems.
    """

    nzones = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
    problem = DynamicArg(
        "problem",
        "contact",
        help="problem number from Komissarov (1999)",
        var_type=str,
        choices=[
            "fast-shock",
            "slow-shock",
            "fast-rarefaction",
            "slow-rarefaction",
            "alfven",
            "compound",
            "st-1",
            "st-2",
            "collision",
        ],
    )

    @simbi_property
    def initial_state(self) -> Sequence[Sequence[float]]:
        # defined as (rho, v1, v2, v3, pg, b1, b2, b3)
        if self.problem == 'fast-shock':
            return (
                (1.000, *beta([25.0, 0.0, 0.0]),    1.000, 20.0, 25.02, 0.0),
                (25.48, *beta([1.091, 0.3923,0.0]), 367.5, 20.0, 49.00, 0.0),
            )
        elif self.problem == 'slow-shock':
            return (
                (1.000, *beta([1.5300, 0.0, 0.0]),     10.00, 10.0, 18.28, 0.0),
                (3.323, *beta([0.9571, -0.6822, 0.0]), 55.36, 10.0, 14.49, 0.0),
            )
        elif self.problem == 'fast-rarefaction':
            return (
                (0.100, *beta([-2.000, 0.0, 0.0]),    1.00, 2.0, 0.000, 0.0),
                (0.562, *beta([-0.212, -0.590, 0.0]), 10.0, 2.0, 4.710, 0.0),
            )
        elif self.problem == 'slow-rarefaction':
            return (
                (1.78e-3, *beta([-0.765, -1.386, 0.0]), 0.1, 1.0, 1.022, 0.0),
                (0.01000, *beta([+0.0, 0.0, 0.0]),      1.0, 1.0, 0.000, 0.0),
            )
        elif self.problem == 'alfven':
            return (
                (1.0, *beta([0.0, 0.0, 0.0]),   1.0, 3.0, 3.0000, 0.0),
                (1.0, *beta([3.70, 5.76, 0.0]), 1.0, 3.0, -6.857, 0.0),
            )
        elif self.problem == 'compound':
            return ( 
                (1.0, *beta([0.0, 0.0, 0.0]),   1.0, 3.0, +3.000, 0.0),
                (1.0, *beta([3.70, 5.76, 0.0]), 1.0, 3.0, -6.857, 0.0),
            )
        elif self.problem == 'st-1':
            return (
                (1.0, 0.0, 0.0, 0.0, 1e3, 1.0, 0.0, 0.0),
                (0.1, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
            )
        elif self.problem == 'st-2':
            return (
                (1.0, 0.0, 0.0, 0.0, 30.0, 0.0, 20.0, 0.0),
                (0.1, 0.0, 0.0, 0.0, 1.00, 0.0, 0.00, 0.0),
            )
        else:
            return (
                (1.0, *beta([+5.0, 0.0, 0.0]), 1.0, 10.0, +10.0, 0.0),
                (1.0, *beta([-5.0, 0.0, 0.0]), 1.0, 10.0, -10.0, 0.0),
            )

    @simbi_property
    def geometry(self) -> Sequence[Sequence[float]]:
        return ((XMIN, XMAX, XMEM), (0.0, 1.0), (0.0, 1.0))

    @simbi_property
    def x1_cell_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[DynamicArg | int]:
        return (self.nzones, 1, 1)

    @simbi_property
    def gamma(self) -> float:
        return (4.0 / 3.0)

    @simbi_property
    def regime(self) -> str:
        return "srmhd"
