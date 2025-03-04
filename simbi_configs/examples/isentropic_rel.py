from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import *
import numpy as np
import argparse

ALPHA_MAX = 2.0
ALPHA_MIN = 1e-3


def range_limited_float_type(arg: Any) -> Any:
    """Type function for argparse - a float within some predefined bounds"""
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < ALPHA_MIN or f >= ALPHA_MAX:
        raise argparse.ArgumentTypeError(
            "Argument must be < " + str(ALPHA_MAX) + " and > " + str(ALPHA_MIN)
        )
    return f


def func(x: NDArray[numpy_float]) -> NDArray[numpy_float]:
    return np.asanyarray(np.sin(2 * np.pi * x))


def rho(alpha: DynamicArg, x: NDArray[numpy_float]) -> NDArray[numpy_float]:
    return np.asanyarray(1.0 + alpha * func(x))


def cs(
    adiabatic_index: DynamicArg,
    rho: Union[NDArray[numpy_float], float],
    pressure: Union[NDArray[numpy_float], float],
) -> NDArray[numpy_float]:
    h = 1.0 + adiabatic_index * pressure / (rho * (adiabatic_index - 1.0))
    return np.asanyarray(np.sqrt(adiabatic_index * pressure / (rho * h)))


def pressure(
    p_ref: float, adiabatic_index: DynamicArg, rho: NDArray[numpy_float], rho_ref: float
) -> NDArray[numpy_float]:
    return np.asanyarray(p_ref * (rho / rho_ref) ** float(adiabatic_index))


def velocity(
    adiabatic_index: DynamicArg,
    rho: NDArray[numpy_float],
    rho_ref: float,
    pressure: NDArray[numpy_float],
    p_ref: float,
) -> NDArray[numpy_float]:
    return np.asanyarray(
        2
        / (adiabatic_index - 1.0)
        * (cs(adiabatic_index, rho, pressure) - cs(adiabatic_index, rho_ref, p_ref))
    )


class IsentropicRelWave(BaseConfig):
    """
    Relativistic Isentropic Pulse in 1D, Entropy conserving
    """

    nzones = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
    adiabatic_index = DynamicArg(
        "ad-index", 4.0 / 3.0, help="Adiabatic gas index", var_type=float
    )
    alpha = DynamicArg(
        "alpha", 0.5, help="Wave amplitude", var_type=range_limited_float_type
    )
    rho_ref = 1.0
    p_ref = 1.0

    def __init__(self) -> None:
        x = np.linspace(0, 1, self.nzones.value, dtype=float)
        self.density = rho(self.alpha, x)
        self.pre = pressure(
            self.p_ref, self.adiabatic_index, self.density, self.rho_ref
        )
        self.beta = velocity(
            self.adiabatic_index, self.density, self.rho_ref, self.pre, self.p_ref
        )

    @simbi_property
    def initial_primitive_state(self) -> Sequence[NDArray[numpy_float]]:
        return (self.density, self.beta, self.pre)

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
        return self.nzones

    @simbi_property
    def adiabatic_index(self) -> DynamicArg:
        return self.adiabatic_index

    @simbi_property
    def regime(self) -> str:
        return "srhd"

    @simbi_property
    def boundary_conditions(self) -> str:
        return "periodic"
