from dataclasses import dataclass
from simbi import BaseConfig, DynamicArg, simbi_property
from typing import Any, Generator, Sequence, Union, Callable
from numpy.typing import NDArray
import numpy as np
import argparse

ALPHA_MAX = 2.0
ALPHA_MIN = 1e-3


def range_limited_float_type(x: str, min_val: float, max_val: float) -> float:
    """Return float value within range"""
    val = float(x)
    if val < min_val or val > max_val:
        raise argparse.ArgumentTypeError(
            f"Value must be between {min_val} and {max_val}"
        )
    return val


@dataclass(frozen=True)
class IsentropicWaveParams:
    """Physical parameters for isentropic wave"""

    rho_ref: float = 1.0
    p_ref: float = 1.0

    def make_wave_function(
        self,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Create wave shape function"""
        return lambda x: np.sin(2 * np.pi * x)


@dataclass(frozen=True)
class IsentropicState:
    """State calculator for isentropic wave"""

    params: IsentropicWaveParams
    adiabatic_index: float
    alpha: float

    def density(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate density at position x"""
        wave = self.params.make_wave_function()
        return 1.0 + self.alpha * wave(x)

    def sound_speed(
        self, rho: NDArray[np.float64], p: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate sound speed"""
        h = 1.0 + self.adiabatic_index * p / (rho * (self.adiabatic_index - 1.0))
        return np.sqrt(self.adiabatic_index * p / (rho * h))

    def pressure(self, rho: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate pressure"""
        return self.params.p_ref * (rho / self.params.rho_ref) ** self.adiabatic_index

    def velocity(
        self, rho: NDArray[np.float64], p: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate velocity"""
        cs_ref = self.sound_speed(self.params.rho_ref, self.params.p_ref)
        cs = self.sound_speed(rho, p)
        return 2.0 / (self.adiabatic_index - 1.0) * (cs - cs_ref)


class IsentropicRelWave(BaseConfig):
    """Relativistic Isentropic Pulse in 1D, Entropy conserving"""

    class config:
        nzones = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
        adiabatic_index = DynamicArg(
            "ad-index", 4.0 / 3.0, help="Adiabatic gas index", var_type=float
        )
        alpha = DynamicArg(
            "alpha",
            0.5,
            help="Wave amplitude",
            var_type=lambda x: range_limited_float_type(x, ALPHA_MIN, ALPHA_MAX),
        )

    def __init__(self) -> None:
        super().__init__()
        self.wave_params = IsentropicWaveParams()
        self.state = IsentropicState(
            self.wave_params, self.config.adiabatic_index, self.config.alpha
        )

    @simbi_property
    def initial_primitive_state(
        self,
    ) -> Generator[tuple[float, float, float], None, None]:
        """Return initial primitive generator"""

        def _initial_state(resolution=(self.config.nzones,), bounds=(0.0, 1.0)):
            dx = (bounds[1] - bounds[0]) / resolution[0]
            x = np.fromiter((i * dx for i in range(resolution[0])), dtype=np.float64)
            rho = self.state.density(x)
            p = self.state.pressure(rho)
            v = self.state.velocity(rho, p)
            yield from zip(rho, v, p)

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
        return "srhd"

    @simbi_property
    def boundary_conditions(self) -> str:
        return "periodic"
