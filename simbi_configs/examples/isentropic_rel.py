from dataclasses import dataclass
from typing import Callable, Sequence, Any
import numpy as np
from numpy.typing import NDArray

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CellSpacing, CoordSystem, Regime
from simbi.core.types.typing import GasStateGenerator, InitialStateType
from pydantic import validator


# Constants
ALPHA_MAX = 2.0
ALPHA_MIN = 1e-3


def range_limited_float(min_val: float, max_val: float) -> Callable[[float], float]:
    """Creates a validator function for float within range"""

    def validate_range(v: float) -> float:
        if v < min_val or v > max_val:
            raise ValueError(f"Value must be between {min_val} and {max_val}")
        return v

    return validate_range


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


class IsentropicState:
    """State calculator for isentropic wave"""

    def __init__(
        self, params: IsentropicWaveParams, adiabatic_index: float, alpha: float
    ):
        self.params = params
        self.adiabatic_index = adiabatic_index
        self.alpha = alpha

    def density(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate density at position x"""
        wave = self.params.make_wave_function()
        return 1.0 + self.alpha * wave(x)

    def sound_speed(
        self, rho: NDArray[np.float64] | float, p: NDArray[np.float64] | float
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


class IsentropicRelWave(SimbiBaseConfig):
    """Relativistic Isentropic Pulse in 1D, Entropy conserving"""

    adiabatic_index: float = SimbiField(4.0 / 3.0, description="Adiabatic gas index")

    alpha: float = SimbiField(0.5, description="Wave amplitude")

    # Required fields from SimbiBaseConfig
    resolution: int = SimbiField(1000, description="Grid resolution")

    bounds: Sequence[Sequence[float]] = SimbiField(
        [(0.0, 1.0)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.SRHD, description="Physics regime")

    # Optional customizations with non-default values
    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    boundary_conditions: str = SimbiField("periodic", description="Boundary conditions")

    # Validators
    @validator("alpha")
    def validate_alpha(cls, v: float) -> float:
        """Validate alpha is within range"""
        return range_limited_float(ALPHA_MIN, ALPHA_MAX)(v)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._wave_params = IsentropicWaveParams()
        self._state = IsentropicState(
            self._wave_params, self.adiabatic_index, self.alpha
        )

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for isentropic wave.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            dx = (self.bounds[0][1] - self.bounds[0][0]) / nx
            x = np.fromiter((i * dx for i in range(nx)), dtype=np.float64)
            rho = self._state.density(x)
            p = self._state.pressure(rho)
            v = self._state.velocity(rho, p)

            for i in range(nx):
                yield (rho[i], v[i], p[i])

        return gas_state
