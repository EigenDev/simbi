# =============================================================================
# isentropic_rel.py
#
# relativistic isentropic pulse in 1d, entropy conserving.
# =============================================================================
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType

# constants
ALPHA_MAX = 2.0
ALPHA_MIN = 1e-3


@dataclass(frozen=True)
class IsentropicWaveParams:
    """physical parameters for isentropic wave."""

    rho_ref: float = 1.0
    p_ref: float = 1.0

    def make_wave_function(
        self,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return lambda x: np.sin(2 * np.pi * x)


class IsentropicState:
    """state calculator for isentropic wave."""

    def __init__(
        self, params: IsentropicWaveParams, adiabatic_index: float, alpha: float
    ):
        self.params = params
        self.adiabatic_index = adiabatic_index
        self.alpha = alpha

    def density(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        wave = self.params.make_wave_function()
        return 1.0 + self.alpha * wave(x)

    def sound_speed(
        self, rho: NDArray[np.float64] | float, p: NDArray[np.float64] | float
    ) -> NDArray[np.float64]:
        h = 1.0 + self.adiabatic_index * p / (
            rho * (self.adiabatic_index - 1.0)
        )
        return np.sqrt(self.adiabatic_index * p / (rho * h))

    def pressure(self, rho: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            self.params.p_ref
            * (rho / self.params.rho_ref) ** self.adiabatic_index
        )

    def velocity(
        self, rho: NDArray[np.float64], p: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        cs_ref = self.sound_speed(self.params.rho_ref, self.params.p_ref)
        cs = self.sound_speed(rho, p)
        return 2.0 / (self.adiabatic_index - 1.0) * (cs - cs_ref)


class IsentropicRelWave(SimbiProblem):
    """relativistic isentropic pulse in 1d, entropy conserving."""

    # physics
    adiabatic_index: float = ProblemParam(
        4.0 / 3.0, description="adiabatic gas index"
    )
    alpha: float = ProblemParam(0.5, cli=True, description="wave amplitude")

    # domain
    resolution: int = ProblemParam(
        1000, cli=True, description="grid resolution"
    )
    bounds: Sequence[Sequence[float]] = ProblemParam(
        [(0.0, 1.0)], description="domain boundaries"
    )
    coord_system: CoordSystem = ProblemParam(
        CoordSystem.CARTESIAN, description="coordinate system"
    )
    regime: Regime = ProblemParam(Regime.SRHD, description="physics regime")
    x1_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="grid spacing in x1 direction"
    )

    # numerics
    boundary_conditions: BoundaryCondition = ProblemParam(
        BoundaryCondition.PERIODIC, description="boundary conditions"
    )

    # simulation control
    end_time: float = ProblemParam(
        1.0, cli=True, checkpoint_safe=True, description="simulation end time"
    )

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if v < ALPHA_MIN or v > ALPHA_MAX:
            raise ValueError(
                f"alpha must be between {ALPHA_MIN} and {ALPHA_MAX}"
            )
        return v

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        wave_params = IsentropicWaveParams()
        state = IsentropicState(wave_params, self.adiabatic_index, self.alpha)
        object.__setattr__(self, "_wave_params", wave_params)
        object.__setattr__(self, "_state", state)

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for isentropic wave."""

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            dx = (self.bounds[0][1] - self.bounds[0][0]) / nx
            x = np.fromiter((ii * dx for ii in range(nx)), dtype=np.float64)
            rho = self._state.density(x)
            p = self._state.pressure(rho)
            v = self._state.velocity(rho, p)

            for ii in range(nx):
                yield (rho[ii], v[ii], p[ii])

        return gas_state
