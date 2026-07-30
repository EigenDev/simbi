# =============================================================================
# isentropic_rel.py
#
# relativistic isentropic pulse in 1d, entropy conserving.
# =============================================================================
from dataclasses import dataclass
from typing import Annotated, Callable, Sequence

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
    p_ref: float = 100.0

    def make_wave_function(
        self, ell: float
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        def wave_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
            wave = np.where(
                np.abs(x) < ell,
                ((x / ell) ** 2.0 - 1.0) ** 4,
                0.0,
            )
            return wave

        return wave_function


class IsentropicState:
    """state calculator for isentropic wave."""

    def __init__(
        self, params: IsentropicWaveParams, adiabatic_index: float, alpha: float
    ):
        self.params = params
        self.adiabatic_index = adiabatic_index
        self.alpha = alpha

    def density(
        self, x: NDArray[np.float64], ell: float
    ) -> NDArray[np.float64]:
        wave = self.params.make_wave_function(ell)
        return self.params.rho_ref * (1.0 + self.alpha * wave(x))

    def pressure(self, rho: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            self.params.p_ref
            * (rho / self.params.rho_ref) ** self.adiabatic_index
        )

    def sound_speed(
        self, rho: NDArray[np.float64] | float, p: NDArray[np.float64] | float
    ) -> NDArray[np.float64]:
        # specific enthalpy h = 1 + epsilon + p/rho, which for an ideal gas closes as
        # h = 1 + (gamma / (gamma - 1)) (p / rho)
        h = 1.0 + self.adiabatic_index * p / (
            rho * (self.adiabatic_index - 1.0)
        )
        return np.sqrt(self.adiabatic_index * p / (rho * h))

    def velocity(
        self, rho: NDArray[np.float64], p: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calculates velocity using the exact Relativistic Riemann Invariant.
        Ref: Zhang & MacFadyen (2006), Section 4.6.
        """
        cs = self.sound_speed(rho, p)
        cs_ref = self.sound_speed(self.params.rho_ref, self.params.p_ref)

        # the relativistic sound speed saturates at sqrt(gamma - 1), so that value is the
        # natural scale for the invariant below.
        sgm1 = np.sqrt(self.adiabatic_index - 1.0)

        # the relativistic riemann invariant along a simple wave:
        # A(cs) = (1 / sqrt(gamma-1)) arctanh(cs / sqrt(gamma-1))
        term_now = (1.0 / sgm1) * np.arctanh(cs / sgm1)
        term_ref = (1.0 / sgm1) * np.arctanh(cs_ref / sgm1)

        # arctanh(v) = term_now - term_ref
        # v = tanh(term_now - term_ref)
        return np.tanh(term_now - term_ref)


class IsentropicRelWave(SimbiProblem):
    """relativistic isentropic pulse in 1d, entropy conserving."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic gas index")
    ]
    alpha: Annotated[
        float, ProblemParam(0.5, cli=True, description="wave amplitude")
    ]

    # domain
    resolution: Annotated[
        int, ProblemParam(1000, cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        Sequence[Sequence[float]],
        ProblemParam([(-0.35, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.RHD, description="physics regime")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(
            CellSpacing.LINEAR, description="grid spacing in x1 direction"
        ),
    ]

    # numerics
    boundary_conditions: Annotated[
        BoundaryCondition,
        ProblemParam(
            BoundaryCondition.PERIODIC, description="boundary conditions"
        ),
    ]

    # simulation control
    end_time: Annotated[
        float,
        ProblemParam(
            1.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if v < ALPHA_MIN or v > ALPHA_MAX:
            raise ValueError(
                f"alpha must be between {ALPHA_MIN} and {ALPHA_MAX}"
            )
        return v

    @property
    def wave_params(self) -> IsentropicWaveParams:
        """physical parameters for isentropic wave."""
        return IsentropicWaveParams()

    @property
    def state(self) -> IsentropicState:
        """state calculator for isentropic wave."""
        return IsentropicState(
            self.wave_params, self.adiabatic_index, self.alpha
        )

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for isentropic wave."""

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            x1 = self.bounds[0][0]
            x2 = self.bounds[0][1]
            dx = (x2 - x1) / nx

            # cell centers sit at (i + 0.5) dx
            x = np.linspace(x1 + 0.5 * dx, x2 - 0.5 * dx, nx)
            ell = 0.3
            rho = self.state.density(x, ell)
            p = self.state.pressure(rho)
            v = self.state.velocity(rho, p)

            for ii in range(nx):
                yield (rho[ii], v[ii], p[ii])

        return gas_state
