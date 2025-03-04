import numpy as np
from dataclasses import dataclass
from typing import Optional
from ..functional.maybe import Maybe
from numpy.typing import NDArray
from typing import Optional
from .calculations import calculate_state_vector
from ..physics.states.state_vector import StateVector


@dataclass(frozen=True)
class InitialState:
    """Initial state configuration"""

    state: StateVector
    staggered_bfields: Optional[list[NDArray[np.float64]]] = None


def calculate_mean_bfields(
    staggered_bfields: list[NDArray[np.float64]],
) -> Optional[list[NDArray[np.float64]]]:
    # calculate mean B-fields from staggered fields
    if not staggered_bfields:
        return None

    b1, b2, b3 = staggered_bfields
    return [
        0.5 * (b1[..., :-1] + b1[..., 1:]),
        0.5 * (b2[:, :-1, :] + b2[:, 1:, :]),
        0.5 * (b3[:-1, ...] + b3[1:, ...]),
    ]


def construct_conserved_state(
    regime: str, adiabatic_index: float, is_mhd: bool, state: NDArray[np.float64]
) -> Maybe[InitialState]:
    """Pure function to construct continuous state"""
    try:
        # substract off the passive scalar term
        n_non_em = len(state) - 4 if is_mhd else len(state) - 1
        rho, *velocity, pressure = state[:n_non_em]
        staggered_bfields = state[n_non_em:-1] if is_mhd else None
        chi = state[-1]

        state_vector = calculate_state_vector(
            adiabatic_index=adiabatic_index,
            rho=rho,
            velocity=velocity,
            pressure=pressure,
            chi=chi,
            regime=regime,
            bfields=(
                calculate_mean_bfields(staggered_bfields) if staggered_bfields else None
            ),
        )

        return Maybe.of(
            InitialState(
                state=state_vector,
                staggered_bfields=staggered_bfields if staggered_bfields else None,
            )
        )

    except Exception as e:
        return Maybe.save_failure(f"Failed to construct state: {str(e)}")
