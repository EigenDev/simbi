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


def get_padded_mean_bfields(
    staggered_bfields: list[NDArray[np.float64]],
    shape: tuple[int, int, int] = None,
) -> Optional[list[NDArray[np.float64]]]:
    """return the padded mean magnetic fields whose shape is the same as the gas variables"""
    if not staggered_bfields:
        return None

    mean_bfields = calculate_mean_bfields(staggered_bfields)
    # the pad widths are the same in each direction, so we only need to compute
    # the half integer distance from one axis
    distance = (shape[0] - mean_bfields[0].shape[0]) // 2
    mean_bfields = [np.pad(b, distance, mode="edge") for b in mean_bfields]
    return mean_bfields


def pad_staggered_fields(
    bfields: list[NDArray[np.float64]],
) -> Optional[list[NDArray[np.float64]]]:
    """pad the staggered fields along perpendicular directions"""
    if not bfields:
        return None
    b1, b2, b3 = bfields
    b1 = np.pad(b1, ((1, 1), (1, 1), (0, 0)), "edge")
    b2 = np.pad(b2, ((1, 1), (0, 0), (1, 1)), "edge")
    b3 = np.pad(b3, ((0, 0), (1, 1), (1, 1)), "edge")
    return [b1, b2, b3]


def construct_conserved_state(
    regime: str,
    adiabatic_index: float,
    prims_and_fields: tuple[NDArray[np.float64], list[NDArray[np.float64]]],
) -> Maybe[InitialState]:
    """Pure function to construct continuous state"""
    try:
        staggered_bfields = prims_and_fields[1]
        prims = prims_and_fields[0]
        pure_hydro = not staggered_bfields
        # substract off the passive scalar term
        ngas = len(prims) - 1 if pure_hydro else 5
        rho, *velocity, pressure = prims[:ngas]
        chi = prims[-1]

        state_vector = calculate_state_vector(
            adiabatic_index=adiabatic_index,
            rho=rho,
            velocity=velocity,
            pressure=pressure,
            chi=chi,
            regime=regime,
            bfields=get_padded_mean_bfields(staggered_bfields, shape=rho.shape),
        )

        return Maybe.of(
            InitialState(
                state=state_vector,
                staggered_bfields=pad_staggered_fields(staggered_bfields),
            )
        )

    except Exception as e:
        return Maybe.save_failure(f"Failed to construct state: {str(e)}")
