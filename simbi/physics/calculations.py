import numpy as np
from numpy.typing import NDArray
from ..functional.maybe import Maybe
from typing import Any, Optional, Sequence
from ..physics import StateVector

nested_array = NDArray[np.float64] | list[NDArray[np.float64]] | list[float]


def calculate_state_vector(
    adiabatic_index: float,
    rho: NDArray[np.float64],
    velocity: Sequence[NDArray[np.float64]],
    pressure: NDArray[np.float64],
    chi: NDArray[np.float64],
    regime: str,
    bfields: Optional[Sequence[NDArray[np.float64]]] = None,
) -> StateVector:
    """Pure function to calculate state vector"""
    try:
        dens = calc_labframe_density(rho, velocity, regime)
        mom = calc_labframe_momentum(
            adiabatic_index,
            rho,
            velocity,
            pressure,
            [] if bfields is None else bfields,
            regime,
        )
        energy = calc_labframe_energy(
            adiabatic_index,
            rho,
            pressure,
            velocity,
            [] if bfields is None else bfields,
            regime,
        )

        return StateVector(
            density=dens,
            momentum=list(mom),
            energy=energy,
            rho_chi=dens * chi,
            mean_magnetic_field=list(bfields) if bfields is not None else None,
        )
    except Exception as e:
        print(f"Failed to calculate state vector: {e}")


def validate_state_vector(state: StateVector) -> Maybe[StateVector]:
    """Validate state vector components"""
    if np.any(np.isnan(state.density)):
        return Maybe.save_failure("Density contains NaN values")
    if any(np.any(np.isnan(m)) for m in state.momentum):
        return Maybe.save_failure("Momentum contains NaN values")
    if np.any(np.isnan(state.energy)):
        return Maybe.save_failure("Energy contains NaN values")
    if state.magnetic_field and any(np.any(np.isnan(b)) for b in state.magnetic_field):
        return Maybe.save_failure("Magnetic field contains NaN values")
    return Maybe.of(state)


def elemental_multiply(a: nested_array, b: nested_array) -> Any:
    if isinstance(a, list):
        return np.array([a[i] * b[i] for i in range(len(a))])
    return a * b


def dot_product(
    a: nested_array,
    b: nested_array,
) -> Any:
    if isinstance(a, list):
        return np.sum([a[i] * b[i] for i in range(len(a))], axis=0)
    return np.sum(a * b, axis=0)


def calc_lorentz_factor(velocity: nested_array, regime: str) -> NDArray[np.float64]:
    vsquared = dot_product(velocity, velocity)
    if regime != "classical" and np.any(vsquared >= 1.0):
        raise ValueError("Lorentz factor is not real. Velocity exceeds speed of light.")
    return 1.0 if regime == "classical" else (1.0 - vsquared) ** (-0.5)


def calc_spec_enthalpy(
    adiabatic_index: float,
    rho: NDArray[np.float64],
    pressure: NDArray[np.float64],
    regime: str,
) -> NDArray[np.float64]:
    return (
        1.0
        if regime == "classical"
        else 1.0 + adiabatic_index * pressure / (rho * (adiabatic_index - 1.0))
    )


def calc_labframe_density(
    rho: NDArray[np.float64], velocity: nested_array, regime: str
) -> NDArray[np.float64]:
    return rho * calc_lorentz_factor(velocity, regime)


def calc_labframe_momentum(
    adiabatic_index: float,
    rho: NDArray[np.float64],
    velocity: nested_array,
    pressure: NDArray[np.float64],
    bfields: nested_array,
    regime: str,
) -> NDArray[np.float64]:
    vdb = dot_product(velocity, bfields) if np.any(bfields) else 0.0
    bsq = dot_product(bfields, bfields) if np.any(bfields) else 0.0
    vdb_bvec = (
        np.array([bn * vdb for bn in bfields])
        if np.any(bfields)
        else [0.0] * len(velocity)
    )

    enthalpy = calc_spec_enthalpy(adiabatic_index, rho, pressure, regime)
    lorentz = calc_lorentz_factor(velocity, regime)
    return np.array(
        [
            (rho * lorentz**2 * enthalpy + bsq) * velocity[i] - vdb_bvec[i]
            for i in range(len(velocity))
        ]
    )


def calc_labframe_energy(
    adiabatic_index: float,
    rho: NDArray[np.float64],
    pressure: NDArray[np.float64],
    velocity: nested_array,
    bfields: nested_array,
    regime: str,
) -> NDArray[np.float64]:
    res: NDArray[np.float64]
    bsq = dot_product(bfields, bfields) if np.any(bfields) else 0.0
    vdb = dot_product(velocity, bfields) if np.any(bfields) else 0.0
    vsq = dot_product(velocity, velocity)
    lorentz = calc_lorentz_factor(velocity, regime)
    enthalpy = calc_spec_enthalpy(adiabatic_index, rho, pressure, regime)

    if regime == "classical":
        res = pressure / (adiabatic_index - 1.0) + 0.5 * rho * vsq + 0.5 * bsq
    else:
        res = (
            rho * lorentz**2 * enthalpy
            - pressure
            - rho * lorentz
            + 0.5 * bsq
            + 0.5 * (bsq * vsq - vdb**2)
        )

    return res
