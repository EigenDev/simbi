import numpy as np
from numpy.typing import NDArray
from typing import Any, Sequence
from enum import Enum, IntEnum
from ..physics import StateVector

Array = NDArray[np.floating[Any]]
nested_array = Sequence[Array]


class VectorComponent(IntEnum):
    X1 = 0
    X2 = 1
    X3 = 2


class VectorMode(Enum):
    Magnitude = "magnitude"
    All = "all"


def calculate_state_vector(
    adiabatic_index: float,
    rho: Array,
    velocity: nested_array,
    pressure: Array,
    chi: Array,
    regime: str,
    bfields: nested_array,
) -> StateVector:
    """Pure function to calculate state vector"""
    try:
        validate_eos(adiabatic_index, regime)
        dens = labframe_density(rho, velocity, regime)
        mom = labframe_momentum(
            rho,
            pressure,
            velocity,
            [] if bfields is None else bfields,
            adiabatic_index,
            regime,
        )

        energy = labframe_energy(
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
            mean_magnetic_field=bfields,
        )
    except Exception as e:
        raise ValueError(f"Error calculating state vector: {e}")


def elemental_multiply(a: nested_array, b: nested_array) -> Any:
    return np.array([a[i] * b[i] for i in range(len(a))])


def dot_product(
    a: nested_array,
    b: nested_array,
) -> Any:
    return np.sum([a[i] * b[i] for i in range(len(a))], axis=0)


def lorentz_factor(
    velocity: nested_array, regime: str, using_gamma_beta: bool = False
) -> Array | float:
    vsquared = dot_product(velocity, velocity)
    if regime != "classical" and np.any(vsquared >= 1.0):
        raise ValueError("Lorentz factor is not real. Velocity exceeds speed of light.")

    if regime == "classical":
        return 1.0
    elif not using_gamma_beta:
        return np.asarray(1.0 / np.sqrt(1.0 - vsquared))
    else:
        return np.asarray(np.sqrt(1.0 + vsquared))


def four_velocity(velocity: nested_array, regime: str, component: int) -> Array:
    lorentz = lorentz_factor(velocity, regime)
    return np.asarray(velocity[component] * lorentz)


def spec_enthalpy(
    adiabatic_index: float,
    rho: Array,
    pressure: Array,
    regime: str,
) -> Array | float:
    if regime == "classical":
        if adiabatic_index == 1.0:
            # Isothermal case - pressure = cs^2 * rho
            # where cs is the isothermal sound speed
            return 1.0 + pressure / rho
        else:
            return 1.0
    else:
        # Adiabatic case
        return 1.0 + adiabatic_index * pressure / (rho * (adiabatic_index - 1.0))


def labframe_density(rho: Array, velocity: nested_array, regime: str) -> Array:
    return rho * lorentz_factor(velocity, regime)


def labframe_energy(
    adiabatic_index: float,
    rho: Array,
    pressure: Array,
    velocity: nested_array,
    bfields: nested_array,
    regime: str,
) -> Array:
    if adiabatic_index == 1.0:
        # this wll be a proxy for the sound speed squared
        return pressure / rho
    res: Array
    bsq = dot_product(bfields, bfields) if np.any(bfields) else 0.0
    vdb = dot_product(velocity, bfields) if np.any(bfields) else 0.0
    vsq = dot_product(velocity, velocity)
    lorentz = lorentz_factor(velocity, regime)
    enthalpy = spec_enthalpy(adiabatic_index, rho, pressure, regime)

    if regime == "classical":
        if adiabatic_index == 1.0:
            # Isothermal case - internal energy term not needed
            res = 0.5 * rho * vsq + 0.5 * bsq
        else:
            # Adiabatic case
            res = pressure / (adiabatic_index - 1.0) + 0.5 * rho * vsq + 0.5 * bsq
    else:
        # Relativistic case - isothermal not allowed
        if adiabatic_index == 1.0:
            raise ValueError(
                "Isothermal EOS (gamma=1) is not physically valid for relativistic flows"
            )
        enthalpy = spec_enthalpy(adiabatic_index, rho, pressure, regime)
        res = (
            rho * lorentz**2 * enthalpy
            - pressure
            - rho * lorentz
            + 0.5 * bsq
            + 0.5 * (bsq * vsq - vdb**2)
        )

    return res


def enthalpy(
    rho: Array,
    pre: Array,
    gamma: float,
    regime: str = "classical",
) -> Array | float:
    """Calculate the enthalpy per particle"""
    if regime == "classical":
        return 1.0
    return 1.0 + gamma * pre / (rho * (gamma - 1.0))


def labframe_energy_density(
    rho: Array,
    pre: Array,
    vel: Sequence[Array],
    bfield: Sequence[Array],
    gamma: float,
    regime: str = "classical",
) -> Array:
    """Calculate the labframe energy density"""
    vsq = sum(v**2 for v in vel)
    if regime == "classical":
        if gamma == 1.0:
            # for the isothermal case,
            # we store the sound speed squared in the
            # energy density
            return pre / rho
        return pre / (gamma - 1.0) + 0.5 * rho * vsq
    elif regime == "srhd":
        lorentz = 1.0 / np.sqrt(1.0 - vsq)
        return np.asarray(
            rho * lorentz**2 * enthalpy(rho, pre, gamma, regime) - pre - rho * lorentz
        )
    elif regime == "srmhd":
        lorentz = 1.0 / np.sqrt(1.0 - vsq)
        bsq = sum(b**2 for b in bfield)
        return np.asarray(
            rho * lorentz**2 * enthalpy(rho, pre, gamma, regime)
            - pre
            - rho * lorentz
            + 0.5 * (bsq**2 + vsq * bsq - dot_product(vel, bfield) ** 2)
        )
    else:
        raise NotImplementedError(f"Regime '{regime}' not implemented")


def labframe_momentum(
    rho: Array,
    pre: Array,
    vel: Sequence[Array],
    bfield: Sequence[Array],
    gamma: float,
    regime: str = "classical",
    mode: VectorMode | VectorComponent = VectorMode.All,
) -> Array:
    """Calculate the labframe momentum"""
    # if user does not specify a component, return the momentum magnitude
    # otherwise, return the component
    if regime == "classical":
        mom_vec = rho * vel
    elif "sr" in regime:
        h = enthalpy(
            rho,
            pre,
            gamma,
            regime,
        )
        d = labframe_density(rho, vel, regime)

        if regime == "srmhd":
            bsq = np.array((sum(b**2 for b in bfield)), dtype=float)
            vdb = dot_product(vel, bfield)
            magnetic_part = bsq * vel - vdb * bfield
        else:
            magnetic_part = 0.0

        mom_vec = d * h * lorentz_factor(vel, regime) * vel + magnetic_part
    else:
        raise NotImplementedError(f"Regime '{regime}' not implemented")

    if mode == VectorMode.Magnitude:
        return np.asarray(np.sqrt(np.sum(mom_vec**2, axis=0)))
    elif mode == VectorMode.All:
        return mom_vec
    else:
        return np.asarray(mom_vec[int(mode.value)])


def magnetic_pressure(
    bfields: Sequence[Array], velocity: Sequence[Array], regime: str
) -> Array:
    """Calculate magnetic pressure"""
    bsq: Array = np.sum([b**2 for b in bfields], axis=0)
    if regime == "srmhd":
        lorentz_factor = 1.0 / np.sqrt(1.0 - dot_product(velocity, velocity))
        vdb = dot_product(velocity, bfields)
        bsq = bsq / lorentz_factor**2 + vdb**2
    return 0.5 * bsq


def total_pressure(
    pre: Array, bfields: Sequence[Array], velocity: Sequence[Array], regime: str
) -> Array:
    """calculate total pressure"""
    if "mhd" not in regime:
        return pre
    else:
        return pre + magnetic_pressure(bfields, velocity, regime)


def enthalpy_density(
    rho: Array,
    pre: Array,
    bfields: Sequence[Array],
    velocity: Sequence[Array],
    adiabatic_index: float,
    regime: str = "classical",
) -> Array:
    if regime == "classical":
        return rho
    elif regime == "srhd":
        return rho * enthalpy(rho, pre, adiabatic_index)
    elif regime == "srmhd":
        return rho * enthalpy(rho, pre, adiabatic_index) + 2.0 * magnetic_pressure(
            bfields,
            velocity,
            regime,
        )
    else:
        raise NotImplementedError(f"Regime '{regime}' not implemented")


def magnetization(rho: Array, bfields: Sequence[Array]) -> Array:
    """Calculate magnetization"""
    return np.asarray(dot_product(bfields, bfields) / rho)


def is_isothermal(adiabatic_index: float) -> bool:
    """Check if simulation is isothermal"""
    return abs(adiabatic_index - 1.0) < 1e-10


def validate_eos(adiabatic_index: float, regime: str) -> None:
    """Validate equation of state is physically consistent"""
    if is_isothermal(adiabatic_index) and regime != "classical":
        raise ValueError(
            "Isothermal equation of state (gamma=1) is only valid for classical flows"
        )
