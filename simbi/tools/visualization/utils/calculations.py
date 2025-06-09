import numpy as np
from typing import Any, Callable
from ..detail.helpers import (
    calc_cell_volume1D,
    calc_cell_volume2D,
    calc_cell_volume3D,
)
from numpy.typing import NDArray
from astropy import units as u
from astropy import constants as const

Array = NDArray[np.floating[Any]]


def calc_cell_volumes(mesh: dict[str, Array], ndim) -> Array:
    """Calculate cell volumes based on dimension"""
    if ndim == 1:
        return calc_cell_volume1D(mesh["x1v"])
    elif ndim == 2:
        return calc_cell_volume2D(mesh["x1v"], mesh["x2v"])
    elif ndim == 3:
        return calc_cell_volume3D(mesh["x1v"], mesh["x2v"], mesh["x3v"])
    else:
        raise ValueError("ndim must be 1, 2, or 3")


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


def labframe_density(rho: Array, lorentz: Array) -> Array:
    """Calculate the labframe density"""
    return rho * lorentz


def labframe_energy_density(
    rho: Array,
    pre: Array,
    vel: list[Array],
    bfield: list[Array],
    gamma: float,
    regime: str = "classical",
) -> Array:
    """Calculate the labframe energy density"""
    vsq = sum(v**2 for v in vel)
    lorentz = 1.0 / np.sqrt(1.0 - vsq)
    if regime == "classical":
        return pre / (gamma - 1.0) + 0.5 * vsq
    elif regime == "srhd":
        return (
            rho * lorentz**2 * enthalpy(rho, pre, gamma, regime) - pre - rho * lorentz
        )
    elif regime == "srmhd":
        bsq = sum(b**2 for b in bfield)
        return (
            rho * lorentz**2 * enthalpy(rho, pre, gamma, regime)
            - pre
            - rho * lorentz
            + 0.5 * (bsq + vsq * bsq - np.dot(vel, bfield) ** 2)
        )
    else:
        raise NotImplementedError(f"Regime '{regime}' not implemented")


def labframe_momentum(val_map: dict[str, Any], component: int | None = None) -> Array:
    """Calculate the labframe momentum"""
    # if user does not specify a component, return the momentum magnitude
    # otherwise, return the component
    v_vec = val_map["vel"]
    if val_map["regime"] == "classical":
        mom_vec = val_map["rho"] * v_vec
    elif "sr" in val_map["regime"]:
        h = enthalpy(
            val_map["rho"], val_map["p"], val_map["adiabatic_index"], val_map["regime"]
        )
        d = labframe_density(val_map["rho"], val_map["W"])

        if val_map["regime"] == "srmhd":
            bsq = np.sum(val_map["bfields"] ** 2, axis=0)
            vdb = np.dot(v_vec, val_map["bfields"])
            magnetic_part = bsq * val_map["vel"] - vdb * val_map["bfields"]
        else:
            magnetic_part = 0.0

        mom_vec = d * h * val_map["W"] * v_vec + magnetic_part
    else:
        raise NotImplementedError(f"Regime '{val_map['regime']}' not implemented")

    if component is not None:
        return mom_vec[component]
    else:
        return np.sqrt(np.sum(mom_vec**2, axis=0))


def magnetic_pressure(bfields: list[Array]) -> Array:
    """Calculate magnetic pressure"""
    bsq: Array = np.sum([b**2 for b in bfields], axis=0)
    return 0.5 * bsq


def total_pressure(
    pre: Array, bfields: list[Array], regime: str = "classical"
) -> Array:
    """calculate total pressure"""
    if regime == "classical":
        return pre
    else:
        return pre + magnetic_pressure(bfields)


def enthalpy_density(
    rho: Array,
    pre: Array,
    bfields: list[Array],
    adiabatic_index: float,
    regime: str = "classical",
) -> Array:
    if regime == "classical":
        return rho
    elif regime == "srhd":
        return rho * enthalpy(rho, pre, adiabatic_index)
    elif regime == "srmhd":
        return rho * enthalpy(rho, pre, adiabatic_index) + 2.0 * magnetic_pressure(
            bfields
        )
    else:
        raise NotImplementedError(f"Regime '{regime}' not implemented")


def create_field_computer(
    fields: dict[str, Any],
) -> Callable[[str], Array]:
    """Create a memory-efficient field computer for visualization"""
    # Cache commonly accessed fields
    common = {
        "rho": fields["rho"],
        "vel": [vn for n in range(fields["rho"].ndim) for vn in fields[f"v{n}"]],
        "p": fields["p"],
        "bfields": [bn for n in range(fields["rho"].ndim) for bn in fields[f"b{n}"]],
        "W": fields["W"],
        "chi": fields["chi"],
        "adiabatic_index": fields["adiabatic_index"],
        "regime": fields["regime"],
    }

    # Define computation functions
    computations = {
        "D": lambda: labframe_density(common["rho"], common["W"]),
        "m1": lambda: labframe_momentum(common, 0),
        "m2": lambda: labframe_momentum(common, 1),
        "m3": lambda: labframe_momentum(common, 2),
        "energy": lambda: labframe_energy_density(**common),
        "enthalpy": lambda: enthalpy(
            common["rho"], common["p"], common["adiabatic_index"], common["regime"]
        ),
        "total_energy": lambda: labframe_energy_density(**common)
        + labframe_density(common["rho"], common["W"]),
        "chi_dens": lambda: common["chi"]
        * labframe_density(common["rho"], common["W"]),
        "u1": lambda: common["W"] * common["vel"][0],
        "u2": lambda: common["W"] * common["vel"][1],
        "u3": lambda: common["W"] * common["vel"][2],
        "ptot": lambda: total_pressure(
            common["p"], common["bfields"], common["regime"]
        ),
        "pmag": lambda: magnetic_pressure(common["bfields"]),
        "sigma": lambda: 2.0 * magnetic_pressure(common["bfields"]) / common["rho"],
        "enthalpy_density": lambda: enthalpy_density(
            common["rho"],
            common["p"],
            common["bfields"],
            common["adiabatic_index"],
            common["regime"],
        ),
    }

    # Return compute function
    def compute(var, indices=None):
        if var not in computations:
            raise NotImplementedError(f"Field '{var}' not implemented")
        result = computations[var]()
        if indices is not None:
            return result[indices]
        return result

    return compute
