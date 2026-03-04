# =============================================================================
# bondi.py
#
# analytic bondi accretion profiles for comparison with simulation data.
# provides density, sound speed, and radial velocity as functions of radius
# using the approximate equilibrium solution from Bondi (1952).
#
# usage:
#   from simbi.analysis.bondi import bondi_profiles
#   profiles = bondi_profiles(r, gamma=5/3, total_mass=1.0, rho_inf=1.0, cs_inf=0.316)
#   rho_analytic = profiles["rho"]
# =============================================================================
import numpy as np


def accretion_coefficient(gamma: float) -> float:
    """bondi accretion eigenvalue lambda(gamma)."""
    if abs(gamma - 1.0) < 1e-5:
        return np.exp(1.5) / 4.0
    if abs(gamma - 5.0 / 3.0) < 1e-5:
        return 0.25
    return float(
        0.25
        * (2.0 / (5.0 - 3.0 * gamma))
        ** ((5.0 - 3.0 * gamma) / (2.0 * (gamma - 1.0)))
    )


def bondi_profiles(
    r: np.ndarray,
    gamma: float,
    total_mass: float = 1.0,
    rho_inf: float = 1.0,
    cs_inf: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    approximate bondi equilibrium profiles.

    args:
        r: radial coordinate array
        gamma: adiabatic index
        total_mass: central mass (or total binary mass)
        rho_inf: ambient density at infinity
        cs_inf: ambient sound speed at infinity

    returns:
        dict with keys "rho", "cs", "vr", each an array matching r.
    """
    r_b = total_mass / cs_inf**2
    rr = r / (2.0 * r_b)
    lam = accretion_coefficient(gamma)

    rho = rho_inf * (1.0 + 0.5 / rr)
    cs = cs_inf * (1.0 + 0.25 * (gamma - 1.0) / rr)
    vr = -0.25 * lam * cs_inf * rr ** (-2.0) * (1.0 - 0.5 / rr)

    return {"rho": rho, "cs": cs, "vr": vr}
