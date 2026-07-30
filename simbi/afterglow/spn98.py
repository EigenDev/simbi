# =============================================================================
# spn98.py
#
# the sari, piran & narayan (1998) analytic synchrotron afterglow: the closed-form
# spectrum + light curve of a decelerating adiabatic relativistic blast wave in a
# uniform medium. this is the ANALYTIC BENCHMARK the numerical (catalog-integrated)
# light curve is validated against -- it pins both the temporal slope and the
# absolute normalization of F_nu(nu, t).
#
# all quantities carry astropy units so the dimensional bookkeeping is checked, not
# assumed. inputs are taken in cgs/gaussian; F_nu is returned in mJy.
#
# reference: sari, piran & narayan 1998, apj 497, l17 (eqs 7, 8, 11). slow cooling
# is nu_m < nu_c (late times); fast cooling is nu_c < nu_m (early times). this is a
# LOCAL-source benchmark (luminosity distance d_l, no cosmological convolution); run
# the comparison at z ~ 0 so SPN98's source-frame relations apply directly.
#
# usage:
#  from astropy import units as u
#  br = spn98_breaks(t=1.0 * u.day, E=1e52 * u.erg, n=1.0 / u.cm**3,
#                    eps_e=0.1, eps_b=0.01, p=2.5)
#  f = spn98_flux(nu=3e9 * u.Hz, breaks=br, d_l=1e26 * u.cm)   # -> mJy Quantity
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u

# the eq-11 prefactors are quoted for these fiducial scalings; keep them explicit so
# the formula reads exactly as printed in the paper.
_E52 = 1.0e52 * u.erg
_D28 = 1.0e28 * u.cm


@dataclass(frozen=True)
class Spn98Breaks:
    """the three numbers that define an instantaneous SPN98 spectrum at one epoch:
    the cooling break nu_c, the injection break nu_m, and the peak flux F_nu_max."""

    nu_c: u.Quantity  # cooling-break frequency [Hz]
    nu_m: u.Quantity  # injection-break frequency [Hz]
    f_nu_max: u.Quantity  # peak flux density [mJy]
    cooling: str  # "slow" (nu_m < nu_c) or "fast" (nu_c < nu_m)


def spn98_breaks(
    t: u.Quantity,
    E: u.Quantity,
    n: u.Quantity,
    eps_e: float,
    eps_b: float,
    p: float = 2.5,
    d_l: u.Quantity = _D28,
) -> Spn98Breaks:
    """the break frequencies + peak flux for an ADIABATIC blast (SPN98 eq 11).

    nu_c     = 2.7e12 eps_b^-3/2 E_52^-1/2 n_1^-1 t_d^-1/2     Hz
    nu_m     = 5.7e14 eps_b^1/2  eps_e^2    E_52^1/2 t_d^-3/2   Hz
    F_nu,max = 1.1e5  eps_b^1/2  E_52       n_1^1/2  D_28^-2    uJy
    """
    t_d = t.to_value(u.day)
    e52 = (E / _E52).to_value(u.one)
    n1 = n.to_value(u.cm**-3)
    d28 = (d_l / _D28).to_value(u.one)

    nu_c = (
        2.7e12 * eps_b**-1.5 * e52**-0.5 * n1**-1.0 * t_d**-0.5
    ) * u.Hz
    nu_m = (
        5.7e14 * eps_b**0.5 * eps_e**2.0 * e52**0.5 * t_d**-1.5
    ) * u.Hz
    f_nu_max = (
        1.1e5 * eps_b**0.5 * e52 * n1**0.5 * d28**-2.0
    ) * u.uJy

    cooling = "slow" if nu_m < nu_c else "fast"
    return Spn98Breaks(
        nu_c=nu_c, nu_m=nu_m, f_nu_max=f_nu_max.to(u.mJy), cooling=cooling
    )


def spn98_flux(nu: u.Quantity, breaks: Spn98Breaks, p: float = 2.5) -> u.Quantity:
    """the SPN98 flux density F_nu at observed frequency `nu` (eqs 7-8), as a broken
    power law through the breaks. returns an astropy Quantity in mJy. accepts scalar
    or array `nu`."""
    x_c = (nu / breaks.nu_c).to_value(u.one)
    x_m = (nu / breaks.nu_m).to_value(u.one)
    ratio_cm = (breaks.nu_c / breaks.nu_m).to_value(u.one)
    ratio_mc = (breaks.nu_m / breaks.nu_c).to_value(u.one)

    if breaks.cooling == "slow":
        # nu_m < nu_c: nu^1/3 (nu<nu_m), nu^-(p-1)/2 (nu_m<nu<nu_c), nu^-p/2 (nu>nu_c).
        shape = np.where(
            x_m < 1.0,
            x_m ** (1.0 / 3.0),
            np.where(
                x_c < 1.0,
                x_m ** (-0.5 * (p - 1.0)),
                ratio_cm ** (-0.5 * (p - 1.0)) * x_c ** (-0.5 * p),
            ),
        )
    else:
        # nu_c < nu_m: nu^1/3 (nu<nu_c), nu^-1/2 (nu_c<nu<nu_m), nu^-p/2 (nu>nu_m).
        shape = np.where(
            x_c < 1.0,
            x_c ** (1.0 / 3.0),
            np.where(
                x_m < 1.0,
                x_c ** (-0.5),
                ratio_mc ** (-0.5) * x_m ** (-0.5 * p),
            ),
        )
    return breaks.f_nu_max * shape


def spn98_lightcurve(
    nu: u.Quantity,
    times: u.Quantity,
    E: u.Quantity,
    n: u.Quantity,
    eps_e: float,
    eps_b: float,
    p: float = 2.5,
    d_l: u.Quantity = _D28,
) -> u.Quantity:
    """F_nu(t) at a fixed observed frequency over `times` (an array Quantity), assembling
    the instantaneous SPN98 spectrum at each epoch. returns mJy. the temporal slope in
    the nu_m < nu < nu_c segment is F_nu ~ t^-3(p-1)/4; above nu_c it is t^-(3p-2)/4."""
    out = np.empty(times.shape) * u.mJy
    for ii, t in enumerate(times):
        br = spn98_breaks(t, E, n, eps_e, eps_b, p, d_l)
        out[ii] = spn98_flux(nu, br, p)
    return out
