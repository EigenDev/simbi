# =============================================================================
# test_spn98.py
#
# regression tests for the sari, piran & narayan (1998) analytic afterglow
# benchmark (spn98.py). these pin the closed-form spectral + temporal power-law
# indices and the break ordering, so the analytic reference the numerical light
# curve is validated against is itself trustworthy.
# =============================================================================

import numpy as np
import pytest
from astropy import units as u

from simbi.afterglow.spn98 import spn98_breaks, spn98_flux, spn98_lightcurve

# a canonical slow-cooling afterglow at fiducial parameters.
_E = 1.0e52 * u.erg
_N = 1.0 / u.cm**3
_EPS_E = 0.1
_EPS_B = 0.01
_P = 2.5
_DL = 1.0e28 * u.cm


def _loglog_slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def test_breaks_slow_cooling_ordering_and_units():
    br = spn98_breaks(1.0 * u.day, _E, _N, _EPS_E, _EPS_B, _P, _DL)
    assert br.cooling == "slow"  # nu_m < nu_c at t = 1 day for these params
    assert br.nu_m < br.nu_c
    assert br.f_nu_max.unit.is_equivalent(u.mJy)
    # eq-11 fiducial values at t = 1 day.
    assert br.nu_m.to_value(u.Hz) == pytest.approx(5.7e11, rel=1e-3)
    assert br.nu_c.to_value(u.Hz) == pytest.approx(2.7e15, rel=1e-3)
    assert br.f_nu_max.to_value(u.mJy) == pytest.approx(11.0, rel=1e-3)


def test_spectral_index_mid_segment():
    # nu_m < nu < nu_c slow-cooling spectral slope is -(p-1)/2.
    br = spn98_breaks(1.0 * u.day, _E, _N, _EPS_E, _EPS_B, _P, _DL)
    nus = np.geomspace(br.nu_m.value * 3, br.nu_c.value / 3, 12) * u.Hz
    f = spn98_flux(nus, br, _P).to_value(u.mJy)
    assert _loglog_slope(nus.to_value(u.Hz), f) == pytest.approx(-(_P - 1) / 2, abs=1e-6)


def test_spectral_index_above_cooling():
    # nu > nu_c slow-cooling spectral slope steepens to -p/2.
    br = spn98_breaks(1.0 * u.day, _E, _N, _EPS_E, _EPS_B, _P, _DL)
    nus = np.geomspace(br.nu_c.value * 3, br.nu_c.value * 300, 12) * u.Hz
    f = spn98_flux(nus, br, _P).to_value(u.mJy)
    assert _loglog_slope(nus.to_value(u.Hz), f) == pytest.approx(-_P / 2, abs=1e-6)


def test_temporal_index_mid_segment():
    # at a fixed frequency in nu_m < nu < nu_c the light curve decays as t^-3(p-1)/4.
    nu = 1.0e15 * u.Hz
    ts = np.geomspace(0.5, 5.0, 10) * u.day
    f = spn98_lightcurve(nu, ts, _E, _N, _EPS_E, _EPS_B, _P, _DL).to_value(u.mJy)
    assert _loglog_slope(ts.to_value(u.day), f) == pytest.approx(
        -3.0 * (_P - 1) / 4.0, abs=1e-6
    )


def test_flux_scales_with_distance_inverse_square():
    # F_nu,max ~ D^-2.
    br1 = spn98_breaks(1.0 * u.day, _E, _N, _EPS_E, _EPS_B, _P, 1.0e28 * u.cm)
    br2 = spn98_breaks(1.0 * u.day, _E, _N, _EPS_E, _EPS_B, _P, 2.0e28 * u.cm)
    ratio = (br1.f_nu_max / br2.f_nu_max).to_value(u.one)
    assert ratio == pytest.approx(4.0, rel=1e-6)
