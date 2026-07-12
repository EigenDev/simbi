# =============================================================================
# test_gr_accretion.py
#
# the reducer's correctness gate: construct a Schwarzschild-KS radial inflow whose
# rest-mass flux is CONSERVED (Mdot exactly constant in r by construction), feed the
# reducer the PHYSICAL velocity the substrate would store, and assert it recovers a
# single r_ex-independent Mdot. this exercises the V^rhat -> u^r conversion (the
# sqrt(h) factor + the shift), sqrt(-g) = r^2 sin(theta), and the shell reduction.
# =============================================================================

import numpy as np
import pytest

from simbi.reader.gr_accretion import accretion_rate, rex_invariance


def _conserved_ks_inflow(r, mass, mdot, v_r_coord):
    """a Schwarzschild-KS radial inflow with EXACTLY constant rest-mass rate `mdot`.
    returns (rho, V_rhat) — the density and the PHYSICAL radial velocity the checkpoint
    stores. `v_r_coord` is the (subluminal) coordinate radial velocity < 0."""
    h = 1.0 + 2.0 * mass / r
    sqrt_h = np.sqrt(h)
    alpha = 1.0 / sqrt_h
    beta_r = 2.0 * mass / (r + 2.0 * mass)
    w = 1.0 / np.sqrt(1.0 - h * v_r_coord**2)  # W = 1/sqrt(1 - gamma_rr (v^r)^2)
    u_r = w * (v_r_coord - beta_r / alpha)
    rho = -mdot / (4.0 * np.pi * r * r * u_r)  # from -4 pi r^2 rho u^r = mdot
    v_rhat = sqrt_h * v_r_coord  # the stored physical (orthonormal) velocity
    return rho, v_rhat


def test_reducer_recovers_a_conserved_ks_inflow_1d():
    mass = 1.0
    r = np.array([3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0])
    mdot_true = 0.37
    v_r = -0.25 * np.ones_like(r)  # subluminal outside r_+: h (v^r)^2 < 1
    rho, v_rhat = _conserved_ks_inflow(r, mass, mdot_true, v_r)

    mdot = accretion_rate(rho, [v_rhat], r, None, mass)
    assert np.allclose(mdot, mdot_true, rtol=1e-12), mdot

    cert = rex_invariance(mdot, r, [3, 5, 10, 20])
    assert cert["relative_spread"] < 1e-12
    assert abs(cert["mean"] - mdot_true) < 1e-12


def test_reducer_is_rex_invariant_2d_axisymmetric():
    # a uniform-in-theta conserved flow: the reducer's r_ex-invariance must be EXACT
    # (the theta quadrature is identical at every radius, so it cancels in the spread).
    mass = 1.0
    r = np.array([3.0, 5.0, 10.0, 20.0])
    n_theta = 64
    theta_faces = np.linspace(0.0, np.pi, n_theta + 1)
    theta = 0.5 * (theta_faces[:-1] + theta_faces[1:])
    dtheta = np.pi / n_theta

    mdot_true = 0.37
    v_r = -0.25 * np.ones_like(r)
    rho1d, v1d = _conserved_ks_inflow(r, mass, mdot_true, v_r)
    rho = np.repeat(rho1d[:, None], n_theta, axis=1)
    v_rhat = np.repeat(v1d[:, None], n_theta, axis=1)

    mdot = accretion_rate(rho, [v_rhat], r, theta, mass, dtheta=dtheta, dphi=2.0 * np.pi)
    cert = rex_invariance(mdot, r, [3, 5, 10, 20])
    # the certificate: Mdot independent of r_ex to roundoff.
    assert cert["relative_spread"] < 1e-12
    # the absolute value matches within the theta midpoint-quadrature error.
    assert abs(cert["mean"] - mdot_true) < 1e-3


def test_superluminal_velocity_is_rejected():
    # h (v^r)^2 >= 1 must fail loud, not silently produce a complex/NaN flux.
    mass = 1.0
    r = np.array([3.0])
    v_rhat = np.array([1.5])  # physical |V| > 1
    with pytest.raises(ValueError):
        accretion_rate(np.array([1.0]), [v_rhat], r, None, mass)
