# =============================================================================
# test_gr_accretion.py
#
# the reducer's correctness gates:
#   1. constructed conserved KS inflow (contravariant v^r, Mdot constant in r by
#      construction) -> the reducer recovers a single r_ex-independent Mdot. exercises
#      the W = 1/sqrt(1 - gamma_rr (v^r)^2) contraction, the beta^r/alpha shift,
#      sqrt(-g) = r^2 sin(theta), and the shell reduction.
#   2. the EXACT michel (1972) transonic solution on schwarzschild -> the reducer
#      reproduces the analytic Mdot = 4 pi jm across the sonic point. this is the
#      convention gate: it fails loudly if the stored velocity is mis-interpreted
#      (orthonormal vs contravariant), which only shows up near the horizon.
# =============================================================================

import math

import numpy as np
import pytest

from simbi.reader.gr_accretion import (
    accretion_from_checkpoint,
    accretion_rate,
    rex_invariance,
    shell_accretion,
)


def _conserved_ks_inflow(r, mass, mdot, v_r):
    """a Schwarzschild-KS radial inflow with EXACTLY constant rest-mass rate `mdot`.
    `v_r` is the (subluminal, negative) CONTRAVARIANT valencia radial velocity; returns
    (rho, v_r) with rho chosen so -4 pi r^2 rho u^r = mdot at every radius."""
    h = 1.0 + 2.0 * mass / r
    beta_over_alpha = (2.0 * mass / (r + 2.0 * mass)) * np.sqrt(h)
    w = 1.0 / np.sqrt(1.0 - h * v_r**2)  # gamma_rr = h in KS
    u_r = w * (v_r - beta_over_alpha)
    rho = -mdot / (4.0 * np.pi * r * r * u_r)
    return rho, v_r


def test_reducer_recovers_a_conserved_ks_inflow_1d():
    mass = 1.0
    r = np.array([3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0])
    mdot_true = 0.37
    v_r = -0.20 * np.ones_like(r)  # contravariant; subluminal: h (v^r)^2 < 1
    rho, v_r = _conserved_ks_inflow(r, mass, mdot_true, v_r)

    mdot = accretion_rate(rho, [v_r], r, None, mass, spacetime="kerr_schild")
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
    v_r = -0.20 * np.ones_like(r)
    rho1d, v1d = _conserved_ks_inflow(r, mass, mdot_true, v_r)
    rho = np.repeat(rho1d[:, None], n_theta, axis=1)
    v_r2d = np.repeat(v1d[:, None], n_theta, axis=1)

    mdot = accretion_rate(
        rho, [v_r2d], r, theta, mass, spacetime="kerr_schild", dtheta=dtheta, dphi=2.0 * np.pi
    )
    cert = rex_invariance(mdot, r, [3, 5, 10, 20])
    assert cert["relative_spread"] < 1e-12
    assert abs(cert["mean"] - mdot_true) < 1e-3  # theta midpoint-quadrature error


def test_reducer_reproduces_analytic_michel_mdot():
    # the convention gate: feed the reducer the EXACT michel primitives (the stored
    # contravariant v^r) on schwarzschild and require the analytic Mdot = 4 pi jm at
    # every radius, including inside the sonic point where the metric is far from flat.
    from simbi_configs.examples.grhd.gr_michel import MichelSolution

    sol = MichelSolution(mass=1.0, gamma=4.0 / 3.0, rho_inf=1.0, p_inf=1.0e-2)
    r = np.array([3.0, 5.0, 10.0, sol.r_sonic, 40.0, 100.0])
    rho = np.empty_like(r)
    v_r = np.empty_like(r)
    for ii, rr in enumerate(r):
        rho_i, v1_i, _ = sol.primitive(rr)
        rho[ii] = rho_i
        v_r[ii] = v1_i  # valencia contravariant radial velocity, negative (inflow)

    mdot = accretion_rate(rho, [v_r], r, None, mass=1.0, spacetime="schwarzschild")
    mdot_analytic = 4.0 * np.pi * sol.jm
    assert np.allclose(mdot, mdot_analytic, rtol=1e-10), (mdot, mdot_analytic)

    cert = rex_invariance(mdot, r, [3.0, 10.0, sol.r_sonic, 100.0])
    assert cert["relative_spread"] < 1e-10
    assert math.isclose(cert["mean"], mdot_analytic, rel_tol=1e-10)


def _log_vertices(rmin, rmax, n):
    return np.logspace(np.log10(rmin), np.log10(rmax), n + 1)


def _radial_centroids(x1v):
    rl, rh = x1v[:-1], x1v[1:]
    return 0.75 * (rh**4 - rl**4) / (rh**3 - rl**3)


def test_shell_accretion_recovers_conserved_flow_on_a_log_grid_1d():
    # a real log-spaced radial vertex grid; the flow is conserved at the SAME
    # volume-weighted centroids the reducer samples, so recovery is exact.
    mass = 1.0
    x1v = _log_vertices(2.5, 50.0, 96)
    r = _radial_centroids(x1v)
    mdot_true = 0.42
    v_r = -0.18 * np.ones_like(r)
    rho, v_r = _conserved_ks_inflow(r, mass, mdot_true, v_r)

    mdot, cert = shell_accretion(rho, [v_r], x1v, None, mass, "kerr_schild")
    assert np.allclose(mdot, mdot_true, rtol=1e-12)
    assert cert["relative_spread"] < 1e-12


def test_shell_accretion_is_rex_invariant_2d_storage_order():
    # 2D storage order is (ntheta, nr); a uniform-in-theta conserved flow must reduce
    # to an r_ex-invariant Mdot after the transpose to reducer order.
    mass = 1.0
    x1v = _log_vertices(2.5, 50.0, 48)
    r = _radial_centroids(x1v)
    n_theta = 40
    x2v = np.linspace(0.0, np.pi, n_theta + 1)
    mdot_true = 0.42
    v_r = -0.18 * np.ones_like(r)
    rho1d, v1d = _conserved_ks_inflow(r, mass, mdot_true, v_r)
    rho = np.repeat(rho1d[None, :], n_theta, axis=0)  # (ntheta, nr)
    v_r2d = np.repeat(v1d[None, :], n_theta, axis=0)

    mdot, cert = shell_accretion(rho, [v_r2d], x1v, x2v, mass, "kerr_schild")
    assert cert["relative_spread"] < 1e-12
    assert abs(cert["mean"] - mdot_true) < 1e-3


class _FakeMesh:
    def __init__(self, x1v, x2v=None):
        self.x1v = x1v
        self.x2v = x2v if x2v is not None else np.array([0.0, np.pi])


class _FakeMeta:
    def __init__(self, spacetime, mass):
        self.spacetime = spacetime
        self.schwarzschild_mass = mass


class _FakeData:
    """a duck-typed SimData exposing only what the wrapper reads."""

    def __init__(self, fields, mesh, meta):
        self._fields = fields
        self.mesh = mesh
        self.metadata = meta

    def get_field(self, name):
        return self._fields[name]

    def available_fields(self):
        return set(self._fields)


def test_accretion_from_checkpoint_uses_self_describing_metadata():
    mass = 1.0
    x1v = _log_vertices(2.5, 50.0, 64)
    r = _radial_centroids(x1v)
    mdot_true = 0.31
    v_r = -0.18 * np.ones_like(r)
    rho, v_r = _conserved_ks_inflow(r, mass, mdot_true, v_r)
    data = _FakeData(
        {"rho": rho, "v1": v_r},
        _FakeMesh(x1v),
        _FakeMeta("kerr_schild", mass),
    )

    mdot, cert = accretion_from_checkpoint(data)  # chart + mass from metadata
    assert np.allclose(mdot, mdot_true, rtol=1e-12)
    assert cert["relative_spread"] < 1e-12


def test_accretion_from_checkpoint_rejects_flat_and_massless():
    x1v = _log_vertices(2.5, 50.0, 8)
    rho = np.ones(8)
    v1 = -0.1 * np.ones(8)
    # a flat (minkowski) chart has no horizon: the certificate is meaningless.
    flat = _FakeData({"rho": rho, "v1": v1}, _FakeMesh(x1v), _FakeMeta("minkowski", 0.0))
    with pytest.raises(ValueError):
        accretion_from_checkpoint(flat)
    # a GR chart with no mass attr must be overridable but fail loud when absent.
    massless = _FakeData(
        {"rho": rho, "v1": v1}, _FakeMesh(x1v), _FakeMeta("kerr_schild", 0.0)
    )
    with pytest.raises(ValueError):
        accretion_from_checkpoint(massless)


def test_superluminal_velocity_is_rejected():
    # gamma_rr (v^r)^2 >= 1 must fail loud, not silently produce a NaN flux.
    mass = 1.0
    r = np.array([3.0])
    v_r = np.array([0.85])  # h (v^r)^2 = (1 + 2/3) 0.7225 > 1 at r = 3
    with pytest.raises(ValueError):
        accretion_rate(np.array([1.0]), [v_r], r, None, mass, spacetime="kerr_schild")


def test_unsupported_chart_is_rejected():
    with pytest.raises(ValueError):
        accretion_rate(np.array([1.0]), [np.array([-0.1])], np.array([3.0]), None, 1.0, spacetime="kerr")
