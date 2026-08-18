# =============================================================================
# test_michel.py
#
# the relativistic bondi analytics the GR reservoir is built from, gated against
# the exact solution they approximate.
# =============================================================================
import math

import pytest
from scipy.optimize import brentq, minimize_scalar

from simbi.functional.michel import critical_point

GAMMA, RHO_INF, P_INF, MASS = 4.0 / 3.0, 1.0, 0.05, 1.0


def _exact(r, crit):
    """the michel state at r, solved directly from the bernoulli invariant."""
    g, k = GAMMA, crit.kappa
    h = lambda rho: 1.0 + g / (g - 1.0) * k * rho ** (g - 1.0)
    h_inf = crit.ambient_enthalpy

    def bernoulli(u):
        return h(crit.rate / (r * r * u)) ** 2 * (1 - 2 * MASS / r + u * u) - h_inf**2

    turn = minimize_scalar(bernoulli, bounds=(1e-6, 5.0), method="bounded").x
    u = brentq(bernoulli, 1e-8, turn)
    return crit.rate / (r * r * u), u


def _far_field(r, crit):
    """the closed-form asymptotic state, mirroring `michel.far_field` in plain python."""
    lever = crit.ambient_enthalpy / (crit.ambient_enthalpy - 1.0) * MASS
    rho = crit.ambient_density * (1.0 + lever / r) ** (1.0 / (GAMMA - 1.0))
    return rho, crit.rate / (r * r * rho)


@pytest.fixture(scope="module")
def crit():
    return critical_point(
        gamma=GAMMA, density=RHO_INF, pressure=P_INF, mass=MASS
    )


def test_the_critical_point_satisfies_its_own_defining_conditions(crit):
    # u_s^2 = M/(2 r_s) and c_s^2 = u_s^2/(1 - 3 u_s^2) are the transonic conditions;
    # solving one and checking the other closes the loop independently of the solver.
    u2 = crit.four_velocity**2
    assert u2 == pytest.approx(MASS / (2.0 * crit.radius), rel=1e-12)
    assert crit.sound_speed**2 == pytest.approx(u2 / (1.0 - 3.0 * u2), rel=1e-12)


def test_the_critical_radius_is_not_the_newtonian_estimate(crit):
    # the estimate this replaced, `(5 - 3 gamma)/4 * r_bondi` on a newtonian sound
    # speed, is wrong by a factor of two here. the gate is that they stay apart: a
    # regression to the newtonian form would send a run looking for its sonic surface
    # at half the radius.
    newtonian = 0.25 * (5.0 - 3.0 * GAMMA) * MASS / (GAMMA * P_INF / RHO_INF)
    assert crit.radius == pytest.approx(7.3455, abs=1e-3)
    assert crit.radius > 1.8 * newtonian


def test_the_far_field_tracks_the_exact_solution_across_the_buffer(crit):
    # the reservoir sits beyond about three sonic radii, where u^2 is negligible
    # against 2M/r and the closed form is accurate to under a percent. inside that
    # the expansion has no claim and the gate does not make one.
    for r in (3.0 * crit.radius, 5.0 * crit.radius, 8.0 * crit.radius):
        rho_ff, u_ff = _far_field(r, crit)
        rho_ex, u_ex = _exact(r, crit)
        assert rho_ff == pytest.approx(rho_ex, rel=0.01), f"density at r = {r:.1f}"
        assert u_ff == pytest.approx(u_ex, rel=0.02), f"velocity at r = {r:.1f}"


def test_the_ambient_clamp_the_reservoir_replaces_is_far_off(crit):
    # the premise of the whole change. if the box face were effectively ambient there
    # would be nothing to fix and the reservoir would be needless machinery.
    rho_face, _ = _exact(3.0 * 15.0, crit)  # L = 3 r_bondi, the config's default cube
    assert rho_face > 1.4 * RHO_INF, (
        f"the face carries rho/rho_inf = {rho_face:.3f}; an ambient dirichlet would be "
        "close enough and this reservoir would not be worth its complexity"
    )


def test_the_valencia_velocity_of_an_inflow_points_outward(crit):
    # the sign a flat intuition gets wrong: the kerr-schild normal observer falls
    # inward faster than a pressure-supported flow, so the stored valencia component
    # is positive even though the gas is accreting.
    r = 45.0
    _, u = _exact(r, crit)
    h = 2.0 * MASS / r
    a = h / math.sqrt(1.0 + h)
    denom = 1.0 + u * u * (1.0 + h)
    v_r = (a - math.sqrt(a * a - denom * (a * a - u * u))) / denom
    assert v_r > 0.0, f"valencia v^r = {v_r:.5f}; an accreting flow reads outward here"
    # and it inverts: W(v - beta/alpha) must return the coordinate four-velocity.
    w = 1.0 / math.sqrt(1.0 - (1.0 + h) * v_r * v_r)
    assert w * (v_r - a) == pytest.approx(-u, rel=1e-9)
