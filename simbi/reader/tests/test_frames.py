# =============================================================================
# test_frames.py
#
# the observer transformations, gated against the michel (relativistic bondi)
# solution, whose sonic surface is known in closed form.
#
# the property under test is the one that makes the layer necessary: the frame a
# field is stored in is not the frame a question is asked in. simbi stores the
# valencia velocity, against the normal observer, and on a horizon-penetrating
# chart that observer falls inward — so a transonic inflow reads subsonic there
# at every radius, including the one where it crosses its own sound speed.
# =============================================================================
import math

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

from simbi.reader import frames

GAMMA, MASS, KAPPA, H_INF = 4.0 / 3.0, 1.0, 0.05, 1.2
# the michel critical point for (gamma, rho_inf, p_inf, M) = (4/3, 1, 0.05, 1):
# u_s^2 = M/2r_s and c_s^2 = u_s^2/(1 - 3u_s^2) solved against the bernoulli
# invariant h^2(1 - 2M/r + u^2) = h_inf^2.
R_SONIC, U_SONIC, RHO_SONIC = 7.345530, 0.2608987, 5.14137


def _h(rho):
    return 1.0 + GAMMA / (GAMMA - 1.0) * KAPPA * rho ** (GAMMA - 1.0)


def _cs(rho):
    t = KAPPA * rho ** (GAMMA - 1.0)
    return math.sqrt(GAMMA * (GAMMA - 1.0) * t / ((GAMMA - 1.0) + GAMMA * t))


def _michel(r):
    """(rho, |u^r|) of the michel solution at areal radius r, on the branch that is
    supersonic inside the critical point and subsonic outside it."""
    # the two branches merge at the critical point, so no bracket spans a sign change
    # there; the critical state is the solution and is quoted rather than solved for.
    if abs(r - R_SONIC) < 1e-9:
        return RHO_SONIC, U_SONIC
    mdot = R_SONIC**2 * RHO_SONIC * U_SONIC

    def bernoulli(u):
        return _h(mdot / (r * r * u)) ** 2 * (1 - 2 * MASS / r + u * u) - H_INF**2

    turn = minimize_scalar(bernoulli, bounds=(1e-6, 5.0), method="bounded").x
    u = brentq(bernoulli, turn, 5.0) if r < R_SONIC else brentq(bernoulli, 1e-8, turn)
    return mdot / (r * r * u), u


def _valencia_from_u(r, u):
    """the valencia velocity a kerr-schild run stores for a radial inflow of
    four-velocity `u^r = -u`, by inverting `u^i = W(v^i - beta^i/alpha)`."""
    adm = _adm(r)
    alpha, beta, h = float(adm.alpha[0]), float(adm.beta[0][0]), float(adm.h[0])
    lim = 0.999 / math.sqrt(1.0 + h)

    def residual(v):
        w = 1.0 / math.sqrt(max(1.0 - (1.0 + h) * v * v, 1e-300))
        return w * (v - beta / alpha) + u

    return brentq(residual, -lim, lim)


def _adm(r):
    return frames.adm_decomposition(
        "schwarzschild_ks", MASS, [np.array([r]), np.array([0.0]), np.array([0.0])]
    )


def _static_mach(r):
    rho, u = _michel(r)
    v = _valencia_from_u(r, u)
    adm = _adm(r)
    ut, ui = frames.four_velocity_from_valencia(
        [np.array([v]), np.array([0.0]), np.array([0.0])], adm
    )
    speed = frames.speed_from_lorentz(frames.static_lorentz(ut, ui, adm))
    return float(speed[0]) / _cs(rho), v, rho


def test_the_static_frame_reads_unity_at_the_michel_sonic_radius():
    # the certificate. the critical point is where a static observer measures the flow
    # at exactly its own sound speed, so the transformation is pinned by a number the
    # michel solution fixes independently of any of this code.
    mach, _, _ = _static_mach(R_SONIC)
    assert mach == pytest.approx(1.0, abs=2.0e-4), (
        f"static-frame mach at the michel critical point is {mach:.6f}, not unity; "
        "the observer change no longer agrees with the closed-form solution"
    )


def test_the_static_frame_brackets_the_sonic_radius():
    # monotone through the crossing, supersonic inside and subsonic outside, so the
    # field locates the surface rather than merely touching unity somewhere.
    inner, _, _ = _static_mach(0.6 * R_SONIC)
    outer, _, _ = _static_mach(1.6 * R_SONIC)
    assert inner > 1.0, f"the flow reads {inner:.4f} inside the sonic radius"
    assert outer < 1.0, f"the flow reads {outer:.4f} outside the sonic radius"


def test_the_stored_valencia_frame_never_reaches_its_sound_speed():
    # the premise of the whole layer, and the reason a `mach` plot of a correct run
    # looks uniformly subsonic: relative to the infalling normal observer, a
    # pressure-supported inflow is slow everywhere, because it always falls slower
    # than the free-falling observer measuring it.
    worst = 0.0
    for r in (2.2, 3.0, 5.0, R_SONIC, 12.0, 30.0):
        _, v, rho = _static_mach(r)
        adm = _adm(r)
        h = float(adm.h[0])
        # the proper speed against the normal observer, gamma_ij v^i v^j on this chart.
        worst = max(worst, abs(v) * math.sqrt(1.0 + h) / _cs(rho))
    assert worst < 0.5, (
        f"the valencia-frame mach number reached {worst:.4f}; this gate's whole point "
        "is that it stays well below unity through a genuinely transonic flow"
    )


def test_a_flat_background_leaves_the_frames_coincident():
    # on minkowski the normal observer IS the static one, so the transformation must be
    # the identity and `mach_static` must agree with `mach` bit for bit.
    adm = frames.adm_decomposition(
        "minkowski", 0.0, [np.array([5.0]), np.array([0.0]), np.array([0.0])]
    )
    v = [np.array([0.3]), np.array([0.0]), np.array([0.0])]
    ut, ui = frames.four_velocity_from_valencia(v, adm)
    speed = frames.speed_from_lorentz(frames.static_lorentz(ut, ui, adm))
    assert float(speed[0]) == pytest.approx(0.3, abs=1e-12)


def test_no_static_observer_inside_the_horizon():
    # the killing vector is null at r = 2M and spacelike inside, so no worldline holds
    # a fixed radius. the field reports that absence rather than a number belonging to
    # no observer.
    for r in (2.0, 1.5, 0.9):
        adm = _adm(r)
        ut, ui = frames.four_velocity_from_valencia(
            [np.array([0.1]), np.array([0.0]), np.array([0.0])], adm
        )
        assert np.isnan(frames.static_lorentz(ut, ui, adm)[0]), (
            f"a static lorentz factor was returned at r = {r}, at or inside the horizon"
        )


def test_an_unsupported_background_is_refused_by_name():
    # a kerr shift carries an azimuthal component this module does not build; treating
    # it as flat would report the valencia frame under a static observer's label.
    with pytest.raises(ValueError, match="kerr"):
        frames.adm_decomposition("kerr_ks", 1.0, [np.array([5.0])] * 3)
