# =============================================================================
# test_fm_kerr_schild_metric.py
#
# the Fishbone-Moncrief B-field seeding metric on the horizon-penetrating
# Kerr-Schild chart. the vector-potential curl normalization reads the spatial
# determinant sqrt(det gamma) and the poloidal (gamma_rr, gamma_thetatheta); on
# a grid whose inner boundary lies inside the horizon these must stay finite
# and regular through r = 2M at every spin. the Boyer-Lindquist areal form
# sqrt(1 - 2M/r) is singular there; the Kerr-Schild form is not.
#
# pins:
# - at a = 0 the determinant equals the analytic Schwarzschild-Kerr-Schild
#   value sqrt(1 + 2M/r) r^2 sin(theta), and gamma_rr = 1 + 2M/r,
#   gamma_thetatheta = r^2, exactly;
# - both are finite and positive across and inside r = 2M (r in {M, 2M, 3M});
# - the a != 0 branch reduces to the a = 0 branch as a -> 0.
# =============================================================================

import math

import pytest

from simbi_configs.examples.grmhd.gr_fishbone_moncrief_mhd import (
    GrFishboneMoncriefMhd,
)


class _Metric:
    """the two metric methods depend only on the mass and spin fields, so a
    lightweight carrier exercises them without the full model construction."""

    def __init__(self, mass: float, spin: float) -> None:
        self.schwarzschild_mass = mass
        self.kerr_spin = spin

    sqrtg = GrFishboneMoncriefMhd._sqrtg
    gamma_poloidal = GrFishboneMoncriefMhd._gamma_poloidal


def test_zero_spin_determinant_matches_schwarzschild_ks() -> None:
    mass = 1.0
    m = _Metric(mass, 0.0)
    for r in (0.5, 1.0, 2.0, 2.0001, 6.0, 24.0):
        for th in (0.05, math.pi / 3, math.pi / 2, math.pi - 0.05):
            analytic = math.sqrt(1.0 + 2.0 * mass / r) * r * r * math.sin(th)
            assert m.sqrtg(r, th) == pytest.approx(analytic, rel=1e-14)
            grr, gtt = m.gamma_poloidal(r, th)
            assert grr == pytest.approx(1.0 + 2.0 * mass / r, rel=1e-14)
            assert gtt == pytest.approx(r * r, rel=1e-14)


def test_metric_is_regular_through_and_inside_the_horizon() -> None:
    # r = 2M is the Schwarzschild horizon; the Boyer-Lindquist areal factor
    # diverges there and goes imaginary inside. the Kerr-Schild form stays
    # finite and positive.
    mass = 1.0
    m = _Metric(mass, 0.0)
    for r in (mass, 2.0 * mass, 3.0 * mass):
        for th in (0.1, math.pi / 2):
            val = m.sqrtg(r, th)
            grr, gtt = m.gamma_poloidal(r, th)
            assert math.isfinite(val) and val > 0.0
            assert math.isfinite(grr) and grr > 0.0
            assert math.isfinite(gtt) and gtt > 0.0


def test_spinning_branch_reduces_to_zero_spin() -> None:
    mass = 1.0
    r, th = 6.0, math.pi / 3
    zero = _Metric(mass, 0.0)
    for a in (1e-6, 1e-4):
        spun = _Metric(mass, a)
        assert spun.sqrtg(r, th) == pytest.approx(zero.sqrtg(r, th), rel=1e-6)
        grr_a, gtt_a = spun.gamma_poloidal(r, th)
        grr_0, gtt_0 = zero.gamma_poloidal(r, th)
        assert grr_a == pytest.approx(grr_0, rel=1e-6)
        assert gtt_a == pytest.approx(gtt_0, rel=1e-6)
