# =============================================================================
# test_fishbone_moncrief_oracle.py
#
# certification of the general-spin fishbone-moncrief oracle against the paper's
# printed disks (fm 1976, ApJ 207:962, figs. 2-3 captions) — five independent
# published test vectors, each pinning l (eq. 3.8 with kappa), the potential
# maximum (ln h)_max and its radius, the equatorial outer edge, and the minimum
# polar angle of the ln h = 0 surface:
#
#   fig. 2 schwarzschild      r_in = 6,    kappa = 1.01      (the a = 0 regression)
#   fig. 2 extreme-kerr co    r_in = 6,    kappa = 1.411698
#   fig. 3 extreme-kerr co    r_in = 2.78, kappa = 1.002
#   fig. 3 extreme-kerr ctr   r_in = 7.75, kappa = 1.003986  (retrograde branch)
#
# plus the KS-chart physicality of the code primitives (subluminal everywhere on
# the disk, the orbiter's v^r = b/sqrt(1+b) drift). pure python — no backend.
# =============================================================================
import math

import numpy as np
import pytest

from simbi_configs.helpers.fishbone_moncrief import FishboneMoncrief

# extreme kerr: the fishbone-moncrief solution sits at a = M but is defined only for
# |a| < M, so the
# vectors evaluate in the limit (the printed 3-4 digit values are insensitive).
_A_EXTREME = 1.0 - 1e-9


def _survey(t: FishboneMoncrief, r_hi: float):
    rr = np.linspace(t.r_in * 1.0001, r_hi, 60000)
    lnh = np.array([t._lnh(r, math.pi / 2) for r in rr])
    imax = int(lnh.argmax())
    edge = rr[np.nonzero(lnh > 0)[0][-1]]
    th = np.linspace(0.2, math.pi / 2, 3000)
    min_th = math.pi / 2
    for r in np.linspace(t.r_in * 1.01, edge, 400):
        inside = [x for x in th if t._lnh(r, x) > 0]
        if inside:
            min_th = min(min_th, inside[0])
    return lnh[imax], rr[imax], edge, math.degrees(min_th)


_PAPER_DISKS = [
    # (name, kwargs, l, lnh_max, r_max, edge, polar_deg, r_survey)
    ("fig2_schw", dict(r_in=6.0, kappa=1.01), 4.92, 0.0153, 16.0, 73.812, 45.2, 110.0),
    (
        "fig2_kerr_co",
        dict(r_in=6.0, kappa=1.411698, spin=_A_EXTREME, chart="ks"),
        4.36,
        0.0223,
        12.9,
        73.812,
        40.0,
        110.0,
    ),
    (
        "fig3_kerr_co",
        dict(r_in=2.78, kappa=1.002, spin=_A_EXTREME, chart="ks"),
        4.02,
        0.0393,
        9.7,
        262.3,
        19.9,
        380.0,
    ),
    (
        "fig3_kerr_ctr",
        dict(r_in=7.75, kappa=1.003986, spin=_A_EXTREME, prograde=False, chart="ks"),
        -5.67,
        0.0158,
        22.3,
        262.3,
        28.2,
        380.0,
    ),
]


@pytest.mark.parametrize("name,kw,l,lnh_max,r_max,edge,polar,r_hi", _PAPER_DISKS)
def test_oracle_reproduces_the_published_disks(
    name, kw, l, lnh_max, r_max, edge, polar, r_hi
):
    t = FishboneMoncrief(mass=1.0, gamma=4.0 / 3.0, rho_max=1.0, **kw)
    assert abs(t.ell - l) < 6e-3, f"{name}: l = {t.ell:.4f} vs paper {l}"
    m, rm, e, p = _survey(t, r_hi)
    assert abs(m - lnh_max) < 1e-4, f"{name}: (ln h)_max = {m:.5f} vs paper {lnh_max}"
    # the potential is flat near its maximum, so the printed location carries the
    # paper's rounding; half a gravitational radius covers it.
    assert abs(rm - r_max) < 0.5, f"{name}: r_max = {rm:.2f} vs paper {r_max}"
    assert abs(e - edge) < 0.5, f"{name}: outer edge = {e:.2f} vs paper {edge}"
    assert abs(p - polar) < 0.2, f"{name}: min polar = {p:.2f} deg vs paper {polar}"


def test_ks_chart_primitives_are_physical() -> None:
    # the extreme-kerr corotating disk in the horizon-penetrating chart: the code
    # primitives must be subluminal everywhere on the disk (the KS spatial norm
    # gamma_ij v^i v^j < 1, including the frame-dragging off-diagonal), with the
    # orbiter's radial drift v^r = b/sqrt(1 + b) against the infalling observers.
    a = _A_EXTREME
    t = FishboneMoncrief(
        mass=1.0,
        r_in=6.0,
        gamma=4.0 / 3.0,
        rho_max=1.0,
        kappa=1.411698,
        spin=a,
        chart="ks",
    )
    for r in np.linspace(6.05, 70.0, 60):
        for th in np.linspace(0.75, math.pi / 2, 12):
            state = t.primitive(r, th)
            if state is None:
                continue
            rho, v_r, v_p, pre = state
            st, ct = math.sin(th), math.cos(th)
            sig = r * r + (a * ct) ** 2
            b = 2.0 * r / sig
            assert abs(v_r - b / math.sqrt(1.0 + b)) < 1e-14
            g_rr = 1.0 + b
            g_rp = -a * st * st * (1.0 + b)
            g_pp = st * st * (sig + a * a * st * st * (1.0 + b))
            v_sq = g_rr * v_r**2 + 2.0 * g_rp * v_r * v_p + g_pp * v_p**2
            assert 0.0 < v_sq < 1.0, (
                f"superluminal disk state at r={r:.1f} th={th:.2f}: {v_sq}"
            )
            assert rho > 0.0 and pre > 0.0


def test_general_oracle_reduces_to_zero_spin() -> None:
    # the general (Sigma, Delta, A) formulas at a = 0 must reproduce the
    # schwarzschild closed forms: the stationary euler balance
    # dp/dr = E[-M/(r^2 f) ... ] holds to the finite-difference floor.
    t = FishboneMoncrief(mass=1.0, r_in=6.0, gamma=4.0 / 3.0, rho_max=1.0, kappa=1.01)
    th = 1.45
    for r in (10.0, 16.0, 30.0):
        dr = 1e-5 * r
        pm = t.primitive(r - dr, th)[3]
        pp = t.primitive(r + dr, th)[3]
        dpdr = (pp - pm) / (2.0 * dr)
        rho, _, vphi, p = t.primitive(r, th)
        f = 1.0 - 2.0 / r
        st = math.sin(th)
        gpp = (r * st) ** 2
        w2 = 1.0 / (1.0 - gpp * vphi**2)
        e = rho * (1.0 + 4.0 * p / rho) * w2
        rhs = e * (-1.0 / (r * r * f) + vphi**2 * r * st * st)
        assert abs(dpdr / rhs - 1.0) < 1e-7, f"euler residual at r={r}"
