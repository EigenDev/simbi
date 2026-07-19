# =============================================================================
# test_kerr_schild_sources.py
#
# the KS geodesic-source ORACLE: the closed-form kerr-schild momentum + energy
# sources (mirrored in symbi-discretize gv/godunov.rs) validated against a DIRECT
# numerical tensor computation of the valencia source vector on the ingoing
# kerr-schild metric — `(1/2) T^{mn} d_r g` (momentum) and
# `alpha(T^{m0} d_m ln alpha - T^{mn} Gamma^0)` (energy), with christoffels from a
# finite-differenced metric. checked at radii spanning OUTSIDE -> INSIDE the
# horizon (2M = 2), so the sources are verified where the schwarzschild chart
# cannot reach. pure numpy; no built backend required.
# =============================================================================
import numpy as np
import pytest

M = 1.0
GAMMA = 4.0 / 3.0


def _metric(r, th):
    b = 2 * M / r
    g = np.zeros((4, 4))
    g[0, 0] = -(1 - b)
    g[0, 1] = g[1, 0] = b
    g[1, 1] = 1 + b
    g[2, 2] = r * r
    g[3, 3] = r * r * np.sin(th) ** 2
    return g


def _four_velocity(r, big_v):
    b = 2 * M / r
    sqh = np.sqrt(1 + b)
    w = 1.0 / np.sqrt(1 - big_v * big_v)
    return np.array([w * sqh, w * (big_v - b) / sqh, 0.0, 0.0])


def _source_numeric(r, big_v, rho, p, th=np.pi / 2, dr=1e-6):
    g = _metric(r, th)
    ginv = np.linalg.inv(g)
    dg = (_metric(r + dr, th) - _metric(r - dr, th)) / (2 * dr)  # d_r g
    lapse = lambda rr: 1.0 / np.sqrt(-np.linalg.inv(_metric(rr, th))[0, 0])
    dln_alpha_dr = (np.log(lapse(r + dr)) - np.log(lapse(r - dr))) / (2 * dr)
    alpha = lapse(r)
    # d_lambda g_{mu nu}: only the radial derivative is nonzero (static, and the
    # d_theta g_phiphi term the t-christoffels would see multiplies g^{t theta}=0).
    dgl = np.zeros((4, 4, 4))
    dgl[1] = dg
    gt = np.zeros((4, 4))  # Gamma^t_{mu nu}
    for mu in range(4):
        for nu in range(4):
            gt[mu, nu] = 0.5 * sum(
                ginv[0, lam] * (dgl[mu, lam, nu] + dgl[nu, lam, mu] - dgl[lam, mu, nu])
                for lam in range(4)
            )
    u = _four_velocity(r, big_v)
    assert abs(g @ u @ u + 1) < 1e-8  # normalization
    eta = 1 + GAMMA / (GAMMA - 1) * p / rho
    tmunu = rho * eta * np.outer(u, u) + p * ginv
    s_sr = 0.5 * np.einsum("mn,mn->", tmunu, dg)
    dln = np.array([0.0, dln_alpha_dr, 0.0, 0.0])
    s_tau = alpha * (
        np.einsum("m,m->", tmunu[:, 0], dln) - np.einsum("mn,mn->", tmunu, gt)
    )
    return s_sr, s_tau


def _source_closed(r, big_v, rho, p):
    # the closed forms implemented in gv/godunov.rs (KerrSchild arms of radial_gravity + nrg_gravity).
    b = 2 * M / r
    h = 1 + b
    w = 1 / np.sqrt(1 - big_v * big_v)
    eta = 1 + GAMMA / (GAMMA - 1) * p / rho
    e = rho * eta * w * w  # E = D + tau + p
    s_sr = -M * e * (1 + big_v) ** 2 / (r * r * h) + 2 * p / r
    s_tau = -M / (r * r * h**1.5) * (e * big_v * (1 + (2 + b) * big_v) - p * (2 + 3 * b))
    return s_sr, s_tau


@pytest.mark.parametrize("r", [8.0, 3.0, 2.0, 1.5, 1.2])  # 2M = 2: spans outside -> inside horizon
@pytest.mark.parametrize("big_v", [-0.6, -0.2, 0.1])
def test_ks_sources_match_numerical_tensor(r, big_v):
    rho, p = 1.3, 0.05
    sn, tn = _source_numeric(r, big_v, rho, p)
    sc, tc = _source_closed(r, big_v, rho, p)
    assert abs(sn - sc) < 1e-4 * (1 + abs(sc)), f"S_Sr {sn} != {sc}"
    assert abs(tn - tc) < 1e-4 * (1 + abs(tc)), f"S_tau {tn} != {tc}"
