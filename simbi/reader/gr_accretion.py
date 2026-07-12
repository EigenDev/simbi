# =============================================================================
# gr_accretion.py
#
# the GR accretion well-posedness certificate: the rest-mass accretion rate as a
# SURFACE FLUX on coordinate spheres, extracted at several radii. in steady state
# a well-posed black-hole flow has Mdot(r_ex) INDEPENDENT of r_ex — the causal
# analog of the Newtonian "Mdot independent of r_mask" gate. the inter-radius
# spread IS the error bar.
#
#   Mdot(r_ex) = - oint rho u^r sqrt(-g) dtheta dphi
#
# Schwarzschild-Kerr-Schild (horizon-penetrating), the only chart this is valid in:
#   h(r)     = 1 + 2M/r
#   alpha    = 1/sqrt(h)                         (lapse; never zero, finite at r_+)
#   beta^r   = 2M/(r + 2M)                       (radial shift; ingoing)
#   sqrt(-g) = alpha sqrt(gamma) = r^2 sin(theta)   (the lapse cancels gamma_rr)
#
# velocity convention: the substrate stores the PHYSICAL (orthonormal) velocity
# V^rhat = sqrt(gamma_rr) v^r = sqrt(h) v^r, and the Lorentz factor is the flat
# W = 1/sqrt(1 - sum_i (V^ihat)^2). the coordinate radial 4-velocity is then
#   u^r = W (v^r - beta^r/alpha) = W (V^rhat/sqrt(h) - beta^r sqrt(h)).
# getting this conversion wrong produces a flux that looks right at large r and is
# garbage near r_+ — validate against the analytic Michel/Bondi Mdot.
# =============================================================================

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

Array = np.ndarray


def _schwarzschild_ks(r: Array, mass: float) -> tuple[Array, Array, Array]:
    """(sqrt_h, alpha, beta_r) for Schwarzschild-KS at coordinate radius r."""
    h = 1.0 + 2.0 * mass / r
    sqrt_h = np.sqrt(h)
    alpha = 1.0 / sqrt_h
    beta_r = 2.0 * mass / (r + 2.0 * mass)
    return sqrt_h, alpha, beta_r


def coordinate_u_r(v_phys: Sequence[Array], r: Array, mass: float) -> tuple[Array, Array]:
    """the coordinate radial 4-velocity u^r and Lorentz factor W from the physical
    velocity components (V^rhat, V^thetahat, ...) at radius r. `v_phys[0]` is the
    radial physical component; broadcasting against `r` is the caller's job."""
    sqrt_h, alpha, beta_r = _schwarzschild_ks(r, mass)
    v_sq = np.zeros_like(np.asarray(v_phys[0], dtype=float))
    for comp in v_phys:
        v_sq = v_sq + np.asarray(comp, dtype=float) ** 2
    if np.any(v_sq >= 1.0):
        raise ValueError("physical velocity exceeds the speed of light")
    w = 1.0 / np.sqrt(1.0 - v_sq)
    v_r_coord = np.asarray(v_phys[0], dtype=float) / sqrt_h
    u_r = w * (v_r_coord - beta_r / alpha)
    return u_r, w


def accretion_rate(
    rho: Array,
    v_phys: Sequence[Array],
    r: Array,
    theta: Array | None,
    mass: float,
    *,
    dtheta: Array | float | None = None,
    dphi: float = 2.0 * np.pi,
) -> Array:
    """the rest-mass accretion rate Mdot(r_ex) at every radial shell.

    `rho` has shape `(nr, ntheta)` (axisymmetric) or `(nr,)` (spherical 1D). `v_phys`
    is the list of PHYSICAL velocity component arrays with the same shape. `r` is the
    radial cell centres `(nr,)`; `theta` the polar cell centres `(ntheta,)` or None
    for 1D. `dtheta`/`dphi` are the coordinate cell widths (1D uses the full sphere).
    Returns `Mdot` of shape `(nr,)` — one accretion rate per shell.
    """
    rho = np.asarray(rho, dtype=float)
    r = np.asarray(r, dtype=float)
    if rho.ndim == 1:
        # spherical 1D: the full-sphere surface integral is 4 pi r^2.
        u_r, _ = coordinate_u_r([np.asarray(v_phys[0], dtype=float)], r, mass)
        return -4.0 * np.pi * r * r * rho * u_r
    if rho.ndim != 2:
        raise ValueError(f"accretion_rate: expected rho of ndim 1 or 2, got {rho.ndim}")
    if theta is None:
        raise ValueError("accretion_rate: 2D rho requires theta cell centres")
    theta = np.asarray(theta, dtype=float)
    if dtheta is None:
        dtheta = np.gradient(theta)
    dtheta_arr = np.broadcast_to(np.asarray(dtheta, dtype=float), theta.shape)

    r_col = r[:, None]  # (nr, 1)
    u_r, _ = coordinate_u_r([np.asarray(c, dtype=float) for c in v_phys], r_col, mass)
    # integrand rho u^r sqrt(-g), sqrt(-g) = r^2 sin(theta).
    integrand = rho * u_r * (r_col * r_col) * np.sin(theta)[None, :]
    # oint dtheta dphi over the polar band (phi integrated as dphi).
    shell = np.sum(integrand * dtheta_arr[None, :], axis=1) * dphi
    return -shell


def rex_invariance(mdot: Array, r: Array, radii: Sequence[float]) -> dict[str, Any]:
    """the certificate: sample Mdot at `radii`, report the relative spread. a
    well-posed steady flow has a spread near truncation error. `mdot`/`r` are the
    per-shell arrays from `accretion_rate`."""
    r = np.asarray(r, dtype=float)
    samples = {}
    for r_ex in radii:
        idx = int(np.argmin(np.abs(r - r_ex)))
        samples[float(r[idx])] = float(mdot[idx])
    vals = np.array(list(samples.values()))
    mean = float(np.mean(vals))
    spread = float((np.max(vals) - np.min(vals)) / abs(mean)) if mean != 0.0 else float("inf")
    return {"samples": samples, "mean": mean, "relative_spread": spread}
