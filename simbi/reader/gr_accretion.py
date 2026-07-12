# =============================================================================
# gr_accretion.py
#
# the GR accretion well-posedness certificate: the rest-mass accretion rate as a
# SURFACE FLUX on coordinate spheres, extracted at several radii. in steady state
# a well-posed black-hole flow has Mdot(r_ex) INDEPENDENT of r_ex — the causal
# analog of the Newtonian "Mdot independent of r_mask" gate. the inter-radius
# spread IS the error bar.
#
#   Mdot(r_ex) = - oint rho u^r sqrt(-g) dtheta dphi,   sqrt(-g) = r^2 sin(theta)
#
# velocity convention: the substrate stores the valencia CONTRAVARIANT 3-velocity
# v^i (the eulerian-observer velocity, = u^i/W + beta^i/alpha), not the physical
# orthonormal V^ihat. the lorentz factor and coordinate 4-velocity are then
#   W   = 1/sqrt(1 - gamma_ij v^i v^j),   u^r = W (v^r - beta^r/alpha).
# getting this conversion wrong (e.g. treating v^r as orthonormal) produces a flux
# that looks right at large r, where the metric flattens, and is garbage near the
# horizon — validate against the exact michel/bondi Mdot.
#
# supported diagonal spherically-symmetric charts (both have sqrt(-g) = r^2 sin(theta)):
#   schwarzschild  f = 1 - 2M/r,   gamma_rr = 1/f,  beta^r = 0,        alpha = sqrt(f)
#   kerr_schild    h = 1 + 2M/r,   gamma_rr = h,    beta^r = 2M/(r+2M), alpha = 1/sqrt(h)
# kerr (non-diagonal spatial metric, frame dragging) is unsupported here.
# =============================================================================

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

Array = np.ndarray

_SUPPORTED_CHARTS = ("schwarzschild", "kerr_schild")


def _radial_metric(r: Array, mass: float, spacetime: str) -> tuple[Array, Array]:
    """(gamma_rr, beta^r/alpha) for a diagonal spherically-symmetric background at
    coordinate radius r. beta^r/alpha is the only shift combination the u^r
    conversion needs. raises on an unsupported chart."""
    if spacetime == "schwarzschild":
        f = 1.0 - 2.0 * mass / r
        return 1.0 / f, np.zeros_like(np.asarray(r, dtype=float))
    if spacetime == "kerr_schild":
        h = 1.0 + 2.0 * mass / r
        beta_r = 2.0 * mass / (r + 2.0 * mass)
        return h, beta_r * np.sqrt(h)  # beta^r/alpha = beta^r sqrt(h)
    raise ValueError(
        f"gr_accretion: unsupported spacetime '{spacetime}'; "
        f"expected one of {_SUPPORTED_CHARTS}"
    )


def coordinate_u_r(
    v_contra: Sequence[Array],
    r: Array,
    mass: float,
    spacetime: str,
    theta: Array | None = None,
) -> tuple[Array, Array]:
    """the coordinate radial 4-velocity u^r and lorentz factor W from the stored
    CONTRAVARIANT valencia velocity components (v^r, [v^theta], [v^phi]) at radius r.
    the polar component is contracted with gamma_thetatheta = r^2, the azimuthal with
    gamma_phiphi = r^2 sin^2(theta) (theta required when v^phi is present)."""
    gamma_rr, beta_over_alpha = _radial_metric(r, mass, spacetime)
    v_r = np.asarray(v_contra[0], dtype=float)
    vv = gamma_rr * v_r * v_r
    if len(v_contra) >= 2:
        vv = vv + (r * r) * np.asarray(v_contra[1], dtype=float) ** 2
    if len(v_contra) >= 3:
        if theta is None:
            raise ValueError("coordinate_u_r: v^phi present but theta not supplied")
        sin_sq = np.sin(theta) ** 2
        vv = vv + (r * r * sin_sq) * np.asarray(v_contra[2], dtype=float) ** 2
    if np.any(vv >= 1.0):
        raise ValueError("valencia 3-velocity is superluminal: gamma_ij v^i v^j >= 1")
    w = 1.0 / np.sqrt(1.0 - vv)
    u_r = w * (v_r - beta_over_alpha)
    return u_r, w


def accretion_rate(
    rho: Array,
    v_contra: Sequence[Array],
    r: Array,
    theta: Array | None,
    mass: float,
    spacetime: str = "kerr_schild",
    *,
    dtheta: Array | float | None = None,
    dphi: float = 2.0 * np.pi,
) -> Array:
    """the rest-mass accretion rate Mdot(r_ex) at every radial shell.

    `rho` has shape `(nr, ntheta)` (axisymmetric) or `(nr,)` (spherical 1D). `v_contra`
    is the list of CONTRAVARIANT valencia velocity component arrays with the same shape.
    `r` is the radial cell centres `(nr,)`; `theta` the polar cell centres `(ntheta,)`
    or None for 1D. `dtheta`/`dphi` are the coordinate cell widths (1D uses the full
    sphere). `spacetime` selects the chart. Returns `Mdot` of shape `(nr,)`.
    """
    rho = np.asarray(rho, dtype=float)
    r = np.asarray(r, dtype=float)
    if rho.ndim == 1:
        # spherical 1D: the full-sphere surface integral is 4 pi r^2.
        u_r, _ = coordinate_u_r([np.asarray(v_contra[0], dtype=float)], r, mass, spacetime)
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
    comps = [np.asarray(c, dtype=float) for c in v_contra]
    u_r, _ = coordinate_u_r(comps, r_col, mass, spacetime, theta=theta[None, :])
    # integrand rho u^r sqrt(-g), sqrt(-g) = r^2 sin(theta).
    integrand = rho * u_r * (r_col * r_col) * np.sin(theta)[None, :]
    # oint dtheta dphi over the polar band (phi integrated as dphi).
    shell = np.sum(integrand * dtheta_arr[None, :], axis=1) * dphi
    return -shell


def _radial_centroids(x1v: Array) -> Array:
    """volume-weighted radial cell centroids from vertices — the radius at which the
    backend samples the primitives: r_vw = (3/4)(rh^4 - rl^4)/(rh^3 - rl^3)."""
    rl = x1v[:-1]
    rh = x1v[1:]
    return 0.75 * (rh**4 - rl**4) / (rh**3 - rl**3)


def _default_radii(r: Array, n: int = 4) -> list[float]:
    """n sample radii spanning the interior, biased off the boundary cells."""
    idx = np.linspace(0.1, 0.9, n) * (len(r) - 1)
    return [float(r[int(round(ii))]) for ii in idx]


def shell_accretion(
    rho: Array,
    v_contra: Sequence[Array],
    x1v: Array,
    x2v: Array | None,
    mass: float,
    spacetime: str,
    *,
    radii: Sequence[float] | None = None,
) -> tuple[Array, dict[str, Any]]:
    """the accretion certificate from mesh-native arrays: `rho`/`v_contra` in checkpoint
    STORAGE order (radial = last axis), `x1v`/`x2v` the radial/polar vertex coordinates.
    computes the cell centroids, reduces the shell flux, and samples r_ex-invariance.
    Returns `(mdot_per_shell, certificate)`."""
    rho = np.asarray(rho, dtype=float)
    r = _radial_centroids(np.asarray(x1v, dtype=float))
    if rho.ndim == 1:
        mdot = accretion_rate(rho, [v_contra[0]], r, None, mass, spacetime=spacetime)
    elif rho.ndim == 2:
        if x2v is None:
            raise ValueError("shell_accretion: 2D field requires polar vertices x2v")
        x2v = np.asarray(x2v, dtype=float)
        theta = 0.5 * (x2v[:-1] + x2v[1:])
        dtheta = np.diff(x2v)
        # storage (ntheta, nr) -> reducer (nr, ntheta).
        comps = [np.asarray(c, dtype=float).T for c in v_contra]
        mdot = accretion_rate(
            rho.T, comps, r, theta, mass, spacetime=spacetime, dtheta=dtheta
        )
    else:
        raise NotImplementedError(
            f"shell_accretion: {rho.ndim}D reduction not supported (1D/2D only)"
        )
    cert = rex_invariance(mdot, r, list(radii) if radii is not None else _default_radii(r))
    return mdot, cert


def accretion_from_checkpoint(
    data: Any,
    *,
    mass: float | None = None,
    spacetime: str | None = None,
    radii: Sequence[float] | None = None,
) -> tuple[Array, dict[str, Any]]:
    """the accretion certificate from an opened checkpoint (`SimData`). the chart and
    black-hole mass default to the self-describing metadata; pass them to override an
    older checkpoint that predates those attrs. reduces v1 (radial) plus v2 (polar,
    2D) — the contravariant valencia components the substrate stores."""
    meta = data.metadata
    chart = spacetime if spacetime is not None else meta.spacetime
    bh_mass = mass if mass is not None else meta.schwarzschild_mass
    if chart not in _SUPPORTED_CHARTS:
        raise ValueError(
            f"accretion_from_checkpoint: chart '{chart}' unsupported; the certificate "
            f"needs a diagonal GR chart {_SUPPORTED_CHARTS}"
        )
    if bh_mass <= 0.0:
        raise ValueError(
            "accretion_from_checkpoint: black-hole mass must be positive; got "
            f"{bh_mass} (pass mass=... for a checkpoint without the attr)"
        )

    mesh = data.mesh
    x1v = np.asarray(mesh.x1v, dtype=float)
    rho = data.get_field("rho")
    v_contra = [data.get_field("v1")]
    x2v = None
    if rho.ndim >= 2:
        x2v = np.asarray(mesh.x2v, dtype=float)
        if "v2" in data.available_fields():
            v_contra.append(data.get_field("v2"))
    return shell_accretion(rho, v_contra, x1v, x2v, bh_mass, chart, radii=radii)


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
