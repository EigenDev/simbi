# =============================================================================
# gr_accretion.py
#
# the GR accretion well-posedness certificate: the rest-mass accretion rate as a
# surface flux on coordinate spheres, extracted at several radii. in steady state
# a well-posed black-hole flow has Mdot(r_ex) independent of r_ex — the causal
# analog of the Newtonian "Mdot independent of r_mask" gate. the inter-radius
# spread is the error bar.
#
#   Mdot(r_ex) = - oint rho u^r sqrt(-g) dtheta dphi,   sqrt(-g) = r^2 sin(theta)
#
# velocity convention: the substrate stores the valencia contravariant 3-velocity
# v^i (the eulerian-observer velocity, = u^i/W + beta^i/alpha); the physical
# orthonormal V^ihat differs by the spatial-metric scale factors. the lorentz
# factor and coordinate 4-velocity are then
#   W   = 1/sqrt(1 - gamma_ij v^i v^j),   u^r = W (v^r - beta^r/alpha).
# getting this conversion wrong (e.g. treating v^r as orthonormal) produces a flux
# that looks right at large r, where the metric flattens, and is garbage near the
# horizon — validate against the exact michel/bondi Mdot.
#
# supported diagonal spherically-symmetric charts (both have sqrt(-g) = r^2 sin(theta)):
#   schwarzschild  f = 1 - 2M/r,   gamma_rr = 1/f,  beta^r = 0,        alpha = sqrt(f)
#   schwarzschild_ks  h = 1 + 2M/r, gamma_rr = h, beta^r = 2M/(r+2M), alpha = 1/sqrt(h)
# kerr sits outside this pair: its spatial metric carries off-diagonal terms and
# frame dragging, which this reducer leaves to a dedicated chart.
# =============================================================================

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

Array = np.ndarray

_SUPPORTED_CHARTS = ("schwarzschild", "schwarzschild_ks")


def _radial_metric(r: Array, mass: float, spacetime: str) -> tuple[Array, Array]:
    """(gamma_rr, beta^r/alpha) for a diagonal spherically-symmetric background at
    coordinate radius r. beta^r/alpha is the only shift combination the u^r
    conversion needs. raises on an unsupported chart."""
    if spacetime == "schwarzschild":
        f = 1.0 - 2.0 * mass / r
        return 1.0 / f, np.zeros_like(np.asarray(r, dtype=float))
    if spacetime == "schwarzschild_ks":
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
    spacetime: str = "schwarzschild_ks",
    *,
    dtheta: Array | float | None = None,
    dphi: float = 2.0 * np.pi,
) -> Array:
    """the rest-mass accretion rate Mdot(r_ex) at every radial shell.

    `rho` has shape `(nr, ntheta)` (axisymmetric) or `(nr,)` (spherical 1D). `v_contra`
    is the list of CONTRAVARIANT valencia velocity component arrays with the same shape.
    `r` is the radial cell centers `(nr,)`; `theta` the polar cell centers `(ntheta,)`
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
        raise ValueError("accretion_rate: 2D rho requires theta cell centers")
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


def cartesian_ks_u_xy(
    v_x: Array,
    v_y: Array,
    x: Array,
    y: Array,
    mass: float,
) -> tuple[Array, Array, Array]:
    """(u^x, u^y, W) from the stored contravariant valencia velocity on the cartesian
    kerr-schild equatorial slice. the spatial metric is NON-diagonal,
    gamma_ij = delta_ij + 2H l_i l_j with H = M/r and l = x_i/r, so the norm carries
    the cross term; the shift is beta^i = (2H/(1+2H)) l^i and alpha = 1/sqrt(1+2H)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v_x = np.asarray(v_x, dtype=float)
    v_y = np.asarray(v_y, dtype=float)
    r = np.sqrt(x * x + y * y)
    two_h = 2.0 * mass / r
    lx, ly = x / r, y / r
    v_dot_l = v_x * lx + v_y * ly
    vv = v_x * v_x + v_y * v_y + two_h * v_dot_l * v_dot_l
    if np.any(vv >= 1.0):
        raise ValueError("valencia 3-velocity is superluminal: gamma_ij v^i v^j >= 1")
    w = 1.0 / np.sqrt(1.0 - vv)
    # beta^i/alpha = (2H/(1+2H)) l^i sqrt(1+2H) = 2H l^i / sqrt(1+2H).
    beta_over_alpha = two_h / np.sqrt(1.0 + two_h)
    u_x = w * (v_x - beta_over_alpha * lx)
    u_y = w * (v_y - beta_over_alpha * ly)
    return u_x, u_y, w


def _bilinear(field: Array, xc: Array, yc: Array, xs: Array, ys: Array) -> Array:
    """bilinear sample of `field` (storage (ny, nx), x fastest) at points (xs, ys),
    given the cell-center coordinate axes xc (nx,) and yc (ny,)."""
    fx = np.clip((xs - xc[0]) / (xc[1] - xc[0]), 0.0, len(xc) - 1.0 - 1e-9)
    fy = np.clip((ys - yc[0]) / (yc[1] - yc[0]), 0.0, len(yc) - 1.0 - 1e-9)
    i0 = fx.astype(int)
    j0 = fy.astype(int)
    tx = fx - i0
    ty = fy - j0
    return (
        field[j0, i0] * (1 - tx) * (1 - ty)
        + field[j0, i0 + 1] * tx * (1 - ty)
        + field[j0 + 1, i0] * (1 - tx) * ty
        + field[j0 + 1, i0 + 1] * tx * ty
    )


def ring_accretion_rate(
    rho: Array,
    v_x: Array,
    v_y: Array,
    xc: Array,
    yc: Array,
    mass: float,
    radii: Sequence[float],
    *,
    n_phi: int = 256,
) -> Array:
    """the rest-mass flux through coordinate circles on the cartesian kerr-schild
    equatorial slice (per unit z-length, the 2D surrogate of Mdot):

        Mdot(r_ex) = - oint sqrt(gamma) alpha rho W (v^i - beta^i/alpha) n_i dl
                   = - oint rho (u^x x + u^y y) dphi

    the second form uses the det-g-flat identity alpha sqrt(gamma) = 1 of the
    kerr-schild chart and n_i dl = x_i dphi on a coordinate circle — no
    densitization factors survive. fields are sampled onto each ring by bilinear
    interpolation from the cell-center grid (`xc`/`yc` the axis coordinates,
    storage (ny, nx) with x fastest). in steady state the continuity equation
    makes the result r_ex-independent for every ring outside the horizon."""
    rho = np.asarray(rho, dtype=float)
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    dphi = 2.0 * np.pi / n_phi
    out = np.empty(len(radii))
    for kk, r_ex in enumerate(radii):
        xs = r_ex * np.cos(phi)
        ys = r_ex * np.sin(phi)
        rho_s = _bilinear(rho, xc, yc, xs, ys)
        vx_s = _bilinear(np.asarray(v_x, dtype=float), xc, yc, xs, ys)
        vy_s = _bilinear(np.asarray(v_y, dtype=float), xc, yc, xs, ys)
        u_x, u_y, _ = cartesian_ks_u_xy(vx_s, vy_s, xs, ys, mass)
        out[kk] = -np.sum(rho_s * (u_x * xs + u_y * ys)) * dphi
    return out


def ring_accretion_from_checkpoint(
    data: Any,
    *,
    mass: float | None = None,
    radii: Sequence[float] | None = None,
    n_phi: int = 256,
) -> tuple[Array, dict[str, Any]]:
    """the excision certificate from an opened cartesian kerr-schild checkpoint
    (`SimData`): ring fluxes at `radii` (default: four radii between the horizon
    and half the domain edge) + the r_ex-invariance summary. fails loud off the
    cartesian kerr-schild chart or on a massless background."""
    meta = data.metadata
    if meta.spacetime != "schwarzschild_ks" or meta.coord_system != "cartesian":
        raise ValueError(
            "ring_accretion_from_checkpoint: needs the cartesian kerr-schild slice; "
            f"got (coords={meta.coord_system}, spacetime={meta.spacetime})"
        )
    bh_mass = mass if mass is not None else meta.schwarzschild_mass
    if bh_mass <= 0.0:
        raise ValueError(
            "ring_accretion_from_checkpoint: black-hole mass must be positive; got "
            f"{bh_mass} (pass mass=... for a checkpoint without the attr)"
        )
    mesh = data.mesh
    x1v = np.asarray(mesh.x1v, dtype=float)
    x2v = np.asarray(mesh.x2v, dtype=float)
    xc = 0.5 * (x1v[:-1] + x1v[1:])
    yc = 0.5 * (x2v[:-1] + x2v[1:])
    rho = data.get_field("rho")
    v_x = data.get_field("v1")
    v_y = data.get_field("v2")

    r_plus = 2.0 * bh_mass
    if radii is None:
        r_edge = 0.5 * min(abs(x1v[0]), x1v[-1], abs(x2v[0]), x2v[-1])
        radii = list(np.linspace(1.25 * r_plus, max(r_edge, 1.5 * r_plus), 4))
    mdot = ring_accretion_rate(rho, v_x, v_y, xc, yc, bh_mass, radii, n_phi=n_phi)
    cert = rex_invariance(mdot, np.asarray(radii, dtype=float), list(radii))
    return mdot, cert


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


# =============================================================================
# the RUNTIME horizon ledger (the first-class immersed-boundary diagnostic)
#
# the excision horizon is auto-created as a first-class BodyKind::Horizon for a
# cartesian kerr-schild excision run. each step the substrate reduces the shell
# flux (a GPU Add-reduction of the outward boundary flux through a coordinate
# sphere at diagnostic_radius) into the body's ledger, checkpointed under the
# `bodies` group. UNLIKE `accretion_rate` above (a post-hoc surface integral),
# this is the flux the scheme ACTUALLY applied, and -- because the code evolves
# the covariant (killing) energy -- Edot is diagnostic_radius-invariant to
# roundoff at steady state.
# =============================================================================


def horizon_ledger(checkpoint: str) -> dict[str, float]:
    """the GR horizon's runtime accretion ledger from a checkpoint's `bodies` group:
    the shell-flux rest-mass rate `mdot` and covariant (killing) energy rate `edot`,
    plus their cumulative totals. the horizon is the MASSLESS body (its gravity is the
    fixed metric, so `mass == 0` distinguishes it from any newtonian sink)."""
    import h5py

    with h5py.File(checkpoint, "r") as f:
        if "bodies" not in f:
            raise ValueError(f"horizon_ledger: '{checkpoint}' has no bodies group")
        g = f["bodies"]
        mass = np.asarray(g["mass"], dtype=float)
        if mass.size == 0:
            raise ValueError("horizon_ledger: no bodies in the checkpoint")
        h = int(np.argmin(mass))  # the massless horizon (metric gravity)
        return {
            "mdot": float(np.asarray(g["accretion_rate"])[h]),
            "edot": float(np.asarray(g["accretion_energy_rate"])[h]),
            "total_accreted_mass": float(np.asarray(g["total_accreted_mass"])[h]),
            "total_accreted_energy": float(np.asarray(g["total_accreted_energy"])[h]),
        }


def horizon_ledger_series(checkpoints: Sequence[str]) -> dict[str, Array]:
    """the horizon accretion time series `(mdot, edot, totals)` over an ORDERED list of
    checkpoint files -- the Mdot(t) / Edot(t) record for the steady-state approach."""
    keys = ("mdot", "edot", "total_accreted_mass", "total_accreted_energy")
    rows = [horizon_ledger(cp) for cp in checkpoints]
    return {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
