# =============================================================================
# frames.py
#
# the observer a stored field is measured against, and the transformations
# between observers on a curved background.
#
# simbi stores the valencia velocity: the fluid three-velocity as measured by the
# EULERIAN (normal) observer, the one whose worldline is orthogonal to the
# constant-time slices. on a flat background that observer is at rest and every
# frame-dependent field means what it looks like it means. on a horizon-penetrating
# chart it is not: the normal observer of ingoing kerr-schild falls inward, and a
# pressure-supported accretion flow always falls SLOWER than free fall, so its speed
# relative to that observer stays small however fast it is moving in any frame an
# observer at infinity would recognize. a michel solution onto a schwarzschild hole
# peaks near mach 0.2 in the valencia frame and never reaches its own sound speed,
# while the transonic surface it does cross is where a STATIC observer measures unity.
#
# this module carries the ADM pieces (lapse, shift, spatial metric) each supported
# background needs, and the observer changes built on them. it reads only the metric
# parameters a checkpoint already records, so a derived field can ask for a quantity
# in a named frame instead of inheriting whichever frame the storage happens to use.
#
# usage:
#   adm = adm_decomposition("schwarzschild_ks", mass=1.0, coords=(x, y, z))
#   W   = eulerian_lorentz(valencia_velocity, adm)
#   u   = four_velocity_from_valencia(valencia_velocity, adm)
#   Ws  = static_lorentz(u, adm)          # NaN where no static observer exists
# =============================================================================
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]

# the backgrounds whose ADM pieces this module carries. a chart outside this set is
# refused by name rather than silently treated as flat, which would report the
# valencia frame under a static observer's label.
SUPPORTED = ("minkowski", "schwarzschild_ks")


@dataclass(frozen=True)
class Adm:
    """the 3+1 pieces of a background, broadcast against the field arrays.

    `alpha` is the lapse, `beta` the contravariant shift `beta^i`, and `gamma` the
    spatial metric as the pair `(delta_ij part, l_i l_j coefficient)` — every chart
    here is conformally flat plus a rank-one radial term, `gamma_ij = delta_ij + H l_i l_j`,
    so storing `H` and the radial unit covector is exact and avoids materializing a
    full tensor per cell. `f` is `-g_tt`, which is what a static observer normalizes
    against and what vanishes at the horizon."""

    alpha: Array
    beta: tuple[Array, ...]
    l: tuple[Array, ...]
    h: Array
    f: Array
    radius: Array


def adm_decomposition(
    spacetime: str, mass: float, coords: Sequence[Array]
) -> Adm:
    """the lapse, shift and spatial metric of `spacetime` at cartesian `coords`.

    ingoing kerr-schild for a schwarzschild hole, in cartesian coordinates, is
    `g = eta + H l l` with `H = 2M/r` and `l_mu = (1, x/r, y/r, z/r)`, giving
    `alpha = 1/sqrt(1+H)`, `beta^i = H l^i/(1+H)` and `gamma_ij = delta_ij + H l_i l_j`.
    the chart is regular across `r = 2M`, which is what lets the grid carry the horizon."""
    if spacetime not in SUPPORTED:
        raise ValueError(
            f"frames: no ADM decomposition for spacetime {spacetime!r}; "
            f"supported: {', '.join(SUPPORTED)}. a kerr background needs the "
            "spin-dependent kerr-schild form, whose shift carries an azimuthal "
            "component this module does not build"
        )
    r = np.sqrt(sum(np.asanyarray(c) ** 2 for c in coords))
    # the origin is a coordinate singularity of the radial covector; the excised
    # interior is masked downstream, so leaving it NaN keeps it out of the arithmetic.
    r_safe = np.where(r > 0.0, r, np.nan)
    l = tuple(np.asanyarray(c) / r_safe for c in coords)
    if spacetime == "minkowski" or mass <= 0.0:
        zero = np.zeros_like(r)
        one = np.ones_like(r)
        return Adm(one, tuple(zero for _ in coords), l, zero, one, r)
    h = 2.0 * mass / r_safe
    alpha = 1.0 / np.sqrt(1.0 + h)
    beta = tuple(h * li / (1.0 + h) for li in l)
    return Adm(alpha, beta, l, h, 1.0 - h, r)


def _lower(v: Sequence[Array], adm: Adm) -> tuple[Array, ...]:
    """`gamma_ij v^j` for the rank-one-plus-flat spatial metric."""
    vl = sum(vi * li for vi, li in zip(v, adm.l))
    return tuple(vi + adm.h * li * vl for vi, li in zip(v, adm.l))


def eulerian_lorentz(v: Sequence[Array], adm: Adm) -> Array:
    """the lorentz factor of the stored valencia velocity against the normal observer,
    `W = 1/sqrt(1 - gamma_ij v^i v^j)`. the spatial metric is what makes this differ from
    the flat expression: a radial velocity is stretched by `sqrt(1 + H)` in proper length."""
    v_low = _lower(v, adm)
    vsq = sum(a * b for a, b in zip(v, v_low))
    return np.asanyarray(1.0 / np.sqrt(np.maximum(1.0 - vsq, 1.0e-300)))


def four_velocity_from_valencia(
    v: Sequence[Array], adm: Adm
) -> tuple[Array, tuple[Array, ...]]:
    """the four-velocity `(u^t, u^i)` of a fluid whose valencia velocity is `v`.

    `u^mu = W (n^mu + v^mu)` with the normal `n^mu = (1/alpha, -beta^i/alpha)`, so
    `u^t = W/alpha` and `u^i = W (v^i - beta^i/alpha)`. the shift term is the whole
    difference between this and the flat `W v^i`, and it is what a frame-blind field
    silently drops on a horizon-penetrating chart."""
    w = eulerian_lorentz(v, adm)
    ut = w / adm.alpha
    ui = tuple(w * (vi - bi / adm.alpha) for vi, bi in zip(v, adm.beta))
    return np.asanyarray(ut), ui


def static_lorentz(
    ut: Array, ui: Sequence[Array], adm: Adm
) -> Array:
    """the lorentz factor a STATIC observer measures, `W_s = -u_t / sqrt(f)`.

    the static observer follows the timelike killing vector, so `u_static^mu` is
    `(1/sqrt(f), 0, 0, 0)` and the contraction reduces to the conserved `u_t`. in
    kerr-schild that is `u_t = -(1 - H) u^t + H l_i u^i`, the off-diagonal term being
    the chart's own; in schwarzschild coordinates the same quantity is `-f u^t`.

    static observers exist only outside the horizon: `f <= 0` at and inside `r = 2M`,
    where the killing vector is null or spacelike and no worldline stays at fixed r.
    the factor is NaN there rather than continued, so a field built on it reports the
    absence of the frame instead of a number belonging to no observer."""
    u_t = -(1.0 - adm.h) * ut + adm.h * sum(li * uii for li, uii in zip(adm.l, ui))
    with np.errstate(invalid="ignore", divide="ignore"):
        w_static = -u_t / np.sqrt(adm.f)
    return np.asanyarray(np.where(adm.f > 0.0, w_static, np.nan))


def speed_from_lorentz(w: Array) -> Array:
    """the three-speed `|v| = sqrt(1 - 1/W^2)` of a lorentz factor, clamped at rest."""
    return np.asanyarray(np.sqrt(np.maximum(1.0 - 1.0 / (w * w), 0.0)))
