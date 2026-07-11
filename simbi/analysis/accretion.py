# =============================================================================
# accretion.py
#
# accretor validation diagnostics (docs/ideas/accretor.md §5):
# - the per-step body-gas exchange series from a checkpoint's
#   `body_diagnostics` group: Mdot(t) = mass_delta/dt, drag F_acc(t)
# - steady-state detection: consecutive dt-weighted window means of Mdot
#   agreeing to a relative tolerance
# - windowed averaging with a fluctuation amplitude (reported together;
#   an unsteady flow gets a mean AND an amplitude, never a bare value)
# - sonic-surface radius vs angle and the on-axis stagnation distance,
#   as pure array functions (the caller extracts cell arrays via the reader)
# usage:
#  diag = load_body_diagnostics(path)
#  t0 = steady_state_time(diag.time, diag.mdot[:, 0])
#  mean, fluct, span = averaged_rate(diag.time, diag.dt, diag.mdot[:, 0], t0)
# =============================================================================

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BodyDiagnostics:
    """the per-step body-gas exchange series of one run segment."""

    time: NDArray[np.float64]  # [n] step-end times
    dt: NDArray[np.float64]  # [n]
    mass_delta: NDArray[np.float64]  # [n, nb] mass removed from the gas
    energy_delta: NDArray[np.float64]  # [n, nb]
    force: NDArray[np.float64]  # [n, nb, ndim]

    @property
    def mdot(self) -> NDArray[np.float64]:
        """the emergent accretion rate per body, [n, nb]."""
        return self.mass_delta / self.dt[:, None]


def load_body_diagnostics(path: str) -> BodyDiagnostics:
    """read the `body_diagnostics` group of a checkpoint."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f["body_diagnostics"]
        return BodyDiagnostics(
            time=np.asarray(g["time"]),
            dt=np.asarray(g["dt"]),
            mass_delta=np.asarray(g["mass_delta"]),
            energy_delta=np.asarray(g["energy_delta"]),
            force=np.asarray(g["force"]),
        )


def _window_mean(
    time: NDArray[np.float64],
    dt: NDArray[np.float64],
    series: NDArray[np.float64],
    t_lo: float,
    t_hi: float,
) -> float:
    """the dt-weighted mean of a per-step series over (t_lo, t_hi]."""
    sel = (time > t_lo) & (time <= t_hi)
    if not np.any(sel):
        return np.nan
    w = dt[sel]
    return float(np.sum(series[sel] * w) / np.sum(w))


def steady_state_time(
    time: NDArray[np.float64],
    series: NDArray[np.float64],
    dt: NDArray[np.float64] | None = None,
    window: float = 5.0,
    tol: float = 0.01,
) -> float | None:
    """the earliest time at which the series is steady: the dt-weighted mean
    over the trailing `window` agrees with the mean over the window before it
    to relative tolerance `tol`. returns None if the series never settles
    (or is shorter than two windows). `window` is in code time units
    (t_B = 1 for the accretor problem)."""
    time = np.asarray(time, dtype=np.float64)
    series = np.asarray(series, dtype=np.float64)
    w = np.asarray(dt, dtype=np.float64) if dt is not None else np.gradient(time)
    for t in time:
        if t - 2.0 * window < time[0]:
            continue
        m_prev = _window_mean(time, w, series, t - 2.0 * window, t - window)
        m_last = _window_mean(time, w, series, t - window, t)
        if np.isnan(m_prev) or np.isnan(m_last):
            continue
        scale = max(abs(m_prev), abs(m_last), 1e-300)
        if abs(m_last - m_prev) < tol * scale:
            return float(t)
    return None


def averaged_rate(
    time: NDArray[np.float64],
    dt: NDArray[np.float64],
    series: NDArray[np.float64],
    t_start: float,
) -> tuple[float, float, float]:
    """the dt-weighted mean, fluctuation amplitude (weighted standard
    deviation), and averaging span of the series over t > t_start. the spec
    wants span >= 10 t_B for a quotable rate — the caller checks the returned
    span and runs longer if it is short."""
    time = np.asarray(time, dtype=np.float64)
    sel = time > t_start
    if not np.any(sel):
        return np.nan, np.nan, 0.0
    w = np.asarray(dt, dtype=np.float64)[sel]
    s = np.asarray(series, dtype=np.float64)[sel]
    mean = float(np.sum(s * w) / np.sum(w))
    var = float(np.sum(w * (s - mean) ** 2) / np.sum(w))
    span = float(time[sel][-1] - t_start)
    return mean, np.sqrt(var), span


def sonic_radius_vs_angle(
    pos: NDArray[np.float64],
    speed: NDArray[np.float64],
    cs: NDArray[np.float64],
    nbins: int = 32,
    axis: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """the sonic-surface radius per angular bin: in each bin of the polar
    angle from the wind axis, the INNERMOST radius where the mach number
    crosses 1 from the supersonic interior to the subsonic exterior
    (linear interpolation between the bracketing cells). `pos` is [n, ndim]
    relative to the body. bins with no crossing return nan (e.g. a fully
    subsonic direction). the well-posedness check is r_mask < min over bins."""
    pos = np.asarray(pos, dtype=np.float64)
    r = np.linalg.norm(pos, axis=1)
    mach = np.asarray(speed, dtype=np.float64) / np.asarray(cs, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arccos(np.clip(pos[:, axis] / np.where(r > 0, r, np.inf), -1.0, 1.0))
    edges = np.linspace(0.0, np.pi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_sonic = np.full(nbins, np.nan)
    for b in range(nbins):
        sel = (theta >= edges[b]) & (theta < edges[b + 1]) & (r > 0)
        if not np.any(sel):
            continue
        order = np.argsort(r[sel])
        rb, mb = r[sel][order], mach[sel][order]
        crossing = np.nonzero((mb[:-1] >= 1.0) & (mb[1:] < 1.0))[0]
        if crossing.size == 0:
            continue
        i = crossing[0]
        # linear interpolation in mach between the bracketing radii.
        f = (mb[i] - 1.0) / (mb[i] - mb[i + 1])
        r_sonic[b] = rb[i] + f * (rb[i + 1] - rb[i])
    return centers, r_sonic


def stagnation_distance(
    x: NDArray[np.float64],
    u_axial: NDArray[np.float64],
) -> float | None:
    """the on-axis stagnation distance: the zero crossing of the axial
    velocity along the (sorted) upstream axis samples, linearly interpolated.
    None if the velocity never changes sign on the sampled axis."""
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u_axial, dtype=np.float64)
    order = np.argsort(x)
    x, u = x[order], u[order]
    sign_change = np.nonzero(u[:-1] * u[1:] < 0.0)[0]
    if sign_change.size == 0:
        return None
    i = sign_change[0]
    f = u[i] / (u[i] - u[i + 1])
    return float(abs(x[i] + f * (x[i + 1] - x[i])))
