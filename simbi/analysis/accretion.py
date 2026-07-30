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


@dataclass(frozen=True)
class DatDiagnostics:
    """the cadence-sampled body-state log (`diagnostics.dat`): one row per body
    per diagnostic interval, appended across the whole run INCLUDING restarts.
    complements the per-step checkpoint series — the cumulative accreted_mass
    column yields exact interval-averaged rates at any cadence, while per-step
    fluctuation amplitudes live only in the checkpoint series."""

    time: NDArray[np.float64]  # [n] sample times
    accreted_mass: NDArray[np.float64]  # [n, nb] cumulative
    accretion_rate: NDArray[np.float64]  # [n, nb] instantaneous (last step)
    force: NDArray[np.float64]  # [n, nb, 3]; nan where the file lacks a column
    mass: NDArray[np.float64]  # [n, nb]


def load_diagnostics_dat(path: str) -> DatDiagnostics:
    """parse `diagnostics.dat`. the header line names the columns, so both the
    current 3-component schema (x y z .. fx fy fz torque_x..) and the legacy
    2d-shaped one (x y .. fx fy torque_z) load; components a file does not
    record come back nan."""
    with open(path) as f:
        first = f.readline()
    if not first.startswith("#"):
        raise ValueError(f"{path}: missing the '# <columns>' header line")
    names = first.lstrip("#").split()
    col = {name: i for i, name in enumerate(names)}
    for required in ("time", "body", "mass", "accreted_mass", "accretion_rate"):
        if required not in col:
            raise ValueError(f"{path}: header lacks required column '{required}'")

    raw = np.loadtxt(path, comments="#", ndmin=2)
    bodies = raw[:, col["body"]].astype(int)
    nb = int(bodies.max()) + 1 if raw.size else 0
    times = np.unique(raw[:, col["time"]])
    n = times.size
    idx = {t: i for i, t in enumerate(times)}
    accreted = np.full((n, nb), np.nan)
    rate = np.full((n, nb), np.nan)
    force = np.full((n, nb, 3), np.nan)
    mass = np.full((n, nb), np.nan)
    for row in raw:
        i, b = idx[row[col["time"]]], int(row[col["body"]])
        for ax, name in enumerate(("fx", "fy", "fz")):
            if name in col:
                force[i, b, ax] = row[col[name]]
        mass[i, b] = row[col["mass"]]
        accreted[i, b] = row[col["accreted_mass"]]
        rate[i, b] = row[col["accretion_rate"]]
    return DatDiagnostics(
        time=times, accreted_mass=accreted, accretion_rate=rate, force=force, mass=mass
    )


def mdot_from_cumulative(
    time: NDArray[np.float64],
    accreted_mass: NDArray[np.float64],
    t_start: float,
) -> float:
    """the EXACT mean accretion rate over (t_start, end] from the cumulative
    accreted-mass column: a difference of totals, immune to the sampling
    cadence (unlike averaging the instantaneous-rate samples, which aliases
    sub-interval variability)."""
    time = np.asarray(time, dtype=np.float64)
    m = np.asarray(accreted_mass, dtype=np.float64)
    sel = np.nonzero(time >= t_start)[0]
    if sel.size < 2:
        return np.nan
    i0, i1 = sel[0], sel[-1]
    return float((m[i1] - m[i0]) / (time[i1] - time[i0]))


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
    if len(time) < 2:
        # np.gradient needs two samples; a shorter series cannot settle.
        return None
    # a NaN sample poisons every window mean it touches, misdiagnosing a settled
    # run as NOT SETTLED — filter with a data-quality warning instead.
    finite = np.isfinite(series) & np.isfinite(time)
    if not finite.all():
        import warnings

        warnings.warn(
            f"steady_state_time: dropping {int((~finite).sum())} non-finite "
            "sample(s) from the series before windowing",
            stacklevel=2,
        )
        time = time[finite]
        series = series[finite]
        if dt is not None:
            dt = np.asarray(dt, dtype=np.float64)[finite]
        if len(time) < 2:
            return None
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


def sphere_flux(
    pos: NDArray[np.float64],
    rho: NDArray[np.float64],
    vel: NDArray[np.float64],
    radii: NDArray[np.float64],
    shell_width: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """the mass and angular-momentum inflow rates through spheres about the
    body: Mdot(r) = -4 pi r^2 <rho v_r> and Ldot_z(r) = -4 pi r^2
    <rho (r x v)_z v_r>, each shell-averaged over cells within
    +- shell_width of the radius. `pos` is [n, ndim] relative to the body,
    `vel` [n, ndim]; the z moment is the only component for 2d data (embedded
    z-hat) and the quotable one for 3d. positive = inflow. the independent
    cross-check of the receipt ledgers (mass_delta/dt and torque_delta):
    receipt == flux is the theorem, and a mismatch is a placement bug, not
    noise. shells with no cells return nan."""
    r = np.sqrt(np.sum(pos**2, axis=1))
    vr = np.sum(pos * vel, axis=1) / np.maximum(r, 1e-300)
    # the z angular momentum needs a plane: 1d data carries mass flux only
    # (Ldot stays 0 when the second position column is absent).
    if pos.shape[1] >= 2:
        lz = rho * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])
    else:
        lz = np.zeros_like(rho)
    mdot = np.full(len(radii), np.nan)
    ldot = np.full(len(radii), np.nan)
    for i, rs in enumerate(np.asarray(radii, dtype=float)):
        shell = (r > rs - shell_width) & (r < rs + shell_width)
        if not np.any(shell):
            continue
        area = 4.0 * np.pi * rs**2
        mdot[i] = -area * np.mean(rho[shell] * vr[shell])
        ldot[i] = -area * np.mean(lz[shell] * vr[shell])
    return mdot, ldot
