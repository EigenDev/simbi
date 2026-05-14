# =============================================================================
# spectrum.py
#
# spectral analysis functions for simulation data.
# shell-averaged kinetic energy power spectrum, lomb-scargle PSD,
# welch-style segment-averaged lomb-scargle, and false-alarm probability.
# pure numpy/scipy — no viz dependency.
#
# usage:
#   from simbi.analysis import shell_averaged_spectrum, lomb_scargle_psd
#   k, Ek = shell_averaged_spectrum(vx, vy, vz, dx)
#   omega, psd = lomb_scargle_psd(times, values)
#   omega, psd = welch_lomb_scargle_psd(times, values, n_segments=8)
#   thresholds = lomb_scargle_fap_levels(100, 1024)
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from scipy.signal import lombscargle
from scipy.stats import binned_statistic

if TYPE_CHECKING:
    from simbi.reader.adapter import SimData


def _subtract_radial_mean(field: np.ndarray, dx: float) -> np.ndarray:
    """subtract the spherically-averaged radial profile from a 3D field.

    computes <f>(r) by binning cells by distance from the domain center,
    then subtracts the interpolated profile from each cell. isolates
    fluctuations from the smooth background gradient.
    """
    nx, ny, nz = field.shape
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0

    ii = np.arange(nx) - cx
    jj = np.arange(ny) - cy
    kk = np.arange(nz) - cz
    r = np.sqrt(
        (ii[:, None, None] * dx) ** 2
        + (jj[None, :, None] * dx) ** 2
        + (kk[None, None, :] * dx) ** 2
    )

    # bin by radius
    r_max = r.max()
    n_bins = max(nx, ny, nz) // 2
    bin_edges = np.linspace(0, r_max, n_bins + 1)

    result = binned_statistic(
        r.ravel(), field.ravel(), statistic="mean", bins=bin_edges
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean_profile = result.statistic

    # fill nans with nearest valid value
    valid = ~np.isnan(mean_profile)
    if not np.any(valid):
        return field
    mean_profile = np.interp(bin_centers, bin_centers[valid], mean_profile[valid])

    # interpolate back to each cell and subtract
    radial_mean = np.interp(r.ravel(), bin_centers, mean_profile).reshape(field.shape)
    return field - radial_mean


def shell_averaged_spectrum(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dx: float,
    subtract_mean: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    compute shell-averaged kinetic energy spectrum from 3D velocity fields.

    args:
        vx, vy, vz: 3D velocity component arrays (same shape)
        dx: uniform cell spacing
        subtract_mean: if true, subtract spherical mean profile before FFT

    returns:
        (k_centers, E_k): wavenumber bin centers and spectrum values
    """
    if subtract_mean:
        vx = _subtract_radial_mean(vx, dx)
        vy = _subtract_radial_mean(vy, dx)
        vz = _subtract_radial_mean(vz, dx)

    nx, ny, nz = vx.shape
    n_cells = nx * ny * nz

    # 3D real FFT of each velocity component
    vx_hat = np.fft.rfftn(vx)
    vy_hat = np.fft.rfftn(vy)
    vz_hat = np.fft.rfftn(vz)

    # kinetic energy density in fourier space
    energy = 0.5 * (
        np.abs(vx_hat) ** 2 + np.abs(vy_hat) ** 2 + np.abs(vz_hat) ** 2
    )

    # parseval normalization
    energy /= n_cells**2

    # wavenumber grid
    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dx) * 2.0 * np.pi
    kz = np.fft.rfftfreq(nz, d=dx) * 2.0 * np.pi

    k_mag = np.sqrt(
        kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    )

    # shell-average into radial wavenumber bins
    dk = 2.0 * np.pi / (nx * dx)
    k_max = k_mag.max()
    bin_edges = np.arange(dk, k_max + dk, dk)

    result = binned_statistic(
        k_mag.ravel(),
        energy.ravel(),
        statistic="sum",
        bins=bin_edges,
    )

    k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    e_k = result.statistic

    # drop bins with no data
    valid = ~np.isnan(e_k) & (e_k > 0)
    return k_centers[valid], e_k[valid]


def shell_averaged_scalar_spectrum(
    field: np.ndarray,
    dx: float,
    subtract_mean: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    compute shell-averaged power spectrum of a scalar field.

    args:
        field: 3D scalar array
        dx: uniform cell spacing
        subtract_mean: if true, subtract spherical mean profile before FFT

    returns:
        (k_centers, P_k): wavenumber bin centers and spectrum values
    """
    if subtract_mean:
        field = _subtract_radial_mean(field, dx)

    nx, ny, nz = field.shape
    n_cells = nx * ny * nz

    f_hat = np.fft.rfftn(field)
    power = np.abs(f_hat) ** 2 / n_cells**2

    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dx) * 2.0 * np.pi
    kz = np.fft.rfftfreq(nz, d=dx) * 2.0 * np.pi

    k_mag = np.sqrt(
        kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    )

    dk = 2.0 * np.pi / (nx * dx)
    k_max = k_mag.max()
    bin_edges = np.arange(dk, k_max + dk, dk)

    result = binned_statistic(
        k_mag.ravel(),
        power.ravel(),
        statistic="sum",
        bins=bin_edges,
    )

    k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    p_k = result.statistic

    valid = ~np.isnan(p_k) & (p_k > 0)
    return k_centers[valid], p_k[valid]


def lomb_scargle_psd(
    times: np.ndarray,
    values: np.ndarray,
    orbital_period: Optional[float] = None,
    n_freqs: int = 1024,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    compute power spectral density via lomb-scargle periodogram.

    handles non-uniform sampling natively. evaluates at n_freqs
    angular frequencies between the minimum resolvable frequency
    and the nyquist estimate.

    args:
        times: 1d array of sample times (need not be uniform)
        values: 1d array of signal values
        orbital_period: if given, normalize output frequencies to Omega_orb
        n_freqs: number of frequency points to evaluate
        normalize: if true, divide PSD by its integral so it sums to 1

    returns:
        (omega, psd): angular frequencies and corresponding power
    """
    # subtract mean (lomb-scargle assumes zero-mean for power interpretation)
    signal = values - np.mean(values)

    # frequency bounds
    t_span = times[-1] - times[0]
    dt_min = np.min(np.diff(times))
    omega_min = 2.0 * np.pi / t_span
    omega_max = np.pi / dt_min  # nyquist-like estimate

    omega = np.linspace(omega_min, omega_max, n_freqs)

    # scipy.signal.lombscargle wants raw times, signal, angular freqs
    # and returns the periodogram (unnormalized by default)
    psd = lombscargle(times, signal, omega, precenter=False, normalize=False)

    # normalize: divide by N/2 to get power consistent with FFT convention
    n = len(signal)
    psd = psd * 2.0 / n

    # normalize frequency axis
    if orbital_period is not None and orbital_period > 0:
        omega_orb = 2.0 * np.pi / orbital_period
        omega = omega / omega_orb

    if normalize:
        total = np.trapezoid(psd, omega)
        if total > 0:
            psd = psd / total

    return omega, psd


def welch_lomb_scargle_psd(
    times: np.ndarray,
    values: np.ndarray,
    orbital_period: Optional[float] = None,
    n_segments: int = 8,
    overlap: float = 0.5,
    n_freqs: int = 1024,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    segment-averaged lomb-scargle PSD (welch-style variance reduction).

    splits the time series into overlapping segments, computes
    lomb-scargle PSD per segment on a shared frequency grid,
    and averages. reduces variance at the cost of frequency resolution.

    args:
        times: 1d array of sample times
        values: 1d array of signal values
        orbital_period: if given, normalize output frequencies to Omega_orb
        n_segments: number of segments
        overlap: fractional overlap between segments (0 to <1)
        n_freqs: number of frequency points to evaluate
        normalize: if true, divide PSD by its integral

    returns:
        (omega, psd): angular frequencies and averaged power
    """
    n = len(times)
    if n_segments < 2:
        return lomb_scargle_psd(
            times, values, orbital_period, n_freqs, normalize
        )

    step = max(1, int(n * (1.0 - overlap) / n_segments))
    seg_len = max(2, int(n / n_segments * (1.0 + overlap)))

    # shared frequency grid from the full time span
    t_span = times[-1] - times[0]
    dt_min = np.min(np.diff(times))
    omega_min = 2.0 * np.pi / t_span
    omega_max = np.pi / dt_min
    omega = np.linspace(omega_min, omega_max, n_freqs)

    psd_sum = np.zeros(n_freqs)
    count = 0

    for ii in range(0, n - seg_len + 1, step):
        seg_t = times[ii : ii + seg_len]
        seg_v = values[ii : ii + seg_len]
        sig = seg_v - np.mean(seg_v)
        seg_psd = lombscargle(
            seg_t, sig, omega, precenter=False, normalize=False
        )
        seg_psd = seg_psd * 2.0 / len(sig)
        psd_sum += seg_psd
        count += 1

    if count == 0:
        return lomb_scargle_psd(
            times, values, orbital_period, n_freqs, normalize
        )

    psd = psd_sum / count

    if orbital_period is not None and orbital_period > 0:
        omega_orb = 2.0 * np.pi / orbital_period
        omega = omega / omega_orb

    if normalize:
        total = np.trapezoid(psd, omega)
        if total > 0:
            psd = psd / total

    return omega, psd


def lomb_scargle_fap_levels(
    n_samples: int,
    n_freqs: int,
    levels: Sequence[float] = (0.01, 0.001),
    psd_normalization: float = 1.0,
) -> dict[float, float]:
    """
    compute PSD thresholds for given false-alarm probability levels.

    uses the baluev (2008) approximation where the number of
    independent frequencies M ~ n_freqs and the exponential
    distribution of the normalized periodogram gives:
        z = -ln(1 - (1-p)^(1/M))

    args:
        n_samples: number of data points in the time series
        n_freqs: number of frequency points evaluated
        levels: false-alarm probabilities (e.g., 0.01 = 1% FAP)
        psd_normalization: factor to convert from normalized z to
            actual PSD units (2/N for our convention)

    returns:
        dict mapping FAP level to PSD threshold
    """
    m_eff = n_freqs
    result = {}
    for p in levels:
        # single-frequency survival probability
        q = (1.0 - p) ** (1.0 / m_eff)
        z = -np.log(1.0 - q)
        # convert from normalized periodogram to our PSD units
        result[p] = z * psd_normalization
    return result


def phase_fold(
    times: np.ndarray,
    values: np.ndarray,
    period: float,
    n_bins: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    fold a time series on a given period and compute binned statistics.

    args:
        times: shape (N,) time array
        values: shape (N,) or (N, M) signal values
        period: folding period
        n_bins: number of phase bins

    returns:
        (phase_centers, mean, std, phase_per_sample)
        phase_centers: shape (n_bins,) bin centers in [0, 1)
        mean: shape (n_bins,) or (n_bins, M) binned mean
        std: shape (n_bins,) or (n_bins, M) binned std
        phase_per_sample: shape (N,) raw phase of each sample
    """
    phase = (times % period) / period
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    digit = np.digitize(phase, bin_edges) - 1
    digit = np.clip(digit, 0, n_bins - 1)

    is_2d = values.ndim == 2
    ncols = values.shape[1] if is_2d else 1
    vals = values if is_2d else values[:, np.newaxis]

    mean = np.full((n_bins, ncols), np.nan)
    std = np.full((n_bins, ncols), np.nan)
    for bb in range(n_bins):
        mask = digit == bb
        if mask.sum() > 0:
            mean[bb] = vals[mask].mean(axis=0)
            std[bb] = vals[mask].std(axis=0)

    if not is_2d:
        mean = mean[:, 0]
        std = std[:, 0]

    return centers, mean, std, phase


# =============================================================================
# composite AMR spectra (band-stitched per level)
# =============================================================================


def _k_nyquist_per_level(data: SimData) -> list[float]:
    """compute per-level nyquist wavenumbers for annotation."""
    hierarchy = data.hierarchy()
    base_mesh = data.checkpoint.levels[0].mesh
    dx_coarse = (
        base_mesh.dims[-1][1] - base_mesh.dims[-1][0]
    ) / base_mesh.global_cells[-1]
    k_nyquist = []
    cumulative_ratio = 1
    for lvl in range(data.num_levels):
        if lvl > 0:
            cumulative_ratio *= hierarchy.ref_ratios[lvl - 1]
        k_nyquist.append(np.pi / (dx_coarse / cumulative_ratio))
    return k_nyquist


def _level_field(
    data: SimData, field_name: str, level: int
) -> tuple[np.ndarray, float]:
    """extract a field from a specific FMR level with its cell spacing."""
    field = data.get_field(field_name, level=level)
    mesh = data.checkpoint.levels[level].mesh
    dx = (mesh.dims[-1][1] - mesh.dims[-1][0]) / mesh.global_cells[-1]
    return field, float(dx)


def _rescale_and_concat(
    spectra: list[tuple[np.ndarray, np.ndarray]],
    k_boundaries: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """stitch per-level spectra into one using non-overlapping k bands.

    each level contributes the k range [k_boundaries[lvl], k_boundaries[lvl+1]).
    adjacent levels are rescaled in their overlap region so amplitudes
    match continuously across boundaries.

    args:
        spectra: list of (k, spectrum) per level, coarsest first
        k_boundaries: N+1 boundaries for N levels. level ii owns
            [k_boundaries[ii], k_boundaries[ii+1]).
    """
    n_levels = len(spectra)
    scales = [1.0] * n_levels

    # chain rescaling: match each finer level to its coarser neighbor
    for ii in range(1, n_levels):
        k_prev, s_prev = spectra[ii - 1]
        k_curr, s_curr = spectra[ii]

        # overlap = region where both levels have data
        k_lo = k_boundaries[ii] * 0.5
        k_hi = k_boundaries[ii]

        mask_prev = (k_prev >= k_lo) & (k_prev < k_hi)
        mask_curr = (k_curr >= k_lo) & (k_curr < k_hi)

        if np.any(mask_prev) and np.any(mask_curr):
            mean_prev = np.exp(np.mean(np.log(s_prev[mask_prev])))
            mean_curr = np.exp(np.mean(np.log(s_curr[mask_curr])))
            scales[ii] = scales[ii - 1] * (mean_prev / mean_curr if mean_curr > 0 else 1.0)
        else:
            scales[ii] = scales[ii - 1]

    # collect non-overlapping bands
    k_parts = []
    s_parts = []
    for ii in range(n_levels):
        k_lvl, s_lvl = spectra[ii]
        lo = k_boundaries[ii]
        hi = k_boundaries[ii + 1]
        mask = (k_lvl >= lo) & (k_lvl < hi)
        if np.any(mask):
            k_parts.append(k_lvl[mask])
            s_parts.append(s_lvl[mask] * scales[ii])

    if not k_parts:
        # fallback: return coarsest level unmodified
        return spectra[0]

    return np.concatenate(k_parts), np.concatenate(s_parts)


def _level_k_boundaries(data: SimData) -> list[float]:
    """compute k band boundaries for each FMR level.

    level ii owns [k_fund(ii), k_fund(ii+1)).
    the finest level extends to its nyquist.
    """
    hierarchy = data.hierarchy()
    base_mesh = data.checkpoint.levels[0].mesh
    dx_coarse = (
        base_mesh.dims[-1][1] - base_mesh.dims[-1][0]
    ) / base_mesh.global_cells[-1]

    # fundamental wavenumber of each level = 2*pi / L_level
    # where L_level is the physical extent of that level's domain
    boundaries = []
    cumulative_ratio = 1

    for lvl in range(data.num_levels):
        if lvl > 0:
            cumulative_ratio *= hierarchy.ref_ratios[lvl - 1]
        mesh = data.checkpoint.levels[lvl].mesh
        extent = mesh.dims[-1][1] - mesh.dims[-1][0]
        k_fund = 2.0 * np.pi / extent
        boundaries.append(k_fund)

    # upper bound for finest level = its nyquist
    dx_finest = dx_coarse / cumulative_ratio
    boundaries.append(np.pi / dx_finest)
    return boundaries


def composite_shell_averaged_spectrum(
    data: SimData,
    fields: Sequence[str] = ("v1", "v2", "v3"),
    subtract_mean: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """kinetic energy spectrum stitched from all FMR levels.

    computes spectrum per level independently, then stitches in k-space.
    each level contributes its authoritative k band (from its fundamental
    mode to the next finer level's fundamental mode). no prolongation
    needed — memory cost is O(max level size), not O(finest uniform grid).

    returns:
        (k_centers, E_k, k_nyquist_per_level)
    """
    spectra = []
    for lvl in range(data.num_levels):
        vx, dx = _level_field(data, fields[0], lvl)
        vy, _ = _level_field(data, fields[1], lvl)
        vz, _ = _level_field(data, fields[2], lvl)
        k, ek = shell_averaged_spectrum(vx, vy, vz, dx, subtract_mean=subtract_mean)
        spectra.append((k, ek))

    boundaries = _level_k_boundaries(data)
    k_out, ek_out = _rescale_and_concat(spectra, boundaries)
    return k_out, ek_out, _k_nyquist_per_level(data)


def composite_shell_averaged_scalar_spectrum(
    data: SimData,
    field: str,
    subtract_mean: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """scalar power spectrum stitched from all FMR levels.

    returns:
        (k_centers, P_k, k_nyquist_per_level)
    """
    spectra = []
    for lvl in range(data.num_levels):
        f, dx = _level_field(data, field, lvl)
        k, pk = shell_averaged_scalar_spectrum(f, dx, subtract_mean=subtract_mean)
        spectra.append((k, pk))

    boundaries = _level_k_boundaries(data)
    k_out, pk_out = _rescale_and_concat(spectra, boundaries)
    return k_out, pk_out, _k_nyquist_per_level(data)


# =============================================================================
# angular power spectrum C_l(r) via spherical harmonic decomposition
# =============================================================================


def _interpolate_to_shell(
    stitched_data: dict[str, np.ndarray],
    field_name: str,
    radius: float,
    n_theta: int,
    n_phi: int,
    shell_width: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """interpolate AMR leaf-cell data onto a regular (theta, phi) grid at given radius.

    selects cells within a radial shell, then maps onto a uniform
    equirectangular grid using nearest-neighbor interpolation.

    args:
        stitched_data: flat arrays from stitch_leaf_cells
        field_name: key prefix (looked up as "{field_name}_flat")
        radius: target shell radius
        n_theta: polar resolution (0, pi)
        n_phi: azimuthal resolution (0, 2*pi)
        shell_width: half-width of the shell. if 0, auto-computed from
            the typical cell spacing in the shell.

    returns:
        (theta_centers, phi_centers, field_on_shell) where field_on_shell
        has shape (n_theta, n_phi).
    """
    from scipy.interpolate import griddata

    x = stitched_data["x_flat"]
    y = stitched_data["y_flat"]
    z = stitched_data.get("z_flat", np.zeros_like(x))
    field = stitched_data[f"{field_name}_flat"]

    r = np.sqrt(x**2 + y**2 + z**2)

    # auto shell width: estimate from cell spacing near the target radius
    if shell_width <= 0:
        vol = stitched_data["volume_flat"]
        dx_est = np.median(vol ** (1.0 / 3.0))
        shell_width = max(2.0 * dx_est, 0.02 * radius)

    mask = np.abs(r - radius) < shell_width
    if np.sum(mask) < 10:
        return (
            np.zeros(0),
            np.zeros(0),
            np.zeros((n_theta, n_phi)),
        )

    x_sel, y_sel, z_sel = x[mask], y[mask], z[mask]
    f_sel = field[mask]

    # convert to spherical coords
    theta_sel = np.arccos(np.clip(z_sel / np.sqrt(x_sel**2 + y_sel**2 + z_sel**2 + 1e-30), -1, 1))
    phi_sel = np.arctan2(y_sel, x_sel) + np.pi  # shift to [0, 2*pi)

    # target grid
    d_theta = np.pi / n_theta
    d_phi = 2.0 * np.pi / n_phi
    theta_centers = np.linspace(d_theta / 2, np.pi - d_theta / 2, n_theta)
    phi_centers = np.linspace(d_phi / 2, 2.0 * np.pi - d_phi / 2, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers, indexing="ij")

    shell_field = griddata(
        (theta_sel, phi_sel),
        f_sel,
        (theta_grid, phi_grid),
        method="nearest",
    )

    return theta_centers, phi_centers, shell_field


def _angular_spectrum_from_shell(
    field_on_shell: np.ndarray,
    theta_centers: np.ndarray,
    subtract_mean: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """compute angular power spectrum C_l from a field on an equirectangular grid.

    uses 2D FFT with sin(theta) quadrature weighting. this is an
    approximate SHT that preserves power-law slopes accurately.

    args:
        field_on_shell: shape (n_theta, n_phi)
        theta_centers: 1d array of polar angle cell centers
        subtract_mean: if true, remove the l=0 (monopole) mode

    returns:
        (ell, C_ell): multipole moments and angular power spectrum
    """
    n_theta, n_phi = field_on_shell.shape

    if subtract_mean:
        # volume-weighted mean on the sphere
        w = np.sin(theta_centers)
        mean = np.sum(field_on_shell * w[:, None]) / np.sum(w) / n_phi
        field_on_shell = field_on_shell - mean

    # quadrature weight for equirectangular grid
    w = np.sin(theta_centers)
    weighted = field_on_shell * w[:, None]

    # 2D FFT
    f_hat = np.fft.fft2(weighted) / (n_theta * n_phi)

    # power in each (l_idx, m_idx)
    power = np.abs(f_hat) ** 2

    # map theta-axis fft index to multipole l
    # l ranges from 0 to n_theta//2 (nyquist)
    l_max = n_theta // 2

    # sum over m (phi-axis) for each l (theta-axis)
    ell = np.arange(1, l_max + 1)  # skip l=0 (monopole)
    c_ell = np.zeros(l_max)

    for ll in range(1, l_max + 1):
        # positive and negative frequency contributions for l
        c_ell[ll - 1] = np.sum(power[ll, :]) + np.sum(power[-ll, :])

    # normalize by (2l+1)
    c_ell /= (2.0 * ell + 1.0)

    return ell, c_ell


def angular_power_spectrum(
    stitched_data: dict[str, np.ndarray],
    field_name: str,
    radii: Sequence[float],
    n_theta: int = 64,
    n_phi: int = 128,
    subtract_mean: bool = False,
    shell_width: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """compute angular power spectrum C_l averaged over multiple radial shells.

    for each shell at radius r:
      - interpolates the field onto a (theta, phi) grid
      - optionally subtracts the shell mean (l=0 monopole)
      - computes C_l via 2D FFT with sin(theta) weighting
      - averages C_l across all requested shells

    args:
        stitched_data: flat arrays from stitch_leaf_cells
        field_name: field key prefix
        radii: shell radii to average over
        n_theta: polar resolution (sets l_max ~ n_theta/2)
        n_phi: azimuthal resolution (should be >= 2*n_theta)
        subtract_mean: remove monopole before computing spectrum
        shell_width: half-width of each radial shell (0 = auto)

    returns:
        (ell, C_ell): multipole moments and averaged angular power spectrum
    """
    c_ell_sum = None
    count = 0

    for radius in radii:
        theta, phi, shell_field = _interpolate_to_shell(
            stitched_data, field_name, radius, n_theta, n_phi, shell_width
        )
        if len(theta) == 0:
            continue

        ell, c_ell = _angular_spectrum_from_shell(
            shell_field, theta, subtract_mean=subtract_mean
        )

        if c_ell_sum is None:
            c_ell_sum = c_ell.copy()
        else:
            c_ell_sum += c_ell
        count += 1

    if c_ell_sum is None or count == 0:
        return np.arange(1, n_theta // 2 + 1, dtype=float), np.zeros(n_theta // 2)

    return ell.astype(float), c_ell_sum / count


def composite_angular_power_spectrum(
    data: "SimData",
    field_name: str,
    radii: Optional[Sequence[float]] = None,
    n_shells: int = 5,
    n_theta: int = 64,
    n_phi: int = 128,
    subtract_mean: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """angular power spectrum using stitched leaf cells from all AMR levels.

    convenience wrapper that handles SimData -> stitch -> angular spectrum.

    args:
        data: simulation checkpoint
        field_name: primitive field name (e.g., "rho", "entropy-measure")
        radii: explicit shell radii. if None, auto-selects n_shells
            logarithmically-spaced shells across the domain.
        n_shells: number of shells when radii is None
        n_theta: polar resolution
        n_phi: azimuthal resolution
        subtract_mean: remove monopole per shell

    returns:
        (ell, C_ell, r_mean): multipole moments, angular power, and
        geometric mean shell radius (for converting ell -> k = ell/r_mean)
    """
    from simbi.analysis.radial_profiles import stitch_leaf_cells

    # build stitch args matching prepare_field_level domain ordering:
    # [x3v, x2v, x1v] so stitch_leaf_cells produces x_flat=x3, y_flat=x2, z_flat=x1
    level_domains = []
    level_values: dict[str, list[np.ndarray]] = {field_name: []}

    for lvl in range(data.num_levels):
        field_arr = data.get_field(field_name, level=lvl)
        mesh = data.level_mesh(lvl, crop_to_owned=True)
        # domain order: slowest to fastest axis (x3, x2, x1)
        coords = [
            getattr(mesh, f"x{ii}v")
            for ii in range(field_arr.ndim, 0, -1)
        ]
        level_domains.append(coords)
        level_values[field_name].append(field_arr)

    stitched = stitch_leaf_cells(level_domains, level_values)

    # auto-select radii if not given
    if radii is None:
        r = np.sqrt(
            stitched["x_flat"] ** 2
            + stitched["y_flat"] ** 2
            + stitched.get("z_flat", np.zeros_like(stitched["x_flat"])) ** 2
        )
        r_pos = r[r > 0]
        if len(r_pos) < 10:
            return np.arange(1, n_theta // 2 + 1, dtype=float), np.zeros(n_theta // 2)
        r_min = np.percentile(r_pos, 5)
        r_max = np.percentile(r_pos, 80)
        radii = np.geomspace(r_min, r_max, n_shells).tolist()

    r_mean = np.exp(np.mean(np.log(radii)))

    ell, c_ell = angular_power_spectrum(
        stitched, field_name, radii,
        n_theta=n_theta, n_phi=n_phi,
        subtract_mean=subtract_mean,
    )
    return ell, c_ell, r_mean


# =============================================================================
# angular velocity power spectrum (non-radial components only)
# =============================================================================


def _interpolate_multi_to_shell(
    stitched_data: dict[str, np.ndarray],
    field_names: Sequence[str],
    radius: float,
    n_theta: int,
    n_phi: int,
    shell_width: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """interpolate multiple fields onto a regular (theta, phi) grid at given radius.

    shares cell selection and grid construction across all fields.

    returns:
        (theta_centers, phi_centers, {field_name: field_on_shell})
        where each field_on_shell has shape (n_theta, n_phi).
    """
    from scipy.interpolate import griddata

    x = stitched_data["x_flat"]
    y = stitched_data["y_flat"]
    z = stitched_data.get("z_flat", np.zeros_like(x))
    r = np.sqrt(x**2 + y**2 + z**2)

    if shell_width <= 0:
        vol = stitched_data["volume_flat"]
        dx_est = np.median(vol ** (1.0 / 3.0))
        shell_width = max(2.0 * dx_est, 0.02 * radius)

    mask = np.abs(r - radius) < shell_width
    if np.sum(mask) < 10:
        return np.zeros(0), np.zeros(0), {}

    x_sel, y_sel, z_sel = x[mask], y[mask], z[mask]
    r_sel = r[mask]

    theta_sel = np.arccos(np.clip(z_sel / (r_sel + 1e-30), -1, 1))
    phi_sel = np.arctan2(y_sel, x_sel) + np.pi

    d_theta = np.pi / n_theta
    d_phi = 2.0 * np.pi / n_phi
    theta_centers = np.linspace(d_theta / 2, np.pi - d_theta / 2, n_theta)
    phi_centers = np.linspace(d_phi / 2, 2.0 * np.pi - d_phi / 2, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers, indexing="ij")

    result = {}
    for name in field_names:
        f_sel = stitched_data[f"{name}_flat"][mask]
        result[name] = griddata(
            (theta_sel, phi_sel), f_sel,
            (theta_grid, phi_grid), method="nearest",
        )

    return theta_centers, phi_centers, result


def _cartesian_to_spherical_velocity(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """convert cartesian velocity components to spherical on an equirectangular grid.

    note: the stitch convention maps x_flat=x3, y_flat=x2, z_flat=x1.
    v1 is the x1-component (fastest axis = z_flat in stitch), etc.
    the spherical decomposition uses the physical (x,y,z) from the
    stitched coordinates, so the mapping is:
      x_phys = z_flat, y_phys = y_flat, z_phys = x_flat
    but for velocity the same permutation applies:
      vx_phys = v3 (z_flat direction), vy_phys = v2, vz_phys = v1

    however, this function takes already-interpolated fields on the
    (theta, phi) grid. the caller is responsible for passing the
    correct mapping. since the radial distance r and angles (theta, phi)
    are computed from the same (x_flat, y_flat, z_flat) coordinates,
    the decomposition is self-consistent regardless of axis labeling.

    args:
        vx, vy, vz: cartesian velocity on (n_theta, n_phi) grid.
            these correspond to the x_flat, y_flat, z_flat directions.
        theta: 1d polar angle centers
        phi: 1d azimuthal angle centers

    returns:
        (v_r, v_theta, v_phi) on the same grid
    """
    # phi was shifted by +pi during interpolation, undo for trig
    phi_phys = phi - np.pi

    sin_t = np.sin(theta)[:, None]
    cos_t = np.cos(theta)[:, None]
    sin_p = np.sin(phi_phys)[None, :]
    cos_p = np.cos(phi_phys)[None, :]

    # standard cartesian -> spherical rotation
    v_r = sin_t * cos_p * vx + sin_t * sin_p * vy + cos_t * vz
    v_theta = cos_t * cos_p * vx + cos_t * sin_p * vy - sin_t * vz
    v_phi = -sin_p * vx + cos_p * vy

    return v_r, v_theta, v_phi


def angular_velocity_power_spectrum(
    stitched_data: dict[str, np.ndarray],
    velocity_fields: Sequence[str],
    radii: Sequence[float],
    n_theta: int = 64,
    n_phi: int = 128,
    subtract_mean: bool = True,
    shell_width: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """angular power spectrum of non-radial velocity components.

    for each shell, interpolates v1, v2, v3 onto (theta, phi),
    converts to spherical (v_r, v_theta, v_phi), then computes
    C_ell = C_ell(v_theta) + C_ell(v_phi).

    v_r is discarded — it contains the Bondi profile and would
    dominate the spectrum. v_theta and v_phi are zero for purely
    radial flow, providing a clean null test.

    args:
        stitched_data: flat arrays from stitch_leaf_cells
        velocity_fields: three field key prefixes (v1, v2, v3)
        radii: shell radii to average over
        n_theta, n_phi: angular resolution
        subtract_mean: remove monopole per component per shell
        shell_width: half-width of each radial shell (0 = auto)

    returns:
        (ell, C_ell): multipole moments and averaged angular power
    """
    c_ell_sum = None
    count = 0

    for radius in radii:
        theta, phi, fields = _interpolate_multi_to_shell(
            stitched_data, velocity_fields,
            radius, n_theta, n_phi, shell_width,
        )
        if len(theta) == 0:
            continue

        vx = fields[velocity_fields[0]]
        vy = fields[velocity_fields[1]]
        vz = fields[velocity_fields[2]]

        _, v_theta, v_phi = _cartesian_to_spherical_velocity(
            vx, vy, vz, theta, phi,
        )

        ell_t, c_t = _angular_spectrum_from_shell(
            v_theta, theta, subtract_mean=subtract_mean,
        )
        ell_p, c_p = _angular_spectrum_from_shell(
            v_phi, theta, subtract_mean=subtract_mean,
        )

        c_ell = c_t + c_p

        if c_ell_sum is None:
            c_ell_sum = c_ell.copy()
        else:
            c_ell_sum += c_ell
        count += 1

    if c_ell_sum is None or count == 0:
        l_max = n_theta // 2
        return np.arange(1, l_max + 1, dtype=float), np.zeros(l_max)

    return ell_t.astype(float), c_ell_sum / count


def composite_angular_velocity_power_spectrum(
    data: "SimData",
    velocity_fields: Sequence[str] = ("v1", "v2", "v3"),
    radii: Optional[Sequence[float]] = None,
    n_shells: int = 5,
    n_theta: int = 64,
    n_phi: int = 128,
    subtract_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """angular velocity power spectrum using stitched leaf cells.

    returns:
        (ell, C_ell, r_mean)
    """
    from simbi.analysis.radial_profiles import stitch_leaf_cells

    level_domains = []
    level_values: dict[str, list[np.ndarray]] = {
        f: [] for f in velocity_fields
    }

    for lvl in range(data.num_levels):
        first_field = data.get_field(velocity_fields[0], level=lvl)
        mesh = data.level_mesh(lvl, crop_to_owned=True)
        coords = [
            getattr(mesh, f"x{ii}v")
            for ii in range(first_field.ndim, 0, -1)
        ]
        level_domains.append(coords)
        for f in velocity_fields:
            level_values[f].append(data.get_field(f, level=lvl))

    stitched = stitch_leaf_cells(level_domains, level_values)

    if radii is None:
        r = np.sqrt(
            stitched["x_flat"] ** 2
            + stitched["y_flat"] ** 2
            + stitched.get("z_flat", np.zeros_like(stitched["x_flat"])) ** 2
        )
        r_pos = r[r > 0]
        if len(r_pos) < 10:
            l_max = n_theta // 2
            return np.arange(1, l_max + 1, dtype=float), np.zeros(l_max), 1.0
        r_min = np.percentile(r_pos, 5)
        r_max = np.percentile(r_pos, 80)
        radii = np.geomspace(r_min, r_max, n_shells).tolist()

    r_mean = np.exp(np.mean(np.log(radii)))

    ell, c_ell = angular_velocity_power_spectrum(
        stitched, velocity_fields, radii,
        n_theta=n_theta, n_phi=n_phi,
        subtract_mean=subtract_mean,
    )
    return ell, c_ell, r_mean
