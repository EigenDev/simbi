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


# =============================================================================
# composite AMR spectra (prolongate to finest grid)
# =============================================================================


def _prolongate_field(
    data: SimData, field_name: str
) -> tuple[np.ndarray, float]:
    """build a uniform grid at the finest resolution and compute its dx.

    upsamples level-0 data via nearest-neighbor repeat, then
    overwrites refined regions with actual fine-level data.
    returns (composite_array, dx_finest).
    """
    hierarchy = data.hierarchy()
    num_levels = data.num_levels

    # total refinement ratio from level 0 to finest
    total_ratio = 1
    for rr in hierarchy.ref_ratios:
        total_ratio *= rr

    # level 0 data and grid info
    base = data.get_field(field_name, level=0)
    base_mesh = data.checkpoint.levels[0].mesh
    ndim = base.ndim

    # dx at finest level (use last axis — fastest varying, x1)
    dx_coarse = (
        base_mesh.dims[-1][1] - base_mesh.dims[-1][0]
    ) / base_mesh.global_cells[-1]
    dx_finest = dx_coarse / total_ratio

    # upsample level 0 to finest resolution (nearest-neighbor)
    composite = base
    for ax in range(ndim):
        composite = np.repeat(composite, total_ratio, axis=ax)

    # overlay each refined level's actual data
    cumulative_ratio = 1
    for lvl in range(1, num_levels):
        ref_ratio = hierarchy.ref_ratios[lvl - 1]
        cumulative_ratio *= ref_ratio

        # ratio from this level to the finest
        level_to_finest = total_ratio // cumulative_ratio

        level_data = data.checkpoint.levels[lvl]
        owned = level_data.partitions[0].owned_domain
        field = data.get_field(field_name, level=lvl)

        # owned_domain indices are in this level's global_cells coords;
        # scale to finest-grid coordinates
        fine_slices = tuple(
            slice(
                owned.start[ax] * level_to_finest,
                owned.fin[ax] * level_to_finest,
            )
            for ax in range(ndim)
        )

        if level_to_finest > 1:
            upsampled = field
            for ax in range(ndim):
                upsampled = np.repeat(upsampled, level_to_finest, axis=ax)
        else:
            upsampled = field

        composite[fine_slices] = upsampled

    return composite, dx_finest


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


def composite_shell_averaged_spectrum(
    data: SimData,
    fields: Sequence[str] = ("v1", "v2", "v3"),
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """kinetic energy spectrum from prolongated composite grid.

    builds a uniform grid at the finest AMR resolution, then
    computes the shell-averaged spectrum via a single FFT.

    returns:
        (k_centers, E_k, k_nyquist_per_level)
    """
    vx, dx = _prolongate_field(data, fields[0])
    vy, _ = _prolongate_field(data, fields[1])
    vz, _ = _prolongate_field(data, fields[2])

    k, ek = shell_averaged_spectrum(vx, vy, vz, dx)
    return k, ek, _k_nyquist_per_level(data)


def composite_shell_averaged_scalar_spectrum(
    data: SimData,
    field: str,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """scalar power spectrum from prolongated composite grid.

    returns:
        (k_centers, P_k, k_nyquist_per_level)
    """
    composite, dx = _prolongate_field(data, field)
    k, pk = shell_averaged_scalar_spectrum(composite, dx)
    return k, pk, _k_nyquist_per_level(data)
