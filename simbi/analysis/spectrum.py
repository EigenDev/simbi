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
from typing import Optional, Sequence

import numpy as np
from scipy.signal import lombscargle
from scipy.stats import binned_statistic


def shell_averaged_spectrum(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    compute shell-averaged kinetic energy spectrum from 3D velocity fields.

    args:
        vx, vy, vz: 3D velocity component arrays (same shape)
        dx: uniform cell spacing

    returns:
        (k_centers, E_k): wavenumber bin centers and spectrum values
    """
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
