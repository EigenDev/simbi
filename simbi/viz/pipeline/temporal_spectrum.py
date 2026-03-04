# =============================================================================
# temporal_spectrum.py
#
# thin wrapper around simbi.analysis.lomb_scargle_psd.
# iterates checkpoints, extracts scalars, calls the pure function,
# packages into PlotData.
#
# features:
# - auto-detects orbital_period from binary_params when --time-scale omitted
# - multi-body systems emit per-body + total power
# - supports welch-style segment averaging via config
# - passes n_samples/n_freqs metadata for FAP computation
# =============================================================================
from typing import Optional, Sequence

import numpy as np

from simbi.analysis import lomb_scargle_psd, welch_lomb_scargle_psd

from ..config import VisualizationConfig
from ..types import FieldData, PlotData
from .time_series import _calculate_time_series_value
from .transforms import load_data

# display names for known fields in PSD context
_PSD_YLABEL: dict[str, str] = {
    "mdot": r"$|\hat{\dot{M}}(\omega)|^2$",
    "maccr": r"$|\hat{M}_{\rm acc}(\omega)|^2$",
}


def _detect_binary_params(data) -> Optional[dict]:
    """extract binary_params from checkpoint body_collection if present."""
    bc = data.body_collection
    if bc is None:
        return None
    return getattr(bc, "binary_params", None)


def _pre_filter(
    times: np.ndarray,
    values: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """gaussian low-pass filter for nearly-uniform time series."""
    from scipy.ndimage import gaussian_filter1d

    dt_median = float(np.median(np.diff(times)))
    sigma_samples = sigma / dt_median
    return gaussian_filter1d(values, sigma=sigma_samples)


def _compute_psd(
    time_array: np.ndarray,
    values: np.ndarray,
    orbital_period: Optional[float],
    config: VisualizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """dispatch to standard or welch lomb-scargle based on config."""
    ts_config = config.temporal_spectrum
    normalize = ts_config.normalize_psd

    if ts_config.psd_method == "welch":
        return welch_lomb_scargle_psd(
            time_array,
            values,
            orbital_period,
            n_segments=ts_config.n_segments,
            overlap=ts_config.overlap,
            normalize=normalize,
        )
    return lomb_scargle_psd(
        time_array, values, orbital_period, normalize=normalize
    )


def create_temporal_spectrum_data(
    files: Sequence[str],
    field_names: Sequence[str],
    config: VisualizationConfig,
) -> PlotData:
    """
    compute temporal power spectrum from a sequence of checkpoints.

    iterates all files, extracts the requested scalar quantities
    (same logic as time_series), then computes lomb-scargle PSD for each.
    """
    times: list[float] = []
    values_over_time: dict[str, list] = {name: [] for name in field_names}
    body_names: list[str] = []
    binary_params: Optional[dict] = None

    weight_field = config.time_series.weight

    for ii, file_path in enumerate(files):
        data = load_data(file_path)
        times.append(data.metadata.time)

        if ii == 0:
            if data.body_collection:
                nbodies = list(data.body_collection.bodies)
                body_names = [
                    rf"$M_{{{jj + 1}}}$" for jj in range(len(nbodies))
                ]
            binary_params = _detect_binary_params(data)

        for name in field_names:
            value = _calculate_time_series_value(data, name, weight_field)
            values_over_time[name].append(value)

    time_array = np.array(times)
    n_samples = len(time_array)

    # auto-detect orbital period from binary params when not set
    orbital_period = config.figure.time_scale
    if orbital_period is None and binary_params is not None:
        orbital_period = binary_params.get("orbital_period")

    # resolve pre-filter width to absolute time units
    filter_width = config.temporal_spectrum.pre_filter_width
    if filter_width is not None and orbital_period is not None:
        sigma_time = filter_width * orbital_period
    elif filter_width is not None:
        sigma_time = filter_width
    else:
        sigma_time = None

    # build x-axis label
    if orbital_period is not None and orbital_period > 0:
        xlabel = r"$\omega / \Omega$"
    else:
        xlabel = r"$\omega$"

    result_fields: list[FieldData] = []

    for name in field_names:
        vals = np.array(values_over_time[name])
        ylabel = _PSD_YLABEL.get(name, rf"$|\hat{{{name}}}(\omega)|^2$")

        if vals.ndim == 2:
            # per-body curves
            for jj in range(vals.shape[1]):
                col = vals[:, jj]
                if sigma_time is not None:
                    col = _pre_filter(time_array, col, sigma_time)
                omega, psd = _compute_psd(
                    time_array, col, orbital_period, config
                )
                result_fields.append(
                    FieldData(
                        name=ylabel,
                        values=psd,
                        domain=[omega],
                        axis_names=[xlabel],
                    )
                )

            # total binary power
            total_vals = vals.sum(axis=1)
            if sigma_time is not None:
                total_vals = _pre_filter(time_array, total_vals, sigma_time)
            omega, psd = _compute_psd(
                time_array, total_vals, orbital_period, config
            )
            result_fields.append(
                FieldData(
                    name=ylabel,
                    values=psd,
                    domain=[omega],
                    axis_names=[xlabel],
                    body_names=[r"$\dot{M}_{\rm tot}$"],
                )
            )
        else:
            if sigma_time is not None:
                vals = _pre_filter(time_array, vals, sigma_time)
            omega, psd = _compute_psd(time_array, vals, orbital_period, config)
            result_fields.append(
                FieldData(
                    name=ylabel,
                    values=psd,
                    domain=[omega],
                    axis_names=[xlabel],
                )
            )

    # compute nyquist frequency (in normalized units if applicable)
    dt_min = float(np.min(np.diff(time_array)))
    omega_nyquist = np.pi / dt_min
    if orbital_period is not None and orbital_period > 0:
        omega_orb = 2.0 * np.pi / orbital_period
        omega_nyquist = omega_nyquist / omega_orb

    # attach metadata for FAP computation and harmonic annotation
    n_freqs = len(result_fields[0].domain[0]) if result_fields else 1024

    return PlotData(
        fields=result_fields,
        time=None,
        dimensions=1,
        extra={
            "n_samples": n_samples,
            "n_freqs": n_freqs,
            "binary_params": binary_params,
            "orbital_period": orbital_period,
            "omega_nyquist": omega_nyquist,
        },
    )
