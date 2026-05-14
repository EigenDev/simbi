# =============================================================================
# phase_fold.py
#
# fold a time series on a known period and bin by orbital phase.
# reuses the same checkpoint extraction as time_series.py.
#
# usage:
#   from simbi.viz.pipeline.phase_fold import create_phase_fold_data
#   plot_data = create_phase_fold_data(files, ["mdot"], config)
# =============================================================================
from typing import Optional, Sequence

import numpy as np

from simbi.analysis import phase_fold

from ..config import VisualizationConfig
from ..types import FieldData, PlotData
from .time_series import _calculate_time_series_value
from .transforms import load_data


def create_phase_fold_data(
    files: Sequence[str],
    field_names: Sequence[str],
    config: VisualizationConfig,
) -> PlotData:
    """
    fold time series data on a known period and return binned phase profiles.

    period is taken from config.figure.time_scale (the orbital period).
    n_bins and show_orbits are taken from config.phase_fold.
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
            bc = data.body_collection
            if bc is not None:
                binary_params = getattr(bc, "binary_params", None)

        for name in field_names:
            value = _calculate_time_series_value(data, name, weight_field)
            values_over_time[name].append(value)

    time_array = np.array(times)

    # resolve period: config.figure.time_scale or auto-detect from binary
    period = config.figure.time_scale
    if period is None and binary_params is not None:
        period = binary_params.get("orbital_period")
    if period is None or period <= 0:
        raise ValueError(
            "phase folding requires a period. "
            "set --time-scale to the orbital period."
        )

    pf_config = config.phase_fold
    n_bins = pf_config.n_bins
    show_orbits = pf_config.show_orbits

    result_fields: list[FieldData] = []

    for name in field_names:
        vals = np.array(values_over_time[name])

        if vals.ndim == 2:
            # per-body: fold the total
            total_vals = vals.sum(axis=1)
            centers, mean, std, raw_phase = phase_fold(
                time_array, total_vals, period, n_bins
            )
            bands = (mean - std, mean + std)

            result_fields.append(
                FieldData(
                    name=name,
                    values=mean,
                    domain=[centers],
                    axis_names=[r"$\phi / 2\pi$"],
                    bands=bands,
                    body_names=[rf"${name}_{{\rm tot}}$"],
                )
            )

            # optionally emit individual orbit traces as a second field
            if show_orbits:
                n_orbits = int(np.floor((time_array[-1] - time_array[0]) / period))
                if n_orbits > 1:
                    orbit_traces = _build_orbit_traces(
                        time_array, total_vals, period, n_bins
                    )
                    if orbit_traces is not None:
                        result_fields.append(orbit_traces)

        else:
            centers, mean, std, raw_phase = phase_fold(
                time_array, vals, period, n_bins
            )
            bands = (mean - std, mean + std)

            result_fields.append(
                FieldData(
                    name=name,
                    values=mean,
                    domain=[centers],
                    axis_names=[r"$\phi / 2\pi$"],
                    bands=bands,
                )
            )

            if show_orbits:
                n_orbits = int(np.floor((time_array[-1] - time_array[0]) / period))
                if n_orbits > 1:
                    orbit_traces = _build_orbit_traces(
                        time_array, vals, period, n_bins
                    )
                    if orbit_traces is not None:
                        result_fields.append(orbit_traces)

    return PlotData(
        fields=result_fields,
        time=None,
        dimensions=1,
        extra={
            "period": period,
            "n_samples": len(time_array),
            "binary_params": binary_params,
        },
    )


def _build_orbit_traces(
    times: np.ndarray,
    values: np.ndarray,
    period: float,
    n_bins: int,
) -> Optional[FieldData]:
    """build per-orbit phase-binned traces stacked as a 2D array."""
    t0 = times[0]
    orbit_idx = ((times - t0) / period).astype(int)
    n_orbits = orbit_idx.max() + 1
    if n_orbits < 2:
        return None

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    phase = (times % period) / period
    digit = np.clip(np.digitize(phase, bin_edges) - 1, 0, n_bins - 1)

    traces = np.full((n_bins, n_orbits), np.nan)
    for oo in range(n_orbits):
        orbit_mask = orbit_idx == oo
        for bb in range(n_bins):
            combined = orbit_mask & (digit == bb)
            if combined.sum() > 0:
                traces[bb, oo] = values[combined].mean()

    return FieldData(
        name="_orbit_traces",
        values=traces,
        domain=[centers],
        axis_names=[r"$\phi / 2\pi$"],
    )
