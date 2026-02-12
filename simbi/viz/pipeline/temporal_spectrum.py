# =============================================================================
# temporal_spectrum.py
#
# thin wrapper around simbi.analysis.lomb_scargle_psd.
# iterates checkpoints, extracts scalars, calls the pure function,
# packages into PlotData.
# =============================================================================
from typing import Sequence

import numpy as np

from simbi.analysis import lomb_scargle_psd

from ..config import VisualizationConfig
from ..types import FieldData, PlotData
from .time_series import _calculate_time_series_value
from .transforms import load_data

# display names for known fields in PSD context
_PSD_YLABEL: dict[str, str] = {
    "mdot": r"$|\hat{\dot{M}}(\omega)|^2$",
    "maccr": r"$|\hat{M}_{\rm acc}(\omega)|^2$",
}


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

    weight_field = config.time_series.weight

    for ii, file_path in enumerate(files):
        data = load_data(file_path)
        times.append(data.metadata.time)

        if ii == 0 and data.body_collection:
            nbodies = list(data.body_collection.bodies)
            body_names = [f"M_{jj}" for jj in range(len(nbodies))]

        for name in field_names:
            value = _calculate_time_series_value(data, name, weight_field)
            values_over_time[name].append(value)

    time_array = np.array(times)
    orbital_period = config.figure.time_scale

    # build x-axis label
    if orbital_period is not None and orbital_period > 0:
        xlabel = r"$\omega / \Omega_{\rm orb}$"
    else:
        xlabel = r"$\omega$"

    result_fields: list[FieldData] = []

    for name in field_names:
        vals = np.array(values_over_time[name])
        ylabel = _PSD_YLABEL.get(name, rf"$|\hat{{{name}}}(\omega)|^2$")

        if vals.ndim == 2:
            for jj in range(vals.shape[1]):
                omega, psd = lomb_scargle_psd(
                    time_array, vals[:, jj], orbital_period
                )
                label = body_names[jj] if jj < len(body_names) else f"body_{jj}"
                result_fields.append(
                    FieldData(
                        name=ylabel,
                        values=psd,
                        domain=[omega],
                        axis_names=[xlabel],
                        body_names=[label],
                    )
                )
        else:
            omega, psd = lomb_scargle_psd(time_array, vals, orbital_period)
            result_fields.append(
                FieldData(
                    name=ylabel,
                    values=psd,
                    domain=[omega],
                    axis_names=[xlabel],
                )
            )

    return PlotData(
        fields=result_fields,
        time=None,
        dimensions=1,
    )
