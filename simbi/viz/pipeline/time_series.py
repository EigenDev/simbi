from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from simbi.reader.adapter import SimData

from ..config import VisualizationConfig
from ..types import Array, FieldData, PlotData
from .transforms import load_data


def _calculate_time_series_value(
    data: SimData, field_name: str, weight_field_name: Optional[str]
) -> np.floating | NDArray[np.floating]:
    """
    Calculates a single scalar value for a given field at a given time.
    """
    # Check for special analysis fields
    if field_name in ["mdot", "maccr"]:
        if data.body_collection is None:
            raise ValueError("No bodies in this run.")

        prop = (
            "accretion_rate" if field_name == "mdot" else "total_accreted_mass"
        )
        return np.array(
            [getattr(v.accretion, prop) for v in data.body_collection.bodies]
        )

    # Standard field calculation
    field = data.get_field(field_name, level=0)  # Default to base level

    if weight_field_name:
        weight_field = data.get_field(weight_field_name, level=0)
        return np.sum(field * weight_field) / np.sum(weight_field)
    else:
        return np.mean(field)


def compute_orbital_averages(
    time: Array, mdot: Array, time_scale: float
) -> tuple[Array, Array]:
    """Compute averages over orbital periods.

    Args:
        time: array of time values
        mdot: array of mdot values
        time_scale: orbital period (e.g. 2π)

    Returns:
        t_bins: array of time bin centers
        mdot_avg: array of averaged mdot values
    """
    n_orbits = (time[-1] - time[0]) / time_scale
    bins = np.linspace(time[0], time[-1], int(n_orbits) + 1)
    t_bins = (bins[1:] + bins[:-1]) / 2  # bin centers
    mdot_avg = np.array(
        [
            np.mean(mdot[(time >= bins[i]) & (time < bins[i + 1])])
            for i in range(len(bins) - 1)
        ]
    )
    return t_bins, mdot_avg


def create_time_series_data(
    files: Sequence[str],
    field_names: Sequence[str],
    config: VisualizationConfig,
) -> PlotData:
    """
    The pipeline for time series.

    Iterates through all files, calculates the requested
    scalar quantities, and returns 1D FieldData objects.
    """

    times: list[float] = []
    field_values_over_time: dict[str, Any] = {name: [] for name in field_names}
    body_names: list[str] = []

    weight_field = config.time_series.weight

    data: SimData = None  # type: ignore
    for i, file_path in enumerate(files):
        data = load_data(file_path)
        times.append(data.metadata.time)

        if i == 0 and data.body_collection:
            nbodies = list(data.body_collection.bodies)
            body_names = [f"M_{i}" for i in range(len(nbodies))]

        for name in field_names:
            value = _calculate_time_series_value(data, name, weight_field)
            field_values_over_time[name].append(value)

    final_fields: list[FieldData] = []
    time_array = np.array(times)

    # if there are two bodies, we label the less massive one M_2
    # and the more massive one M_1
    if len(body_names) == 2 and data.body_collection is not None:
        masses = []
        for body in data.body_collection.bodies:
            masses.append(body.mass)
        if masses[0] > masses[1]:
            body_names = ["M_1", "M_2"]
        else:
            body_names = ["M_2", "M_1"]

    for name in field_names:
        values_array = np.array(field_values_over_time[name])

        field_body_names = body_names if values_array.ndim == 2 else None
        time_units = config.figure.time_units

        if time_units:
            time_units = f"[{time_units}]"

        final_fields.append(
            FieldData(
                name=name,
                values=values_array,
                domain=[time_array],
                spacing_types=["linear"],
                axis_names=[f"$t${time_units}"],
                body_names=field_body_names,
            )
        )

        # Handle special case: orbital averages for mdot
        # This creates a *new* derived field
        if name in ["mdot", "maccr"] and config.figure.time_scale:
            if values_array.ndim == 2:
                ma_times, ma_total_mdot = compute_orbital_averages(
                    time_array,
                    np.sum(values_array, axis=1),
                    time_scale=config.figure.time_scale or 2 * np.pi,
                )
                if ma_times.any():
                    final_fields.append(
                        FieldData(
                            name=r"$\langle \dot{M} \rangle_{\rm tot}$ (orbital)",
                            values=ma_total_mdot,
                            domain=[ma_times],
                            spacing_types=["linear"],
                        )
                    )

    return PlotData(
        fields=final_fields,
        body_collection=None,  # Not relevant for time series
        time=None,  # Not relevant
        dimensions=1,
        coord_system=None,  # Not relevant
        hierarchy=None,
    )
