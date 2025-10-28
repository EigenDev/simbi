from typing import Sequence

import numpy as np

from ..core.types import FieldData, PlotData
from .transforms import load_data


def create_time_series_data(
    files: Sequence[str],
    field_names: Sequence[str] = ["rho"],
) -> PlotData:
    """
    Create a time series of plot data for animation testing.

    Args:
        num_frames: Number of frames to create
        field_names: List of field names to create
        domain_size: Size of the spatial domain

    Returns:
        List of PlotData objects representing a time series
    """
    times: list[float] = []
    field_values = {field: [] for field in field_names}
    for file_path in files:
        sim_data = load_data(file_path)
        time = sim_data.metadata.time
        times.append(time)
        for field in field_names:
            if field in sim_data.fields:
                field_values[field].append(np.mean(sim_data[field]))
            elif field in ["mdot", "maccr"]:
                if not sim_data.bodies:
                    raise ValueError("No bodies in this run.")

                if not any(v.accretion for _, v in sim_data.bodies.items()):
                    raise ValueError(
                        "This run did not include accreting bodies"
                    )
                prop = (
                    "accretion_rate"
                    if field == "mdot"
                    else "total_accreted_mass"
                )

                field_values[field].append(
                    np.array(
                        [
                            getattr(v.accretion, prop)
                            for _, v in sim_data.bodies.items()
                        ]
                    )
                )

    return PlotData(
        fields=[
            FieldData(
                name=field, values=np.array(vals), domain=[np.array(times)]
            )
            for field, vals in field_values.items()
        ]
    )
