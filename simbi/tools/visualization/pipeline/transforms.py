"""Pure functions for transforming visualization data."""

from typing import Any, Callable, Optional, Sequence, TypeVar

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ....core.types import ProcessedData
from ....functional import curry
from ....reader import SimData, parse_data, read_raw_data
from ..core.config import VisualizationConfig
from ..core.figure import Figure
from ..core.types import Array, CoordSystem, FieldData, PlotData

T = TypeVar("T")


def extract_field(field_name: str) -> Callable[[SimData], Array]:
    """Extract a named field from simulation data."""
    return lambda data: data[field_name]


def extract_coordinate(coord_name: str) -> Callable[[SimData], Array | None]:
    """Extract a coordinate array from simulation data mesh."""
    return lambda data: getattr(data.mesh, coord_name, None)


def get_domain_for_dimension(data: SimData, ndim: int) -> Sequence[Array]:
    """Get coordinate arrays for the specified number of dimensions."""
    coords = []
    for i in range(1, ndim + 1):
        coord = extract_coordinate(f"x{i}v")(data)
        if coord is not None:
            coords.append(coord)
    return coords


def find_slice_index(coord_array: Array, position: float) -> int:
    """Find the index closest to the specified position in a coordinate array."""
    return int(np.abs(coord_array - position).argmin())


def slice_field(values: Array, axis: int, index: int) -> Array:
    """Slice a field array along the specified axis at the given index."""
    if axis == 0:
        return values[index, ...]
    elif axis == 1:
        return values[:, index, ...]
    elif axis == 2:
        return values[:, :, index, ...]
    return values  # Default case, should not reach here for 1-3D data


def create_slicer(
    axis_name: str, position: float
) -> Callable[[dict[str, Any]], tuple[Array, Sequence[Array]]]:
    """Create a function that slices data along the specified axis."""

    def slicer(data_dict: dict[str, Any]) -> tuple[Array, Sequence[Array]]:
        values = data_dict["values"]
        domain = data_dict["domain"]

        axis_map = {"x1": 2, "x2": 1, "x3": 0}
        axis = axis_map.get(axis_name)

        if axis is None or axis >= len(domain):
            return values, domain

        idx = find_slice_index(domain[axis], position)
        sliced_values = slice_field(values, axis, idx)

        # Create new domain without the sliced dimension
        new_domain = [d for i, d in enumerate(domain) if i != axis]

        return sliced_values, new_domain

    return slicer


def create_field_data(name: str, values: Array, domain: Sequence[Array]) -> FieldData:
    """Create a FieldData object from raw arrays."""
    return FieldData(name=name, values=values, domain=list(domain))


def transform_field(
    data: SimData,
    field_name: str,
    ndim: int,
    slice_config: Optional[dict[str, Any]] = None,
) -> FieldData:
    """Transform a single field based on dimension and slicing configuration."""
    values = extract_field(field_name)(data)
    domain = get_domain_for_dimension(data, ndim)

    if slice_config and "axis" in slice_config and "position" in slice_config:
        slicer = create_slicer(slice_config["axis"], slice_config["position"])
        values, domain = slicer({"values": values, "domain": domain})

    return create_field_data(field_name, values, domain)


def get_effective_dimensions(data: SimData, config: VisualizationConfig) -> int:
    """Determine the effective number of dimensions to use based on data and config."""
    return min(config.plot.ndim, data.metadata.dimensions)


def get_slice_config(config: VisualizationConfig) -> Optional[dict[str, Any]]:
    """Extract slicing configuration from visualization config."""
    if config.multidim.slice_along:
        return {
            "axis": config.multidim.slice_along,
            "position": config.multidim.slice_position,
        }

    if config.plot.ndim == 3:
        axis_map = {1: "x1", 2: "x2", 3: "x3"}
        return {
            "axis": axis_map.get(config.multidim.projection[-1]),
            "position": config.multidim.slice_position,
        }
    return None


def create_plot_data(
    data: SimData, field_names: Sequence[str], config: VisualizationConfig
) -> PlotData:
    """Create plot data from simulation data"""
    ndim = get_effective_dimensions(data, config)
    slice_config = get_slice_config(config)
    field_transform = curry(transform_field, data, ndim=ndim, slice_config=slice_config)
    field_data = [field_transform(name) for name in field_names]
    return PlotData(
        fields=field_data,
        bodies=data.bodies,
        time=data.metadata.time,
        dimensions=ndim,
        coord_system=CoordSystem(data.metadata.coord_system),
    )


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
                    raise ValueError("This run did not include accreting bodies")
                prop = "accretion_rate" if field == "mdot" else "total_accreted_mass"

                field_values[field].append(
                    np.array(
                        [getattr(v.accretion, prop) for _, v in sim_data.bodies.items()]
                    )
                )

    return PlotData(
        fields=[
            FieldData(name=field, values=np.array(vals), domain=[np.array(times)])
            for field, vals in field_values.items()
        ]
    )


def load_data(file_path: str) -> SimData:
    """
    Load simulation data from a file.

    Args:
        file_path: Path to the simulation data file

    Returns:
        Loaded simulation data
    """
    data: ProcessedData
    with h5py.File(file_path, "r") as file:
        raw_data = read_raw_data(file).unwrap()
        data = parse_data(raw_data).unwrap()
    return SimData(data)


def prepare_figure(config: VisualizationConfig, nfiles: int = 1) -> Figure:
    """Create and prepare a figure based on configuration."""
    config.theme.apply(nfields=len(config.plot.fields), nfiles=nfiles)
    fig = plt.figure(figsize=config.style.fig_size)
    ax = fig.add_subplot(111)

    figure = Figure(config)

    figure.fig = fig
    figure.axes["main"] = ax

    return figure
