"""Pure functions for transforming visualization data."""

from dataclasses import dataclass
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


# Centralized axis mapping
AXIS_MAP = {"x1": 2, "x2": 1, "x3": 0}


@dataclass
class SlicePlan:
    """Plan for how to slice the data."""

    slice_indices: list[int]  # Indices to slice at
    slice_axes: list[int]  # Which axes to slice
    remaining_axes: list[int]  # Which axes remain after slicing


def find_slice_index(coord_array: Array, position: float) -> int:
    """Find the index closest to the specified position in a coordinate array."""
    return int(np.abs(coord_array - position).argmin())


def plan_slice(
    axis_names: set[str], positions: Sequence[float], domain: Sequence[Array]
) -> SlicePlan:
    """Plan how to slice the data - pure function that doesn't modify anything."""
    # Convert axis names to indices
    slice_axes = [AXIS_MAP[name] for name in axis_names]

    # Find slice indices for each axis
    slice_indices = []
    for axis, position in zip(slice_axes, positions):
        if axis < len(domain):
            idx = find_slice_index(domain[axis], position)
            slice_indices.append(idx)
        else:
            slice_indices.append(0)  # fallback

    # Determine remaining axes
    all_axes = set(range(len(domain)))
    remaining_axes = sorted(all_axes - set(slice_axes))

    return SlicePlan(
        slice_indices=slice_indices,
        slice_axes=slice_axes,
        remaining_axes=remaining_axes,
    )


def execute_slice(
    values: Array, domain: Sequence[Array], plan: SlicePlan
) -> tuple[Array, Sequence[Array]]:
    """Execute the slice plan on the data."""
    # Start with original values
    sliced_values = values

    # Apply slices in reverse order to maintain index validity
    for axis, idx in zip(
        reversed(plan.slice_axes), reversed(plan.slice_indices)
    ):
        sliced_values = np.take(sliced_values, idx, axis=axis)

    # Create new domain with only remaining dimensions
    new_domain = [domain[i] for i in plan.remaining_axes]

    return sliced_values, new_domain


def create_slicer(
    axis_names: set[str], positions: Sequence[float]
) -> Callable[[dict[str, Any]], tuple[Array, Sequence[Array]]]:
    """Create a function that slices data along specified axes at given positions."""

    def slicer(data_dict: dict[str, Any]) -> tuple[Array, Sequence[Array]]:
        values = data_dict["values"]
        domain = data_dict["domain"]
        plan = plan_slice(axis_names, positions, domain)
        return execute_slice(values, domain, plan)

    return slicer


# Convenience functions for common slice patterns
def slice_to_2d(axis_name: str, position: float) -> Callable:
    """Create a slicer that reduces 3D data to 2D by slicing along one axis."""
    return create_slicer({axis_name}, [position])


def slice_to_1d(axis_names: set[str], positions: Sequence[float]) -> Callable:
    """Create a slicer that reduces ND data to 1D by slicing along multiple axes."""
    return create_slicer(axis_names, positions)


def extract_field(field_name: str) -> Callable[[SimData], Array]:
    """Extract a named field from simulation data."""
    return lambda data: data[field_name]


def extract_coordinate(coord_name: str) -> Callable[[SimData], Array | None]:
    """Extract a coordinate array from simulation data mesh."""
    return lambda data: getattr(data.mesh, coord_name, None)


def get_domain_for_dimension(
    data: SimData, ndim: int, vertices: bool = True
) -> Sequence[Array]:
    """Get coordinate arrays for the specified number of dimensions."""
    coords = []
    suffix = "v" if vertices else "c"
    for i in range(1, ndim + 1):
        coord = extract_coordinate(f"x{i}{suffix}")(data)
        if coord is not None:
            coords.append(coord)
    return coords


def slice_field(values: Array, axis: int, index: int) -> Array:
    """Slice a field array along the specified axis at the given index."""
    if axis == 0:
        return values[index, ...]
    elif axis == 1:
        return values[:, index, ...]
    elif axis == 2:
        return values[:, :, index, ...]
    return values  # Default case, should not reach here for 1-3D data


def create_field_data(
    name: str, values: Array, domain: Sequence[Array]
) -> FieldData:
    """Create a FieldData object from raw arrays."""
    return FieldData(name=name, values=values, domain=list(domain))


def create_slicer_from_config(slice_config: dict[str, Any]) -> Callable:
    """Create a slicer function from configuration dictionary."""
    # Handle the old "axis" + "position" pattern (2D slice from 3D)
    if "axis" in slice_config and "position" in slice_config:
        return slice_to_2d(slice_config["axis"], slice_config["position"])

    # Handle the "orthogonal_ax" + "orthogonal_pos" pattern (1D slice from ND)
    elif "orthogonal_ax" in slice_config and "orthogonal_pos" in slice_config:
        axis_names = set(slice_config["orthogonal_ax"])
        positions = slice_config["orthogonal_pos"]
        return slice_to_1d(axis_names, positions)

    # No slicing needed
    else:
        return lambda data_dict: (data_dict["values"], data_dict["domain"])


def transform_field(
    data: SimData,
    field_name: str,
    ndim: int,
    slice_config: Optional[dict[str, Any]] = None,
) -> FieldData:
    """Transform a single field based on dimension and slicing configuration."""
    if ndim in (2, 3):
        vertices = True
    elif slice_config and "orthogonal_ax" in slice_config:
        vertices = False
    else:
        vertices = False
    values = extract_field(field_name)(data)
    domain = get_domain_for_dimension(data, ndim, vertices=vertices)
    if slice_config:
        slicer = create_slicer_from_config(slice_config)
        values, domain = slicer({"values": values, "domain": domain})

    return create_field_data(field_name, values, domain)


def get_effective_dimensions(data: SimData, config: VisualizationConfig) -> int:
    """Determine the effective number of dimensions to use based on data and config."""
    return min(config.plot.ndim, data.metadata.dimensions)


def get_slice_config(config: VisualizationConfig) -> Optional[dict[str, Any]]:
    """Extract slicing configuration from visualization config with robust error handling."""
    if not hasattr(config, "multidim"):
        return None

    # Case 1: Explicit slice_along specified (orthogonal slice)
    if hasattr(config.multidim, "slice_along") and config.multidim.slice_along:
        return _create_orthogonal_slice_config(config)

    # Case 2: 3D data needs projection (axis slice)
    if config.plot.ndim == 3 and hasattr(config.multidim, "projection"):
        return _create_projection_slice_config(config)

    return None


def _create_orthogonal_slice_config(
    config: VisualizationConfig,
) -> Optional[dict[str, Any]]:
    """Create config for orthogonal slicing (keeping one axis, slicing others)."""
    slice_along = config.multidim.slice_along
    ndim = config.plot.ndim

    # Validate slice_along is valid
    valid_axes = {"x1", "x2", "x3"}
    if slice_along not in valid_axes:
        return None

    # Get coordinates for slicing
    coords = getattr(config.multidim, "coords", {})
    if not coords:
        return None

    # Determine which axes to slice (orthogonal to slice_along)
    if ndim == 2:
        all_axes_2d = {"x1", "x2"}
        orthogonal_axes = all_axes_2d - {slice_along}
    else:  # ndim == 3
        all_axes_3d = {"x1", "x2", "x3"}
        orthogonal_axes = all_axes_3d - {slice_along}

    # Ensure we have positions for all orthogonal axes
    orthogonal_positions = []
    coord_vals = list(coords.values())
    for i, axis in enumerate(sorted(orthogonal_axes)):  # Sort for consistency)
        orthogonal_positions.append(coord_vals[i])

    return {
        "orthogonal_ax": sorted(orthogonal_axes),
        "orthogonal_pos": orthogonal_positions,
    }


def _create_projection_slice_config(
    config: VisualizationConfig,
) -> Optional[dict[str, Any]]:
    """Create config for projection slicing (removing one axis)."""
    projection = getattr(config.multidim, "projection", [])
    slice_position = getattr(config.multidim, "slice_position", None)

    if not projection or slice_position is None:
        return None

    # Map dimension number to axis name
    dim_to_axis = {1: "x1", 2: "x2", 3: "x3"}

    # Get the last projection dimension (the one to slice away)
    last_dim = projection[-1]
    if last_dim not in dim_to_axis:
        return None

    return {"axis": dim_to_axis[last_dim], "position": slice_position}


def create_plot_data(
    data: SimData, field_names: Sequence[str], config: VisualizationConfig
) -> PlotData:
    """Create plot data from simulation data"""
    ndim = get_effective_dimensions(data, config)
    slice_config = get_slice_config(config)
    field_transform = curry(
        transform_field, data, ndim=ndim, slice_config=slice_config
    )
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
