"""Pure functions for transforming visualization data."""

from typing import Callable, Sequence, Optional, Any, TypeVar
from ..core.types import Array, FieldData, PlotData, CoordSystem
from ..core.config import VisualizationConfig
from ....reader import SimData
from ....functional import curry
import numpy as np

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

        # Map axis name to index
        axis_map = {"x1": 0, "x2": 1, "x3": 2}
        axis = axis_map.get(axis_name)

        if axis is None or axis >= len(domain):
            return values, domain

        # Find the slice index
        idx = find_slice_index(domain[axis], position)

        # Slice the values
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
    # Extract raw field values
    values = extract_field(field_name)(data)

    # Get domain for the specified dimension
    domain = get_domain_for_dimension(data, ndim)

    # Apply slicing if configured
    if slice_config and "axis" in slice_config and "position" in slice_config:
        slicer = create_slicer(slice_config["axis"], slice_config["position"])
        values, domain = slicer({"values": values, "domain": domain})

    # Create and return the field data
    return create_field_data(field_name, values, domain)


# ---- Dimension handling ----


def get_effective_dimensions(data: SimData, config: VisualizationConfig) -> int:
    """Determine the effective number of dimensions to use based on data and config."""
    return min(config.plot.ndim, data.metadata.dimensions)


def get_slice_config(config: VisualizationConfig) -> Optional[dict[str, Any]]:
    """Extract slicing configuration from visualization config."""
    if config.multidim.slice_along:
        return {
            "axis": config.multidim.slice_along,
            "position": 0.0,  # Default position, would come from config in real impl
        }
    return None


def create_plot_data(
    data: SimData, field_names: Sequence[str], config: VisualizationConfig
) -> PlotData:
    """Create plot data from simulation data using a functional pipeline."""
    # Determine effective dimensions
    ndim = get_effective_dimensions(data, config)

    # Get slicing configuration
    slice_config = get_slice_config(config)

    # Transform each field
    field_transform = curry(transform_field, data, ndim=ndim, slice_config=slice_config)
    field_data = [field_transform(name) for name in field_names]

    # Create and return plot data
    return PlotData(
        fields=field_data,
        time=data.metadata.time,
        dimensions=ndim,
        coord_system=CoordSystem(data.metadata.coord_system),
    )
