"""
Pure functions for robust data slicing and transformation.

This implementation assumes a "logical" coordinate system (x1, x2, x3)
and maps it to a "data" array index system (axis 0, 1, 2).
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np

from simbi.types import ProcessedData
from simbi.reader import SimData, parse_data, read_raw_data

from ..config import VisualizationConfig
from ..figure import Figure
from ..types import Array, CoordSystem, FieldData

# Maps logical axis names (user-facing) to the data's array index.
# This example assumes data is stored (nz, ny, nx) or (x3, x2, x1).
AXIS_MAP = {"x1": 2, "x2": 1, "x3": 0}
INV_AXIS_MAP = {v: k for k, v in AXIS_MAP.items()}

# -----------------------------
FIELD_ALIASES = {
    # "Sigma": "rho",
    "b1": "b1_mean",
    "b2": "b2_mean",
    "b3": "b3_mean",
    "mdot": "accretion_rate",
    "maccr": "accreted_mass",
}


@dataclass
class SlicePlan:
    """A complete, explicit plan for slicing and reordering data."""

    # The tuple to index the NumPy array, e.g., (5, slice(None), slice(None))
    index_tuple: tuple

    # The *original* indices of the domain arrays to keep,
    # already in the new logical order (e.g., [2, 1] for [x1_arr, x2_arr])
    final_domain_indices: list[int]

    # The transpose order to apply to the data *after* slicing
    # e.g., (1, 0) to turn (ny, nx) -> (nx, ny)
    transpose_order: tuple[int, ...]

    # The logical names of the final axes, e.g., ["x1", "x2"]
    final_axis_names: list[str]


def find_slice_index(coord_array: Array, position: float) -> int:
    """Find the index closest to the specified position in a coordinate array."""
    return int(np.abs(coord_array - position).argmin())


def plan_slice(
    domain: Sequence[Array] | Array, slice_spec: Optional[dict[str, float]]
) -> SlicePlan:
    """
    Creates a comprehensive slice plan from a simple specification.

    Args:
        domain: The *full* list of coordinate arrays, in the same
                order as the data dimensions (e.g., [x3_arr, x2_arr, x1_arr]).
        slice_spec: A dict of {axis_name: position}, e.g., {"x3": 0.0, "x2": 0.1}.
                    If None or empty, a pass-through plan is created.
    """
    ndim = len(domain)

    if not slice_spec:
        # Pass-through Plan (e.g., Full 3D)
        logical_names = [INV_AXIS_MAP.get(i, f"dim_{i}") for i in range(ndim)]
        return SlicePlan(
            index_tuple=tuple(slice(None) for _ in range(ndim)),
            final_domain_indices=list(range(ndim)),
            transpose_order=tuple(range(ndim)),
            final_axis_names=logical_names,
        )

    # Slicing Plan
    index_tuple_list: list[Any] = [slice(None)] * ndim
    sliced_axes_indices = set()

    # Build the index tuple
    for axis_name, position in slice_spec.items():
        if axis_name not in AXIS_MAP:
            raise KeyError(f"Invalid axis name '{axis_name}' in slice config.")

        axis_index = AXIS_MAP[axis_name]

        if axis_index >= ndim:
            raise IndexError(
                f"Axis {axis_name} (index {axis_index}) out of bounds "
                f"for {ndim}D data."
            )

        coord_array = domain[axis_index]
        slice_index = find_slice_index(coord_array, position)

        index_tuple_list[axis_index] = slice_index
        sliced_axes_indices.add(axis_index)

    # Determine which axes remain and their *logical* order
    remaining_indices = sorted(list(set(range(ndim)) - sliced_axes_indices))

    # Get logical names of remaining axes, e.g., ["x1", "x2"]
    # We sort this to ensure the final plot is always in a
    # predictable order (e.g., x1, then x2, then x3).
    remaining_logical_names = sorted(
        [INV_AXIS_MAP[i] for i in remaining_indices],
        key=lambda name: name,  # Sorts alphabetically: x1, x2, x3
    )

    # Get original domain indices in the new logical order
    # e.g., [2, 1] (index for "x1", index for "x2")
    final_domain_indices = [AXIS_MAP[name] for name in remaining_logical_names]

    # Create the transpose order
    # This maps the sliced data's axes to the new logical order
    # e.g., if remaining_indices is [1, 2] (for ny, nx)
    # and final_domain_indices is [2, 1] (for x1, x2)
    # We need to map axis 1 -> new axis 0 (nx)
    # and axis 0 -> new axis 1 (ny)
    # The transpose_order is [1, 0]
    transpose_order = tuple(range(len(remaining_indices)))
    return SlicePlan(
        index_tuple=tuple(index_tuple_list),
        final_domain_indices=final_domain_indices,
        transpose_order=transpose_order,
        final_axis_names=remaining_logical_names,
    )


def execute_slice(
    values: Array, domain: Sequence[Array] | Array, plan: SlicePlan
) -> tuple[Array, Sequence[Array]]:
    """Executes a pre-computed slice plan."""

    sliced_values = values[plan.index_tuple]

    # Reorder (transpose) the data to match logical order
    # For a 1D slice, sliced_values is (e.g.,) 100-long,
    # transpose_order is (0,), so this does nothing, which is correct. (I think)
    transposed_values = sliced_values.transpose(plan.transpose_order)

    # Reorder the domain to match
    new_domain = [domain[i] for i in plan.final_domain_indices]

    return transposed_values, new_domain


def create_field_data(
    name: str,
    values: Array,
    domain: Sequence[Array],
    slice_plan: SlicePlan,
) -> FieldData:
    """Create a FieldData object from raw arrays."""
    return FieldData(
        name=name,
        values=values,
        domain=list(domain),
        axis_names=slice_plan.final_axis_names,
    )


def prepare_field_level(
    data: SimData,
    field_name: str,
    level: int,
    effective_dim: int,
) -> FieldData:
    """
    Prepares a single FieldData object for a specific level.
    NO slicing is performed.
    """
    values = data.get_field(field_name, level)
    mesh = data.level_mesh(level)

    # Get the full domain, in data-storage order (e.g., nz, ny, nx)
    full_domain = [getattr(mesh, f"x{i}v") for i in range(values.ndim, 0, -1)]

    assert values.ndim == len(full_domain), (
        f"Data dim ({values.ndim}) mismatch with domain dim ({len(full_domain)})"
    )

    name = f"{field_name}_L{level}" if level > 0 else field_name

    # Return the full-dimensional data
    return FieldData(name=name, values=values, domain=list(full_domain))


def prepare_figure(
    config: VisualizationConfig,
    nfiles: int = 1,
    projection: Literal["polar", "cartesian"] | None = None,
    nlvls: int = 1,
    coord_system: CoordSystem = CoordSystem.CARTESIAN,
) -> Figure:
    """Create and prepare a figure based on configuration."""
    import matplotlib.pyplot as plt

    config.theme.apply(nfields=len(config.plot.fields) * nlvls, nfiles=nfiles)
    if projection == "polar":
        fig, ax = plt.subplots(
            1,
            1,
            figsize=config.style.fig_size,
            subplot_kw={"projection": "polar"},
            layout="constrained",
        )
    else:
        fig = plt.figure(figsize=config.style.fig_size)
        ax = fig.add_subplot(111)

    figure = Figure(config)

    figure.fig = fig
    figure.axes["main"] = ax
    figure.coord_system = coord_system

    return figure


def extract_field(field_name: str) -> Callable[[SimData], Array]:
    """Extract a named field from simulation data."""
    return lambda data: data[FIELD_ALIASES.get(field_name, field_name)]


def load_data(file_path: str) -> SimData:
    """
    Load simulation data from a file.

    Args:
        file_path: Path to the simulation data file

    Returns:
        Loaded simulation data
    """
    import h5py

    data: ProcessedData
    with h5py.File(file_path, "r") as file:
        raw_data = read_raw_data(file).unwrap()
        data = parse_data(raw_data).unwrap()

    return SimData(data)


def get_effective_dimensions(data: SimData, config: VisualizationConfig) -> int:
    """Determine the effective number of dimensions to use based on data and config."""
    x = sum(r > 1 for r in data.mesh.shape)
    return min(config.plot.ndim, x)


def _block_average(arr: Array, block_shape: tuple[int, ...]) -> Array:
    """N-dimensional block averaging (downsampling)."""
    if len(arr.shape) != len(block_shape):
        raise ValueError("Array shape and block shape must have same ndim.")

    new_shape = []
    for i, dim in enumerate(arr.shape):
        block_size = block_shape[i]
        if dim % block_size != 0:
            raise ValueError(
                f"Axis {i} (size {dim}) not divisible by block size {block_size}"
            )
        new_shape.extend([dim // block_size, block_size])

    reshaped = arr.reshape(tuple(new_shape))
    avg_axes = tuple(range(1, len(new_shape), 2))
    return np.mean(reshaped, axis=avg_axes)


def _compose_pcolormesh(fields_2d: list[FieldData]) -> FieldData:
    """
    Composes 2D fields for pcolormesh rendering.

    If FMR, this "squashes" fine levels onto the base grid.
    If Unigrid, this is a no-op.
    """
    # Unigrid case: Just return the single 2D field.
    if len(fields_2d) == 1:
        return fields_2d[0]

    # FMR case: Start with the base level (L0)
    base_field = fields_2d[0]
    base_x, base_y = base_field.domain
    composited_values = base_field.values.copy()

    # Loop over finer levels and "squash" them onto the base
    for fine_field in fields_2d[1:]:
        fine_x, fine_y = fine_field.domain
        fine_values = fine_field.values

        # Find overlapping indices in the base grid
        i_start = np.searchsorted(base_x, fine_x[0], side="left")
        i_end = np.searchsorted(base_x, fine_x[-1], side="right")
        j_start = np.searchsorted(base_y, fine_y[0], side="left")
        j_end = np.searchsorted(base_y, fine_y[-1], side="right")

        # Get the sub-region of the base grid that is covered
        coarse_nx = i_end - i_start
        coarse_ny = j_end - j_start

        # Check if the fine grid dimensions are a multiple
        # of the coarse grid region it's covering
        fine_ny, fine_nx = fine_values.shape
        if fine_nx % coarse_nx != 0 or fine_ny % coarse_ny != 0:
            # Grid mismatch, cannot perform clean block averaging
            continue

        # Downsample the fine data to the coarse grid's resolution
        ref_ratio_x = fine_nx // coarse_nx
        ref_ratio_y = fine_ny // coarse_ny

        averaged_fine_data = _block_average(
            fine_values, (ref_ratio_y, ref_ratio_x)
        )

        # Overwrite the base grid data with the averaged fine data
        composited_values[j_start:j_end, i_start:i_end] = averaged_fine_data

    # Return a new 2D FieldData object
    return FieldData(
        name=base_field.name,
        values=composited_values,
        domain=base_field.domain,
        # ndim=2,
    )


def _compose_polygons(fields_2d: Sequence[FieldData]) -> FieldData:
    """
    Composes 2D fields for polygon rendering.

    Converts all cells from all levels into a list of patches and values.
    Returns a 1D FieldData object adhering to the "Polygon Contract".
    """
    all_patches = []
    all_values = []

    # We must track regions covered by finer levels to avoid overplotting
    refined_regions = []

    # Iterate from finest level (end of list) to coarsest (start)
    for field in reversed(fields_2d):
        x_edges, y_edges = field.domain
        values = field.values

        # Create cell patches for this level
        for j in range(len(y_edges) - 1):
            for i in range(len(x_edges) - 1):
                # Check if this cell is covered by an already-processed
                # finer level
                cell_x_center = (x_edges[i] + x_edges[i + 1]) / 2
                cell_y_center = (y_edges[j] + y_edges[j + 1]) / 2

                is_covered = False
                for region in refined_regions:
                    if (
                        region["xmin"] <= cell_x_center <= region["xmax"]
                        and region["ymin"] <= cell_y_center <= region["ymax"]
                    ):
                        is_covered = True
                        break

                if is_covered:
                    continue

                # Not covered, so add this "leaf cell"
                patch = [
                    (x_edges[i], y_edges[j]),
                    (x_edges[i + 1], y_edges[j]),
                    (x_edges[i + 1], y_edges[j + 1]),
                    (x_edges[i], y_edges[j + 1]),
                ]
                all_patches.append(patch)
                all_values.append(values[j, i])

        # Add this level's domain to the list of refined regions
        refined_regions.append(
            {
                "xmin": x_edges[0],
                "xmax": x_edges[-1],
                "ymin": y_edges[0],
                "ymax": y_edges[-1],
            }
        )

    axis_names = fields_2d[0].axis_names
    # Return a new 1D FieldData object (the "Polygon Contract")
    return FieldData(
        name=f"{fields_2d[0].name}_polygons",
        values=np.array(all_values),
        domain=np.array(all_patches),
        axis_names=axis_names,
    )


def compose_2d_render(
    fields_2d: list[FieldData],
    render_mode: Literal["pcolormesh", "polygons"],
) -> FieldData:
    """
    Composes a list of 2D fields into a single renderable FieldData object.

    This is the "stitching" step for 2D FMR or polygon plots.
    """
    if not fields_2d:
        raise ValueError("Cannot compose an empty list of fields.")

    if render_mode == "pcolormesh":
        # Returns a single 2D FieldData object
        return _compose_pcolormesh(fields_2d)
    elif render_mode == "polygons":
        # Returns a single 1D FieldData object (of polygons)
        return _compose_polygons(fields_2d)
    else:
        raise ValueError(f"Unknown render_mode: {render_mode}")
