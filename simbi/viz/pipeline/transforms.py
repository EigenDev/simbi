"""
Pure functions for robust data slicing and transformation.

This implementation assumes a "logical" coordinate system (x1, x2, x3)
and maps it to a "data" array index system (axis 0, 1, 2).
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np

from simbi.reader.adapter import SimData

from ..config import VisualizationConfig
from ..figure import Figure
from ..types import Array, CoordSystem, FieldData

# maps logical axis names (user-facing) to the data's array index.
# this example assumes data is stored (nz, ny, nx) or (x3, x2, x1).
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
    """Find the index closest to the specified position in a coordinate array.

    the coordinate array may be the VERTEX array (n+1 entries for n cells): a
    position at or past the upper edge argmins to index n, which overflows the
    n-cell values array. clamp to the last CELL index, and refuse a position
    outside the domain with a message naming the bounds."""
    lo, hi = float(coord_array[0]), float(coord_array[-1])
    if not (min(lo, hi) <= position <= max(lo, hi)):
        raise ValueError(
            f"slice position {position} is outside the domain [{lo:g}, {hi:g}] "
            "— check the --slice value"
        )
    idx = int(np.abs(coord_array - position).argmin())
    # a vertex array has one more entry than the cell-value axis it indexes.
    return min(idx, len(coord_array) - 2) if len(coord_array) >= 2 else 0


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
        # pass-through plan (e.g., full 3D)
        logical_names = [INV_AXIS_MAP.get(i, f"dim_{i}") for i in range(ndim)]
        return SlicePlan(
            index_tuple=tuple(slice(None) for _ in range(ndim)),
            final_domain_indices=list(range(ndim)),
            transpose_order=tuple(range(ndim)),
            final_axis_names=logical_names,
        )

    # slicing plan
    index_tuple_list: list[Any] = [slice(None)] * ndim
    sliced_axes_indices = set()

    # build the index tuple
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

    # the surviving axes in DATA-AXIS order (ascending index). `sliced_values` keeps its
    # axes in exactly this order, so the DOMAIN must stay in the same order for
    # domain[k] to be the coordinate array of values axis k -- the plotter reads
    # domain[0] as the outer/slower axis (vertical) and domain[1] as the inner/faster
    # (horizontal), matching the unsliced/native path. reordering the domain into logical
    # (x1,x2,x3) order WITHOUT a matching value transpose is what transposed a non-square
    # slice: the domain claimed (x1,x2) while the values stayed (x2,x1), so the coordinate
    # lengths no longer matched the value grid.
    remaining_indices = sorted(set(range(ndim)) - sliced_axes_indices)
    final_domain_indices = remaining_indices
    transpose_order = tuple(range(len(remaining_indices)))
    # AXIS LABELS, by contrast, are the logical names in forward (x1,x2,x3) order: the
    # labeler pairs axis_names[0] with the horizontal axis, which is domain[1] = the
    # inner axis = the lowest-numbered surviving logical axis. this mirrors the native
    # convention (domain reversed, names forward), so a slice labels like an unsliced plot.
    final_axis_names = sorted(INV_AXIS_MAP[i] for i in remaining_indices)
    return SlicePlan(
        index_tuple=tuple(index_tuple_list),
        final_domain_indices=final_domain_indices,
        transpose_order=transpose_order,
        final_axis_names=final_axis_names,
    )


def execute_slice(
    values: Array, domain: Sequence[Array] | Array, plan: SlicePlan
) -> tuple[Array, Sequence[Array]]:
    """Executes a pre-computed slice plan."""

    sliced_values = values[plan.index_tuple]

    # reorder (transpose) the data to match logical order
    # for a 1D slice, sliced_values is (e.g.,) 100-long,
    # transpose_order is (0,), so this does nothing, which is correct.
    transposed_values = sliced_values.transpose(plan.transpose_order)

    # reorder the domain to match
    new_domain = [domain[i] for i in plan.final_domain_indices]

    return transposed_values, new_domain


def prepare_field_level(
    data: SimData,
    field_name: str,
    level: int,
    effective_dim: int,
    crop_to_owned: bool = False,
) -> FieldData:
    """
    Prepares a single FieldData object for a specific level.
    Automatically squeezes singleton dimensions for quasi-1D/2D data.

    If crop_to_owned=True and level > 0, returns only the refined region
    with appropriate coordinate bounds.
    """
    values = data.get_field(field_name, level, crop_to_owned=crop_to_owned)
    mesh = data.level_mesh(level, crop_to_owned=crop_to_owned)

    # detect if field is face-centered by comparing shape to mesh
    # mesh shape is (nz, ny, nx) in storage order
    mesh_shape = mesh.shape
    is_face_centered = any(
        values.shape[i] == mesh_shape[i] + 1 for i in range(values.ndim)
    )

    if is_face_centered:
        # average face-centered field to cell centers
        # determine which axis is staggered
        for axis in range(values.ndim):
            if values.shape[axis] == mesh_shape[axis] + 1:
                # average along this axis
                slices_left = [slice(None)] * values.ndim
                slices_right = [slice(None)] * values.ndim
                slices_left[axis] = slice(None, -1)
                slices_right[axis] = slice(1, None)
                values = 0.5 * (
                    values[tuple(slices_left)] + values[tuple(slices_right)]
                )
                break

    # Get the domain, in data-storage order (e.g., nz, ny, nx)
    # if crop_to_owned=True, mesh already has cropped coordinates
    # use vertices (edges) for the domain
    # polygon plots need edges; other plots will extract centers if needed
    full_domain = [getattr(mesh, f"x{i}v") for i in range(values.ndim, 0, -1)]

    assert values.ndim == len(full_domain), (
        f"Data dim ({values.ndim}) mismatch with domain dim ({len(full_domain)})"
    )

    # squeeze singleton dimensions for quasi-1D/2D cases
    # keep only dimensions where size > 1
    non_singleton_axes = [i for i in range(values.ndim) if values.shape[i] > 1]

    if len(non_singleton_axes) != values.ndim:
        # squeeze the values
        values = values.squeeze()

        # keep only non-singleton domain axes
        full_domain = [full_domain[i] for i in non_singleton_axes]

    name = f"{field_name}_L{level}" if level > 0 else field_name

    # if effective dim is less than current dim, further squeeze
    # the axis map matters here: a 3D problem with symmetry reduces to a
    # quasi-2D or quasi-1D dataset, so the axis names are selected accordingly.
    axis_names = [INV_AXIS_MAP[i] for i in non_singleton_axes]
    if effective_dim == 1:
        axis_names = ["x1"]
    if effective_dim == 2:
        axis_names = ["x1", "x2"]

    # extract spacing types for non-singleton axes
    mesh_spacing_types = mesh.spacing_types
    spacing_types = [mesh_spacing_types[i] for i in non_singleton_axes]

    # return dimensionally-reduced data for quasi-1D/2D
    return FieldData(
        name=name,
        values=values,
        domain=list(full_domain),
        spacing_types=spacing_types,
        axis_names=axis_names,
        coord_system=CoordSystem(data.metadata.coord_system),
        time=data.metadata.time,
    )


def prepare_figure(
    config: VisualizationConfig,
    nfiles: int = 1,
    projection: Literal["polar", "cartesian"] | None = None,
    nlvls: int = 1,
    coord_system: CoordSystem = CoordSystem.CARTESIAN,
    formatter: Optional[object] = None,
    overlay_mode: bool = False,
) -> Figure:
    """Create and prepare a figure based on configuration.

    Accepts an optional `formatter` argument which will be forwarded to the
    Figure constructor. This allows callers (and tests) to inject a custom
    formatter instance or policy.
    """
    import matplotlib.pyplot as plt

    config.theme.apply(
        nfiles=nfiles,
        nfields=len(config.plot.fields) * nlvls,
        overlay_mode=overlay_mode,
    )
    if projection == "polar":
        fig, ax = plt.subplots(
            1,
            1,
            figsize=config.figure.fig_size,
            subplot_kw={"projection": "polar"},
            layout="constrained",
        )
    else:
        fig = plt.figure(figsize=config.figure.fig_size)
        ax = fig.add_subplot(111)

    # pass optional formatter into the Figure so it can control layout policy
    figure = Figure(config, formatter=formatter)

    figure.fig = fig
    figure.axes["main"] = ax
    figure.coord_system = coord_system

    return figure


def extract_field(field_name: str) -> Callable[[SimData], Array]:
    """Extract a named field from simulation data."""
    return lambda data: data[FIELD_ALIASES.get(field_name, field_name)]


def load_data(file_path: str) -> SimData:
    """
    load simulation data from a file using io.

    Args:
        file_path: path to the checkpoint file

    Returns:
        SimData adapter wrapping Checkpoint
    """
    from simbi.reader import read_checkpoint
    from simbi.reader.adapter import SimData

    checkpoint = read_checkpoint(file_path).unwrap()
    return SimData(checkpoint)


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

    If refined, this "squashes" fine levels onto the base grid.
    If Unigrid, this is a no-op.
    """
    # unigrid case: just return the single 2D field.
    if len(fields_2d) == 1:
        return fields_2d[0]

    # refined case: start with the base level (L0)
    base_field = fields_2d[0]
    base_x, base_y = base_field.domain
    composited_values = base_field.values.copy()

    # loop over finer levels and "squash" them onto the base
    for fine_field in fields_2d[1:]:
        fine_x, fine_y = fine_field.domain
        fine_values = fine_field.values

        # find overlapping indices in the base grid
        i_start = np.searchsorted(base_x, fine_x[0], side="left")
        i_end = np.searchsorted(base_x, fine_x[-1], side="right")
        j_start = np.searchsorted(base_y, fine_y[0], side="left")
        j_end = np.searchsorted(base_y, fine_y[-1], side="right")

        # get the sub-region of the base grid that is covered
        coarse_nx = i_end - i_start
        coarse_ny = j_end - j_start

        # check if the fine grid dimensions are a multiple
        # of the coarse grid region it's covering
        fine_ny, fine_nx = fine_values.shape
        if fine_nx % coarse_nx != 0 or fine_ny % coarse_ny != 0:
            # grid mismatch, cannot perform clean block averaging
            continue

        # downsample the fine data to the coarse grid's resolution
        ref_ratio_x = fine_nx // coarse_nx
        ref_ratio_y = fine_ny // coarse_ny

        averaged_fine_data = _block_average(
            fine_values, (ref_ratio_y, ref_ratio_x)
        )

        # overwrite the base grid data with the averaged fine data
        composited_values[j_start:j_end, i_start:i_end] = averaged_fine_data

    # return a new 2D FieldData object
    return FieldData(
        name=base_field.name,
        time=base_field.time,
        values=composited_values,
        domain=base_field.domain,
        spacing_types=base_field.spacing_types,
        axis_names=base_field.axis_names,
        coord_system=base_field.coord_system,
    )


def _compose_polygons(fields_2d: Sequence[FieldData]) -> FieldData:
    """
    Composes 2D fields for polygon rendering.

    Converts all cells from all levels into a list of patches and values.
    Returns a 1D FieldData object adhering to the "Polygon Contract".
    """
    all_patches = []
    all_values = []

    # track regions covered by finer levels to avoid overplotting
    refined_regions = []

    # iterate from finest level (end of list) to coarsest (start)
    for field in reversed(fields_2d):
        # field.domain is in DATA-STORAGE order (slow..fast = [y, x] for 2D), matching the
        # values array shape (ny, nx) -- see prepare_field_level. unpack it the same way, as
        # [y, x]: unpacking as (x, y) makes the y-edges drive the x-loop and `values[j, i]` run off axis 0 on a
        # non-square (refined) patch. a square level hides the swap (both edge arrays are equal).
        y_edges, x_edges = field.domain
        values = field.values

        # create cell patches for this level
        for j in range(len(y_edges) - 1):
            for i in range(len(x_edges) - 1):
                # check if this cell is covered by an already-processed
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

                # not covered, so add this "leaf cell"
                patch = [
                    (x_edges[i], y_edges[j]),
                    (x_edges[i + 1], y_edges[j]),
                    (x_edges[i + 1], y_edges[j + 1]),
                    (x_edges[i], y_edges[j + 1]),
                ]
                all_patches.append(patch)
                all_values.append(values[j, i])

        # add this level's domain to the list of refined regions
        refined_regions.append(
            {
                "xmin": x_edges[0],
                "xmax": x_edges[-1],
                "ymin": y_edges[0],
                "ymax": y_edges[-1],
            }
        )

    # convert refined_regions to level_bounds tuples (xmin, xmax, ymin, ymax)
    # reverse to get coarsest-to-finest order (level 0, 1, 2, ...)
    level_bounds: list[tuple[float, float, float, float]] = [
        (r["xmin"], r["xmax"], r["ymin"], r["ymax"])
        for r in reversed(refined_regions)
    ]

    axis_names = fields_2d[0].axis_names
    # return a new 1D FieldData object (the "Polygon Contract")
    return FieldData(
        name=f"{fields_2d[0].name}_polygons",
        values=np.array(all_values),
        domain=np.array(all_patches),
        axis_names=axis_names,
        coord_system=fields_2d[0].coord_system,
        time=fields_2d[0].time,
        level_bounds=level_bounds if len(level_bounds) > 1 else None,
    )


def compose_2d_render(
    fields_2d: list[FieldData],
    render_mode: Literal["pcolormesh", "polygons"],
) -> FieldData:
    """
    Composes a list of 2D fields into a single renderable FieldData object.

    This is the "stitching" step for 2D refined or polygon plots.
    """
    if not fields_2d:
        raise ValueError("Cannot compose an empty list of fields.")

    if render_mode == "pcolormesh":
        # returns a single 2D FieldData object
        return _compose_pcolormesh(fields_2d)
    elif render_mode == "polygons":
        # returns a single 1D FieldData object (of polygons)
        return _compose_polygons(fields_2d)
    else:
        raise ValueError(f"Unknown render_mode: {render_mode}")


def compose_fields_for_render(
    fields: Sequence[FieldData], config: "VisualizationConfig"
) -> Sequence[FieldData]:
    """
    pure function: applies composition logic based on refinement and dimensionality.

    returns fields ready for component dispatch (may be polygons, pcolormesh, or unchanged).
    """
    if not fields:
        return fields

    nlvls = 1 + sum("_L" in f.name for f in fields)
    is_refined = nlvls > 1

    # refined data MUST use polygons (pcolormesh can't handle different grids)
    if is_refined:
        use_polygons = True
    else:
        use_polygons = config.refinement.render_mode == "polygons"

    is_2d = fields[0].ndim == 2
    if is_2d and use_polygons:
        return [_compose_polygons(list(fields))]
    else:
        return fields
