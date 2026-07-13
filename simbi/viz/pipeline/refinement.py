from dataclasses import dataclass
from typing import Sequence

import numpy as np

from simbi.reader.adapter import SimData
from simbi.reader.computation import FieldComputationError


def _get_field_loud(data: SimData, field_name: str, level: int):
    """get_field with a composite-path failure that NAMES the available fields
    instead of a bare KeyError from deep inside the level walk."""
    try:
        return data.get_field(field_name, level)
    except KeyError:
        available = ", ".join(sorted(data.available_fields(0)))
        raise FieldComputationError(
            f"field '{field_name}' is not available at level {level}; "
            f"available fields: {available}"
        ) from None
from simbi.types import MeshConfig

from ..types import Array, FieldData
from .transforms import (
    INV_AXIS_MAP,
)


@dataclass
class BoxND:
    """N-dimensional box for refinement region"""

    lower: tuple[int, ...]  # Lower indices in coarse grid (data order: z, y, x)
    upper: tuple[int, ...]  # Upper indices in coarse grid (data order: z, y, x)
    ref_ratio: int  # Refinement ratio to next level

    @property
    def ndim(self) -> int:
        return len(self.lower)

    def contains(self, indices: tuple[int, ...]) -> bool:
        """Check if indices are within this box"""
        return all(
            x <= i <= u for i, x, u in zip(indices, self.lower, self.upper)
        )


def compute_refinement_boxes(
    coarse_mesh: MeshConfig, fine_mesh: MeshConfig, ref_ratio: int, ndim: int
) -> list[BoxND]:
    """
    Compute boxes representing refined regions in coarse grid coordinates.

    Ensures box indices are in data order (e.g., z, y, x).
    """
    lower_indices = []
    upper_indices = []

    # Iterate in data order (axis 0, 1, 2...)
    for axis_index in range(ndim):
        # Get logical name (x3, x2, x1) from data index (0, 1, 2)
        logical_name = INV_AXIS_MAP.get(axis_index)
        if logical_name is None:
            continue  # Or raise error if mapping is incomplete

        coarse_coords = coarse_mesh.get(f"{logical_name}c")
        fine_coords = fine_mesh.get(f"{logical_name}c")

        if coarse_coords is None or fine_coords is None:
            # Fallback or error if coordinates are missing
            lower_indices.append(0)
            upper_indices.append(0)
            continue

        # Find coarse grid indices that bound the fine grid
        lower = np.searchsorted(coarse_coords, fine_coords[0], side="left")
        upper = np.searchsorted(coarse_coords, fine_coords[-1], side="right")
        # Ensure upper index is inclusive and valid
        upper = max(lower, upper - 1)

        lower_indices.append(lower)
        upper_indices.append(upper)

    # Assumes a single contiguous refined box
    box = BoxND(
        lower=tuple(lower_indices),
        upper=tuple(upper_indices),
        ref_ratio=ref_ratio,
    )
    return [box]


def create_composite_field(
    field_name: str, data: SimData, active_levels: set[int], ndim: int
) -> tuple[Array, Sequence[Array]]:
    """
    Create a composite view of a field combining multiple refinement levels.

    Returns:
        A tuple of (composite_values, data_order_domain).
    """
    base_mesh = data.level_mesh(0)

    # Get base coordinates in *logical* order (x1, x2, x3)
    logical_coords = []
    for i in range(1, ndim + 1):
        coord = getattr(base_mesh, f"x{i}v")
        if coord is not None:
            logical_coords.append(coord)

    # Reverse to get *data* order (x3, x2, x1) to match array shape
    data_order_domain = list(reversed(logical_coords))

    if not data.has_refinement() or len(active_levels - {0}) == 0:
        # Single level case
        values = _get_field_loud(data, field_name, 0)
        return values, data_order_domain

    # Start with base level data
    base_values = _get_field_loud(data, field_name, 0)
    composite = base_values.copy()

    hierarchy = data.hierarchy()
    if not hierarchy:
        return (
            composite,
            data_order_domain,
        )  # Should not happen if has_refinement

    # Process each active refinement level in order
    for level in sorted(active_levels - {0}):
        if level >= data.num_levels:
            continue

        level_values = _get_field_loud(data, field_name, level)
        level_mesh = data.level_mesh(level)
        ref_ratio = hierarchy.ref_ratios[level - 1]

        # Compute refinement boxes for this level
        boxes = compute_refinement_boxes(base_mesh, level_mesh, ref_ratio, ndim)

        # Update composite with refined data
        composite = overlay_refined_data(
            composite, level_values, boxes, ref_ratio
        )

    return composite, data_order_domain


def block_average(arr: Array, block_shape: tuple[int, ...]) -> Array:
    """N-dimensional block averaging (downsampling)."""
    if len(arr.shape) != len(block_shape):
        raise ValueError("Array shape and block shape must have same ndim.")

    # New shape for reshaping, e.g., (nz, Rz, ny, Ry, nx, Rx)
    new_shape = []
    for i, dim in enumerate(arr.shape):
        block_size = block_shape[i]
        if dim % block_size != 0:
            raise ValueError(
                f"Axis {i} (size {dim}) not divisible by "
                f"block size {block_size}"
            )
        new_shape.extend([dim // block_size, block_size])

    reshaped = arr.reshape(tuple(new_shape))

    # Axes to average over (the new block axes)
    avg_axes = tuple(range(1, len(new_shape), 2))

    return np.mean(reshaped, axis=avg_axes)


def overlay_refined_data(
    coarse_data: Array, fine_data: Array, boxes: list[BoxND], ref_ratio: int
) -> Array:
    """Overlay fine grid data onto coarse grid within refinement boxes"""
    result = coarse_data.copy()

    for box in boxes:
        # Get coarse grid region indices (z, y, x)
        slices = tuple(slice(x, u + 1) for x, u in zip(box.lower, box.upper))

        # Calculate corresponding fine grid indices
        # Note: This assumes the fine_data array starts at index 0
        # for the refined region.
        fine_slices = tuple(
            slice(0, (u - x + 1) * ref_ratio)
            for x, u in zip(box.lower, box.upper)
        )

        fine_region = fine_data[fine_slices]

        # Average fine data to coarse grid resolution
        block_shape = tuple(ref_ratio for _ in range(fine_region.ndim))
        averaged = block_average(fine_region, block_shape)

        # Ensure averaged data has the correct shape
        expected_shape = tuple(u - x + 1 for x, u in zip(box.lower, box.upper))
        if averaged.shape != expected_shape:
            # This can happen due to off-by-one in box computation
            # Simple truncation is one fix:
            s = tuple(slice(0, s) for s in expected_shape)
            averaged = averaged[s]

        # Update coarse grid with averaged fine data
        result[slices] = averaged

    return result


def prepare_composite_field(
    data: SimData,
    field_name: str,
    active_levels: set[int],
    effective_dim: int,
) -> FieldData:
    """
    Prepares a single composite FieldData object.
    NO slicing is performed.
    """
    # This returns (nz, ny, nx) data and [x3_arr, x2_arr, x1_arr] domain
    values, data_order_domain = create_composite_field(
        field_name, data, active_levels, effective_dim
    )

    if values.ndim != len(data_order_domain):
        raise ValueError(
            f"Composite field dim ({values.ndim}) does not match "
            f"domain dim ({len(data_order_domain)})"
        )

    # Return the full-dimensional data
    # use base level mesh for spacing types
    base_mesh = data.level_mesh(0)
    spacing_types = list(base_mesh.spacing_types)

    return FieldData(
        name=field_name,
        values=values,
        domain=list(data_order_domain),
        spacing_types=spacing_types,
        time=data.metadata.time,
    )
