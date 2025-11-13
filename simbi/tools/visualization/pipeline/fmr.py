from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ....core.types import Array, MeshConfig
from ....reader.lazy import SimData
from ..core.types import FieldData
from .transforms import create_field_data, create_slicer_from_config


@dataclass
class BoxND:
    """N-dimensional box for refinement region"""

    lower: tuple[int, ...]  # Lower indices in coarse grid
    upper: tuple[int, ...]  # Upper indices in coarse grid
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
    coarse_mesh: MeshConfig, fine_mesh: MeshConfig, ref_ratio: int
) -> list[BoxND]:
    """Compute boxes representing refined regions in coarse grid coordinates"""
    boxes = []

    # Convert fine mesh bounds to coarse grid indices
    for dim in range(len(coarse_mesh.shape)):
        coarse_coords = coarse_mesh.get(f"x{dim + 1}c")
        fine_coords = fine_mesh.get(f"x{dim + 1}c")
        if coarse_coords is None or fine_coords is None:
            continue

        # Find coarse grid indices that bound the fine grid
        lower = np.searchsorted(coarse_coords, fine_coords[0])
        upper = np.searchsorted(coarse_coords, fine_coords[-1])

        boxes.append(BoxND(lower=(lower,), upper=(upper,), ref_ratio=ref_ratio))

    return boxes


def create_composite_field(
    field_name: str, data: SimData, active_levels: set[int], ndim: int
) -> tuple[Array, Sequence[Array]]:
    """Create a composite view of a field combining multiple refinement levels"""
    if not data.has_refinement():
        # Single level case
        values = data.get_field(field_name, 0)
        mesh = data.base_mesh
        coords = []
        for i in range(1, ndim + 1):
            coord = getattr(mesh, f"x{i}c")
            if coord is not None:
                coords.append(coord)
        return values, coords

    # Start with base level data
    base_values = data.get_field(field_name, 0)
    base_mesh = data.base_mesh

    # Get base coordinates
    base_coords = []
    for i in range(1, ndim + 1):
        coord = getattr(base_mesh, f"x{i}c")
        if coord is not None:
            base_coords.append(coord)

    # Create composite array starting with base level
    composite = base_values.copy()

    # Process each active refinement level in order
    for level in sorted(active_levels - {0}):
        if level >= data.num_levels:
            continue

        # Get level data
        level_values = data.get_field(field_name, level)
        level_mesh = data.level_mesh(level)

        # Get refinement ratio from hierarchy
        if not data.hierarchy():
            continue
        ref_ratio = data.hierarchy().ref_ratios[level - 1]

        # Compute refinement boxes for this level
        boxes = compute_refinement_boxes(base_mesh, level_mesh, ref_ratio)

        # Update composite with refined data
        composite = overlay_refined_data(
            composite, level_values, boxes, ref_ratio
        )

    return composite, base_coords


def overlay_refined_data(
    coarse_data: Array, fine_data: Array, boxes: list[BoxND], ref_ratio: int
) -> Array:
    """Overlay fine grid data onto coarse grid within refinement boxes"""
    result = coarse_data.copy()

    for box in boxes:
        # Get coarse grid region indices
        slices = tuple(slice(x, u + 1) for x, u in zip(box.lower, box.upper))

        # Calculate corresponding fine grid indices
        fine_slices = tuple(
            slice(x * ref_ratio, (u + 1) * ref_ratio)
            for x, u in zip(box.lower, box.upper)
        )

        # Replace coarse data with averaged fine data
        fine_region = fine_data[fine_slices]

        # Average fine data to coarse grid resolution
        # For each coarse cell, average the corresponding fine cells
        axes = tuple(range(fine_region.ndim))
        kernel_size = tuple(ref_ratio for _ in range(fine_region.ndim))
        averaged = nd_reshape_mean(fine_region, kernel_size, axes)

        # Update coarse grid with averaged fine data
        result[slices] = averaged

    return result


def nd_reshape_mean(
    arr: Array, shape: tuple[int, ...], axes: tuple[int, ...]
) -> Array:
    """N-dimensional reshaping and averaging for arbitrary dimensions"""
    if len(shape) != len(axes):
        raise ValueError("Shape and axes must have same length")

    # Calculate output shape
    out_shape = list(arr.shape)
    for axis, size in zip(axes, shape):
        out_shape[axis] = out_shape[axis] // size

    # Reshape and average
    new_shape = []
    for i, dim in enumerate(arr.shape):
        if i in axes:
            idx = axes.index(i)
            new_shape.extend([out_shape[i], shape[idx]])
        else:
            new_shape.append(dim)

    reshaped = arr.reshape(new_shape)

    # Average along reshape axes
    for i, axis in enumerate(axes):
        reshaped = np.mean(reshaped, axis=axis + i + 1)

    return reshaped


def transform_composite_field(
    data: SimData,
    field_name: str,
    active_levels: set[int],
    effective_dim: int,
    slice_config: Optional[dict[str, Any]] = None,
) -> FieldData:
    """Transform field data creating a composite view across levels"""
    # Create composite field
    values, coords = create_composite_field(
        field_name, data, active_levels, effective_dim
    )

    # Apply slicing if needed
    if slice_config and effective_dim >= 2:
        slicer = create_slicer_from_config(slice_config)
        values, coords = slicer({"values": values, "domain": coords})

    return create_field_data(field_name, values, coords)
