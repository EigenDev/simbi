from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class RefinedLineData:
    """Represents line data for a single field across refinement levels.

    Attributes:
        level_data: list of (x, y) arrays per refinement level
        boundaries: list of (x, y) coordinates where refinement levels change
        max_level: Maximum refinement level in the data
    """

    level_data: list[tuple[Array, Array]]
    boundaries: list[tuple[float, float]]
    max_level: int


def detect_level_boundaries(
    x_data: Array, y_data: Array, level_info: Array
) -> list[tuple[float, float]]:
    """Detect points where refinement levels change.

    Args:
        x_data: x-coordinates of the data
        y_data: y-coordinates of the data
        level_info: Array containing refinement level for each point

    Returns:
        list of (x, y) coordinates where refinement levels change
    """
    boundaries = []
    if len(level_info) < 2:
        return boundaries

    # Find indices where level changes
    level_changes = np.where(np.diff(level_info) != 0)[0]

    # For each change, record the boundary point
    for idx in level_changes:
        # Use the point at the change boundary
        boundaries.append((float(x_data[idx]), float(y_data[idx])))

    return boundaries


def split_by_refinement_level(
    x_data: Array, y_data: Array, level_info: Array
) -> dict[int, tuple[Array, Array]]:
    """Split data into separate arrays by refinement level.

    Args:
        x_data: x-coordinates of the data
        y_data: y-coordinates of the data
        level_info: Array containing refinement level for each point

    Returns:
        dictionary mapping refinement level to (x_data, y_data) arrays
    """
    unique_levels = np.unique(level_info)
    level_dict = {}

    for level in unique_levels:
        mask = level_info == level
        level_dict[int(level)] = (x_data[mask], y_data[mask])

    return level_dict


def compose_line_segments(
    field: "FieldData", show_all_levels: bool = True
) -> RefinedLineData:
    """Process field data into refinement-aware line segments.

    Args:
        field: Field data containing values and refinement information
        show_all_levels: If True, return data for all levels. Otherwise,
                        return only the finest available level at each point.

    Returns:
        RefinedLineData object containing processed line segments
    """
    # Extract base data
    x_data = field.domain[0] if field.domain else np.arange(len(field.values))
    y_data = field.values

    # Get refinement levels
    level_info = (
        field.refinement_level
        if hasattr(field, "refinement_level")
        else np.zeros_like(x_data)
    )

    # Detect level boundaries
    boundaries = detect_level_boundaries(x_data, y_data, level_info)

    # Split data by level
    level_dict = split_by_refinement_level(x_data, y_data, level_info)

    if not show_all_levels:
        # If we only want the finest level at each point,
        # we need to process the data differently
        # This is a placeholder for now - we'll implement this later
        pass

    # Convert dictionary to list sorted by level
    max_level = max(level_dict.keys())
    level_data = [
        level_dict[level]
        for level in range(max_level + 1)
        if level in level_dict
    ]

    return RefinedLineData(
        level_data=level_data, boundaries=boundaries, max_level=max_level
    )
