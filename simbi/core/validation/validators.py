"""
Validation functions for simbi configurations.

This module provides validators for various configuration components.
"""

from typing import Sequence, Union
from pydantic import ValidationInfo
from enum import Enum


class BoundaryCondition(str, Enum):
    """Valid boundary conditions"""

    OUTFLOW = "outflow"
    REFLECTING = "reflecting"
    DYNAMIC = "dynamic"
    PERIODIC = "periodic"


def validate_bounds(
    bounds: Sequence[Sequence[float]], info: ValidationInfo
) -> Sequence[Sequence[float]]:
    """Validate domain bounds.

    Args:
        bounds: Sequence of [min, max] bounds for each dimension
        info: Validation context information

    Returns:
        Validated bounds

    Raises:
        ValueError: If bounds are invalid
    """
    # Ensure each bound is a pair
    for i, bound in enumerate(bounds):
        if len(bound) != 2:
            raise ValueError(f"Bound {i} must be [min, max], got {bound}")

        # Ensure min < max
        if bound[0] >= bound[1]:
            raise ValueError(f"Bound {i} must have min < max, got {bound}")

    return bounds


def validate_boundary_conditions(
    bcs: Union[str, list[str]], info: ValidationInfo
) -> list[str]:
    """Validate boundary conditions.

    Args:
        bcs: Boundary conditions (single string or Sequence)
        info: Validation context information

    Returns:
        Sequence of validated boundary conditions

    Raises:
        ValueError: If boundary conditions are invalid
    """
    # Get model data to determine dimensionality
    model_data = info.data

    # Determine dimensionality if available, default to 1
    dim = 1
    if hasattr(info.data, "get") and "resolution" in info.data:
        res = info.data["resolution"]
        if isinstance(res, Sequence):
            dim = len(res)

    # Convert string to Sequence
    if isinstance(bcs, str):
        bcs = [bcs] * (2 * dim)  # 2 faces per dimension

    # Validate length
    expected_len = 2 * dim  # 2 faces per dimension
    if len(bcs) != expected_len and len(bcs) != expected_len // 2:
        raise ValueError(
            f"Expected {expected_len} or {expected_len // 2} boundary conditions for {dim}D, got {len(bcs)}"
        )

    # Validate values
    valid_bcs = [bc.value for bc in BoundaryCondition]
    for bc in bcs:
        if bc not in valid_bcs:
            raise ValueError(
                f"Invalid boundary condition: {bc}. Valid options: {valid_bcs}"
            )

    # If given half the needed BCs, duplicate for inner/outer
    if len(bcs) == expected_len // 2:
        bcs = bcs * 2

    return bcs
