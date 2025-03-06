from typing import Sequence
from .config.constants import (
    AVAILABLE_COORD_SYSTEMS,
    AVAILABLE_REGIMES,
    AVAILABLE_BOUNDARY_CONDITIONS,
)


def validate_coordinate_system(coord_system: str) -> None:
    if coord_system not in AVAILABLE_COORD_SYSTEMS:
        raise ValueError(
            f"Invalid coordinate system. Expected one of: {AVAILABLE_COORD_SYSTEMS}. "
            f"Got: {coord_system}"
        )


def validate_boundary_conditions(bcs: Sequence[str], dimensionality: int) -> list[str]:
    bcs = list(helpers.to_iterable(bcs))
    invalid_bcs = [bc for bc in bcs if bc not in AVAILABLE_BOUNDARY_CONDITIONS]
    if invalid_bcs:
        raise ValueError(
            f"Invalid boundary condition(s): {invalid_bcs}. "
            f"Expected one of: {AVAILABLE_BOUNDARY_CONDITIONS}."
        )
    return _normalize_boundary_conditions(bcs, dimensionality)
