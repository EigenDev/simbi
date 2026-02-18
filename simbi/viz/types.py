# =============================================================================
# types.py
#
# core type definitions for the visualization system.
# - FieldData: immutable container for a single field and its domain
# - PlotData: package of fields + metadata needed by components
# - RenderResult: return type for component.render()
# - Bounds/ColorRange: numeric range helpers
# =============================================================================
from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Optional, Sequence

from pydantic import BaseModel, field_validator

from simbi.reader.io import HierarchyInfo
from simbi.types import Array


class CoordSystem(str, Enum):
    """Coordinate system enumeration."""

    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    PLANAR_CYLINDRICAL = "planar_cylindrical"
    AXIS_CYLINDRICAL = "axis_cylindrical"


class FieldData(BaseModel):
    """immutable container for a single field and its coordinate domain."""

    name: str
    values: Array
    domain: Sequence[Array] | Array
    spacing_types: Optional[Sequence[str]] = None
    time: Optional[float] = None
    coord_system: Optional[CoordSystem] = None
    axis_names: Optional[Sequence[str]] = None
    body_names: Optional[Sequence[str]] = None
    level_bounds: Optional[Sequence[tuple[float, float, float, float]]] = None
    bands: Optional[tuple[Array, Array]] = None  # (lower, upper) percentile bands

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True,
    }

    @property
    def ndim(self) -> int:
        return self.values.ndim


class RenderResult(NamedTuple):
    """return value for component.render()."""

    artists: dict[str, object]
    metadata: Optional[dict[str, object]] = None


class PlotData(BaseModel):
    """package of fields + metadata for visualization."""

    fields: Sequence[FieldData]
    body_collection: Optional[object] = None
    time: Optional[float] = None
    dimensions: Optional[int] = None
    coord_system: Optional[CoordSystem] = None
    hierarchy: Optional[HierarchyInfo] = None
    extra: Optional[dict] = None

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        if v is None:
            return v
        if v < 1 or v > 3:
            raise ValueError(f"Dimensions must be between 1 and 3, got {v}")
        return v

    def count_plot_lines(self) -> int:
        """count total number of individual lines that will be rendered."""
        total = 0
        for field in self.fields:
            if field.values.ndim == 1:
                total += 1
            elif field.values.ndim == 2:
                total += field.values.shape[1]
        return total

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True,
    }


PlotData.model_rebuild()


class Bounds(BaseModel):
    """numeric range (min, max)."""

    min: float | None
    max: float | None

    model_config = {"frozen": True}

    @field_validator("max")
    @classmethod
    def max_greater_than_min(cls, v: float | None, info) -> float | None:
        if v is None:
            return None
        values = info.data
        if "min" in values and v <= values["min"]:
            raise ValueError(
                f"Max value {v} must be greater than min value {values['min']}"
            )
        return v


class ColorRange(Bounds):
    """color range for mapping data values to colors."""

    pass
