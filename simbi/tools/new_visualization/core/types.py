"""Core type definitions for the visualization system."""

from enum import Enum
from typing import (
    Sequence,
    TypeVar,
)
from pydantic import BaseModel, field_validator
from ....core.types import Array, IArray

# Type variables for generic functions
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class CoordSystem(str, Enum):
    """Coordinate system enumeration."""

    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    PLANAR_CYLINDRICAL = "planar_cylindrical"
    AXIS_CYLINDRICAL = "axis_cylindrical"


class FieldData(BaseModel):
    """
    Data for a single field to visualize.

    Attributes:
        name: Name of the field (e.g., "rho", "pressure")
        values: Array of field values
        domain: Sequence of coordinate arrays, one for each dimension
               (e.g., [x_coords, y_coords] for a 2D field)
    """

    name: str
    values: Array
    domain: Sequence[Array]  # Coordinate arrays for each dimension
    model_config = {
        "arbitrary_types_allowed": True,  # Allow arbitrary types like Array
        "frozen": True,  # Make instances immutable
    }

    @field_validator("domain")
    @classmethod
    def validate_domain_dimensions(
        cls, domain: Sequence[Array], info
    ) -> Sequence[Array]:
        """Validate that the domain has the correct number of dimensions for the field values."""
        values = info.data
        if "values" in values:
            field_ndim = values["values"].ndim
            if len(domain) != field_ndim:
                raise ValueError(
                    f"Domain has {len(domain)} dimensions but field values have {field_ndim} dimensions"
                )
        return domain

    @property
    def ndim(self) -> int:
        """Number of dimensions in the field."""
        return self.values.ndim


class PlotData(BaseModel):
    """
    Complete data needed for visualization.

    Attributes:
        fields: Sequence of field data objects to visualize
        time: Simulation time
        dimensions: Number of spatial dimensions in the simulation
        coord_system: Coordinate system used
    """

    fields: Sequence[FieldData]
    time: float
    dimensions: int
    coord_system: CoordSystem
    model_config = {
        "arbitrary_types_allowed": True,  # Allow arbitrary types like CoordSystem
        "frozen": True,  # Make instances immutable
    }

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Validate that dimensions is between 1 and 3."""
        if v < 1 or v > 3:
            raise ValueError(f"Dimensions must be between 1 and 3, got {v}")
        return v


class Bounds(BaseModel):
    """
    Bounds for a dimension (min, max).

    Attributes:
        min: Minimum value
        max: Maximum value
    """

    min: float
    max: float
    model_config = {
        "frozen": True,  # Make instances immutable
    }

    @field_validator("max")
    @classmethod
    def max_greater_than_min(cls, v: float, info) -> float:
        """Validate that max is greater than min."""
        values = info.data
        if "min" in values and v <= values["min"]:
            raise ValueError(
                f"Max value {v} must be greater than min value {values['min']}"
            )
        return v


class ColorRange(Bounds):
    """
    Color range for a visualization (min, max).

    This is used to define the mapping from data values to colors.
    """

    pass
