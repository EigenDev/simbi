# =============================================================================
# types.py
#
# core type definitions and contracts for the visualization system.
#
# this module defines the data carriers used across the viz pipeline:
#   - FieldData: immutable container for a single field and its domain
#   - PlotData: package of fields + metadata needed by components
#   - RenderResult: standardized contract returned by component.render()
#   - Bounds/ColorRange: simple numeric range helpers used by style/config
#
# render contract (RenderResult):
#   a component returns the artists it drew, plus the facts the figure and the
#   formatter cannot work out for themselves: which artist a colorbar
#   describes and what to call it, the extent it drew (a mesh collection
#   carries no data limit of its own), the names of its series, and whether it
#   is a vector overlay riding on another artist.
#
#   each of those is a named field rather than a free-form dictionary entry.
#   an unrecognised key silently disables a feature, and a key that means
#   something else silently enables one: `label` on a field render once
#   switched on the legend handling meant for lines.
#
#   example:
#     RenderResult(
#         artists={"mesh": quadmesh},
#         mappable=quadmesh,
#         colorbar_label="rho",
#         view_bounds=(x0, x1, y0, y1),
#     )
# =============================================================================
"""Core type definitions for the visualization system."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence, TypeVar

from pydantic import BaseModel, field_validator

from simbi.types import Array, HierarchyData
from simbi.types.bodies import (
    BodySystemConfig,
    ImmersedBodyConfig,
)

# type variables for generic functions
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
body_system_t = BodySystemConfig | Sequence[ImmersedBodyConfig] | None


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
        name: Name of the field (e.g., "rho", "pressure", "mdot_vs_r")
        values: Array of field values
        domain: Sequence of coordinate arrays, one for each dimension
               (e.g., [x_coords, y_coords] for a 2D field). for 1D fields
               domain will commonly be a single array of bin centers.
        spacing_types: optional tuple of spacing types ("linear", "log") per dimension
                      used for correct cell center computation
        coord_system: optional CoordSystem hint for formatting (polar/cartesian)
        axis_names: optional human-readable axis names to use as xlabel/ylabel
        body_names: optional list of body identifiers present in the data
    """

    name: str
    values: Array
    domain: Sequence[Array] | Array
    spacing_types: Optional[Sequence[str]] = None
    time: Optional[float] = None
    coord_system: Optional[CoordSystem] = None
    axis_names: Optional[Sequence[str]] = None
    body_names: Optional[Sequence[str]] = None
    # level bounds for AMR visualization: list of (xmin, xmax, ymin, ymax) per level
    level_bounds: Optional[Sequence[tuple[float, float, float, float]]] = None
    model_config = {
        "arbitrary_types_allowed": True,  # Allow arbitrary types like Array
        "frozen": True,  # Make instances immutable
    }

    @property
    def ndim(self) -> int:
        """Number of dimensions in the field."""
        return self.values.ndim


class PolygonData(BaseModel):
    """Cells drawn one polygon at a time.

    This is what a level hierarchy composes to: a quadmesh is a single
    logically-rectangular lattice and cannot carry cells of two different
    sizes, so a refined field is drawn as a soup of independent quadrilaterals
    instead.

    It is a separate type from FieldData because it is a different shape of
    thing. A field has an axis per dimension and a coordinate array per axis;
    this has one flat list of cells and a list of corners each. Handed to
    something that slices or bins a field, the corner axis reads as a
    coordinate axis and the answer is meaningless rather than wrong-looking.

    Attributes:
        patches: (n_cells, 4, 2) corners, anticlockwise from the lower-left
        values: one value per cell
        level_bounds: (xmin, xmax, ymin, ymax) per level, coarsest first
    """

    name: str
    patches: Array
    values: Array
    coord_system: Optional[CoordSystem] = None
    time: Optional[float] = None
    axis_names: Optional[Sequence[str]] = None
    level_bounds: Optional[Sequence[tuple[float, float, float, float]]] = None

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    @field_validator("patches")
    @classmethod
    def validate_patches(cls, v: Array) -> Array:
        if v.ndim != 3 or v.shape[1:] != (4, 2):
            raise ValueError(
                f"patches must be (n_cells, 4, 2) quadrilateral corners, got {v.shape}"
            )
        return v

    @property
    def ndim(self) -> int:
        """the dimension of the region drawn, which is a plane however the
        cells are stored."""
        return 2


class RenderResult(BaseModel):
    """
    Standardized return value for component.render().

    Purpose:
      - provide a stable, typed contract for component outputs so the
        orchestration layer (Figure) and layout layer (FigureFormatter)
        can make safe decisions about legends, colorbars, and labels.

    Every field beyond `artists` tells the formatter something it cannot work
    out for itself, and each is named rather than passed in a free dictionary:
    a key the formatter does not recognise silently disables a feature, and a
    key it recognises for something else silently enables one. `label` on a
    field render, for instance, once switched on legend handling meant for
    lines.
    """

    # semantic key -> matplotlib artist ('mesh', 'collection', 'line', ...)
    artists: dict[str, object]

    # the colour-mapped artist a colorbar describes, and the quantity it draws.
    # a vector overlay is colour-mapped too but reads off the field beneath it,
    # so it leaves these unset.
    mappable: Optional[object] = None
    colorbar_label: Optional[str] = None

    # (x_min, x_max, y_min, y_max) of what was drawn, in the axes' own
    # coordinates. a mesh collection carries no data limit of its own, so this
    # is how the figure composes a view that holds every component.
    view_bounds: Optional[tuple[float, float, float, float]] = None

    # names for the drawn series: one becomes an axis label, several a legend
    labels: Sequence[str] = ()

    # a direction field drawn over another artist, which owns neither the
    # colorbar nor the axis labels
    is_vector: bool = False

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True,
    }


class PlotData(BaseModel):
    """
    Complete data needed for visualization.

    Attributes:
        fields: Sequence of FieldData objects to visualize
        time: simulation time (float)
        dimensions: Number of spatial dimensions in the simulation
        coord_system: Coordinate system used
        hierarchy: optional AMR hierarchy data (for refinement-aware components)
    """

    fields: Sequence[FieldData]
    time: Optional[float] = None
    dimensions: Optional[int] = None
    coord_system: Optional[CoordSystem] = None
    hierarchy: Optional[HierarchyData] = None

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Validate that dimensions is between 1 and 3."""
        if v is None:
            return v
        if v < 1 or v > 3:
            raise ValueError(f"Dimensions must be between 1 and 3, got {v}")
        return v

    def has_refinement(self) -> bool:
        """Check if this plot data contains refinement levels"""
        return self.hierarchy is not None

    def get_level_fields(self, level: int) -> list[FieldData]:
        """Get all fields for a specific level"""
        return [f for f in self.fields if f.name.endswith(f"_L{level}")]

    def get_base_fields(self) -> list[FieldData]:
        """Get all base level fields"""
        return [f for f in self.fields if "_L" not in f.name]

    def count_plot_lines(self) -> int:
        """Count total number of individual lines that will be rendered."""
        total = 0
        for field in self.fields:
            if field.values.ndim == 1:
                total += 1
            elif field.values.ndim == 2:
                # 2d field data means N_bodies individual lines
                total += field.values.shape[1]
        return total

    model_config = {
        "arbitrary_types_allowed": True,  # Allow arbitrary types like CoordSystem
        "frozen": True,  # Make instances immutable
    }


class Bounds(BaseModel):
    """
    Bounds for a dimension (min, max).

    Either end may be left out, and an absent end means the data sets it:
    clipping the top of a colour scale while the bottom follows the field is an
    ordinary thing to ask for, and requiring both ends turns it into a
    "field required" error on a range that is perfectly well formed.

    Attributes:
        min: Minimum value, or None to take it from the data
        max: Maximum value, or None to take it from the data
    """

    min: float | None = None
    max: float | None = None
    model_config = {
        "frozen": True,  # Make instances immutable
    }

    @field_validator("max")
    @classmethod
    def max_greater_than_min(cls, v: float | None, info) -> float | None:
        """Validate that max is greater than min.

        either end may be left open, in which case there is nothing to compare
        it against: a half-open range asks for one limit and lets the data set
        the other.
        """
        low = info.data.get("min")
        if v is None or low is None:
            return v

        if v <= low:
            raise ValueError(
                f"Max value {v} must be greater than min value {low}"
            )
        return v


class ColorRange(Bounds):
    """
    Color range for a visualization (min, max).

    This is used to define the mapping from data values to colors. Components
    and formatters should prefer the style configuration but honor an explicit
    ColorRange when present in component metadata or props.
    """

    pass
