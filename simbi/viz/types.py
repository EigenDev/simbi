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
# render contract (renderresult):
#   components MUST return a RenderResult describing the matplotlib artists
#   they created and any metadata that will help the figure/formatter decide
#   on layout and additional presentation (legend, colorbar, etc).
#
#   rationale:
#     - previously components returned loose dicts or lists. that forced
#       the Figure to guess semantics and do formatting. renderresult makes
#       component outputs explicit and testable and lets the FigureFormatter
#       safely decide about colorbars, legends, and axis labels.
#
#   minimal expectations:
#     - `artists` (dict[str, object]): mapping of semantic keys -> artist object
#         common semantic keys:
#           - "mesh": a QuadMesh / pcolormesh artist (mappable for colorbar)
#           - "collection": a PolyCollection or PatchCollection (mappable)
#           - "line" / "lines": Line2D or list of Line2D objects
#           - "quiver": Quiver object
#           - "streamplot": the streamplot return object
#           - "refs", "vlines", etc: auxiliary artists that belong to the
#             component but are not mappables
#
#     - `metadata` (optional dict[str, object]): hints about semantics and
#         presentation preferences. metadata is advisory only; the
#         FigureFormatter applies policies conservatively.
#
#   recommended metadata keys (convention used across repo)
#     - "mappable": an explicit reference to the mappable artist (mesh or collection)
#         type: matplotlib artist object (ScalarMappable)
#         use: direct pointer for the formatter to create a colorbar
#
#     - "label": preferred label string for the component (e.g., "$\rho$")
#         type: str
#         use: used for y-axis labels, legend entries or colorbar labels when
#              the component doesn't provide them via artist properties.
#
#     - "is_line": boolean
#         type: bool
#         use: explicitly mark the component as line-like for legend policy.
#
#     - "is_vector": boolean
#         type: bool
#         use: indicates vector-field visualizations (quiver). typically this
#              suppresses legend behavior and signals specialized formatting.
#
#     - "preferred_cmap": string or Colormap
#         type: str | matplotlib.colors.Colormap
#         use: suggests a colormap when constructing the mappable; the
#              component still owns the actual artist creation.
#
#     - "color_range": {"min": float, "max": float} or ColorRange
#         type: dict | ColorRange
#         use: explicit vmin/vmax guiding color normalization. components
#              should honor style.config but can include this override when
#              the analysis needs a fixed stretch.
#
#   examples:
#     RenderResult(
#         artists={"mesh": quadmesh, "quiver": quiv},
#         metadata={"mappable": quadmesh, "is_vector": True}
#     )
#
#     RenderResult(
#         artists={"line": main_line, "refs": [vline1, vline2]},
#         metadata={"label": r"$\rho$", "is_line": True}
#     )
#
# notes:
#   - components that create no artists may return RenderResult(artists={}, metadata={})
#   - the figure and formatter must tolerate legacy returns (plain dict or list),
#     but new code should use RenderResult for clarity.
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

# Type variables for generic functions
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


class RenderResult(BaseModel):
    """
    Standardized return value for component.render().

    Purpose:
      - provide a stable, typed contract for component outputs so the
        orchestration layer (Figure) and layout layer (FigureFormatter)
        can make safe decisions about legends, colorbars, and labels.

    Contents:
      - artists: mapping of semantic keys (e.g., 'mesh', 'collection', 'line')
                 to matplotlib artist objects or other renderables.
      - metadata: optional dictionary for extra information about the render
                  (e.g., {'mappable': quadmesh, 'label': '$\rho$', 'is_line': True}).

    Best practices:
      - prefer returning a `mappable` in metadata when the component creates a
        mesh/collection intended for a colorbar. this avoids idiosyncratic
        guesses by the formatter.
      - include a 'label' metadata when a component represents a single
        conceptual dataset (useful for axis labels and legends).
      - keep metadata advisory: the formatter may ignore keys it doesn't
        understand; avoid relying on side-effects.
    """

    artists: dict[str, object]
    metadata: Optional[dict[str, object]] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True,
    }


class PlotData(BaseModel):
    """
    Complete data needed for visualization.

    Attributes:
        fields: Sequence of FieldData objects to visualize
        bodies: optional mapping of body name -> Body metadata
        time: simulation time (float)
        dimensions: Number of spatial dimensions in the simulation
        coord_system: Coordinate system used
        hierarchy: optional AMR hierarchy data (for refinement-aware components)
    """

    fields: Sequence[FieldData]
    # BodyCollection - avoid circular import
    body_collection: Optional[object] = None
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

    Attributes:
        min: Minimum value
        max: Maximum value
    """

    min: float | None
    max: float | None
    model_config = {
        "frozen": True,  # Make instances immutable
    }

    @field_validator("max")
    @classmethod
    def max_greater_than_min(cls, v: float | None, info) -> float | None:
        """Validate that max is greater than min."""
        if v is None:
            return None

        values = info.data
        if "min" in values and v <= values["min"]:
            raise ValueError(
                f"Max value {v} must be greater than min value {values['min']}"
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
