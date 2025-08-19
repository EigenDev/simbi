"""Multidimensional plot component for visualization."""

from typing import Optional, Literal
import numpy as np
import matplotlib.colors as mcolors
from pydantic import field_validator, ValidationInfo
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh

from ..core.types import PlotData, FieldData, Array, ColorRange
from .interface import ComponentProps

# ---- Pure transformation functions ----


def extract_mesh_coordinates(field: FieldData) -> tuple[Array, Array]:
    """Extract mesh coordinates from field domain."""
    if len(field.domain) < 2:
        # Handle 1D case by creating a dummy y coordinate
        x = field.domain[0]
        y = np.array([0, 1])
        return x, y

    return field.domain[0], field.domain[1]


def prepare_mesh_data(field: FieldData) -> tuple[Array, Array, Array]:
    """
    Prepare data for mesh visualization.

    Returns:
        Tuple of (X coordinates, Y coordinates, Z values)
    """
    # Extract coordinates
    x, y = extract_mesh_coordinates(field)

    # Get values
    values = field.values

    # Handle dimensionality
    if values.ndim == 1 and x.size == values.size:
        # 1D values along x, expand to 2D
        values = np.tile(values, (2, 1)).T
    elif values.ndim > 2:
        # Take first slice of higher-dim data
        values = values[:, :, 0] if values.ndim > 2 else values

    # Ensure proper orientation
    if x.ndim == 1 and y.ndim == 1:
        # Create 2D mesh grid if inputs are 1D
        X, Y = np.meshgrid(x, y)
    else:
        # Use as provided if already 2D
        X, Y = x, y

    return X, Y, values


def create_color_normalization(
    values: Array,
    color_range: Optional[ColorRange],
    log_scale: bool = False,
    power: float = 1.0,
    linear_fields: list[str] | None = None,
) -> mcolors.Normalize:
    """Create color normalization based on data and settings."""
    # Get min/max from data if not provided
    vmin = color_range.min if color_range else np.nanmin(values)
    vmax = color_range.max if color_range else np.nanmax(values)

    # Handle uniform data
    if np.allclose(vmin, vmax, rtol=1e-10):
        eps = max(float(abs(vmin) * 1e-2), 0.1)
        vmin -= eps
        vmax += eps

    # Create appropriate normalization
    if log_scale:
        # Ensure positive values for log scale
        if vmin <= 0:
            # Find smallest positive value
            pos_min = np.nanmin(values[values > 0]) if np.any(values > 0) else 1e-10
            vmin = pos_min * 0.9

        return mcolors.LogNorm(vmin=float(vmin), vmax=float(vmax))
    else:
        # Use power norm with configurable gamma
        return mcolors.PowerNorm(gamma=power, vmin=float(vmin), vmax=float(vmax))


# ---- Component class ----


class MultidimPlotProps(ComponentProps):
    """Properties for multidimensional plot component."""

    field_index: int = 0
    cmap: str = "viridis"
    color_range: Optional[ColorRange] = None
    log_scale: bool = False
    power: float = 1.0
    shading: Literal["auto", "nearest", "gouraud", "flat"] = "auto"
    alpha: float = 1.0

    @field_validator("field_index")
    @classmethod
    def validate_field_index(cls, v: int, info: ValidationInfo) -> int:
        """Validate that field index is non-negative."""
        if v < 0:
            raise ValueError(f"Field index must be non-negative, got {v}")
        return v

    @field_validator("power")
    @classmethod
    def validate_power(cls, v: float, info: ValidationInfo) -> float:
        """Validate that power is positive."""
        if v <= 0:
            raise ValueError(f"Power must be positive, got {v}")
        return v

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float, info: ValidationInfo) -> float:
        """Validate that alpha is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {v}")
        return v

    @field_validator("shading")
    @classmethod
    def validate_shading(cls, v: str, info: ValidationInfo) -> str:
        """Validate shading option."""
        valid_options = ["auto", "flat", "gouraud", "nearest"]
        if v not in valid_options:
            raise ValueError(f"Shading must be one of {valid_options}, got {v}")
        return v


class MultidimPlotComponent:
    """Multidimensional plot visualization component."""

    def __init__(self, props: MultidimPlotProps):
        """Initialize the multidimensional plot component."""
        self.props = props
        self._mesh: Optional[QuadMesh] = None
        self._initialized: bool = False
        self._linear_fields = [
            "velocity",
            "momentum",
            "gamma_beta",
        ]  # Fields that shouldn't use log scale

    def initialize(self, fig: Figure, ax: Axes) -> None:
        """Initialize the component with figure and axes."""
        self.fig = fig
        self.ax = ax
        self._initialized = True

    def update(self, props: MultidimPlotProps) -> None:
        """Update component properties."""
        prev_props = self.props
        self.props = props

        # Update mesh if it exists
        if self._mesh and hasattr(self, "ax"):
            # Update colormap if changed
            if prev_props.cmap != props.cmap:
                self._mesh.set_cmap(props.cmap)

            # Update alpha if changed
            if prev_props.alpha != props.alpha:
                self._mesh.set_alpha(props.alpha)

    def render(self, data: PlotData) -> Optional[QuadMesh]:
        """Render the multidimensional plot with data."""
        if not self._initialized or not hasattr(self, "ax"):
            raise RuntimeError("Component not initialized. Call initialize() first.")

        # Skip if field index is out of range
        if self.props.field_index >= len(data.fields):
            return self._mesh

        # Get field data
        field = data.fields[self.props.field_index]

        # Prepare mesh data
        X, Y, values = prepare_mesh_data(field)

        # Create or update mesh
        if self._mesh is None:
            # Create new mesh
            self._mesh = self.ax.pcolormesh(
                X,
                Y,
                values,
                cmap=self.props.cmap,
                shading=self.props.shading,
                alpha=self.props.alpha,
            )
        else:
            # Update existing mesh
            self._update_mesh(X, Y, values)

        # Apply color normalization
        norm = create_color_normalization(
            values,
            self.props.color_range,
            self.props.log_scale,
            self.props.power,
            self._linear_fields,
        )
        self._mesh.set_norm(norm)

        return self._mesh

    def _update_mesh(self, X: Array, Y: Array, values: Array) -> None:
        """Update existing mesh with new data."""
        if self._mesh is None:
            raise RuntimeError("Mesh is not initialized. Call render() first.")

        # Check if the mesh coordinates have changed
        # if X.shape != shape_x or Y.shape != shape_y:
        #     # If mesh shape changed, recreate the mesh
        #     if self._mesh in self.ax.collections:
        #         self._mesh.remove()

        #     self._mesh = self.ax.pcolormesh(
        #         X,
        #         Y,
        #         values,
        #         cmap=self.props.cmap,
        #         shading=self.props.shading,
        #         alpha=self.props.alpha,
        #     )
        # else:
        # Just update the values if coordinates are the same
        self._mesh.set_array(values.ravel())

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, "ax") and self._mesh is not None:
            if self._mesh in self.ax.collections:
                self._mesh.remove()
            self._mesh = None
