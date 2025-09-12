"""Multidimensional plot component for visualization."""

from typing import Literal, Optional

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from pydantic import ValidationInfo, field_validator

from simbi.core.types.bodies import Body
from simbi.tools.visualization.formatters.multidim import (
    format_multidim_plot_axes,
)

from ..core.config import StyleConfig
from ..core.types import Array, ColorRange, PlotData
from .interface import Component, ComponentProps


def create_color_normalization(
    values: Array,
    color_range: ColorRange,
    log_scale: bool = False,
    power: float = 1.0,
    linear_fields: list[str] | None = None,
) -> mcolors.Normalize:
    """Create color normalization based on data and settings."""
    # Get min/max from data if not provided
    vmin = color_range.min or np.nanmin(values)
    vmax = color_range.max or np.nanmax(values)

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
            pos_min = (
                np.nanmin(values[values > 0]) if np.any(values > 0) else 1e-10
            )
            vmin = pos_min * 0.9

        return mcolors.LogNorm(vmin=float(vmin), vmax=float(vmax))
    else:
        # Use power norm with configurable gamma
        return mcolors.PowerNorm(
            gamma=power, vmin=float(vmin), vmax=float(vmax)
        )


class MultidimPlotProps(ComponentProps):
    """Properties for multidimensional plot component."""

    cmap: str = "viridis"
    color_range: ColorRange = ColorRange(min=None, max=None)
    field_index: int = 0
    log_scale: bool = False
    power: float = 1.0
    shading: Literal["auto", "nearest", "gouraud", "flat"] = "auto"
    alpha: float = 1.0
    projection: tuple[int, int, int] = (1, 2, 3)
    plot_type: Literal["polar", "cartesian"] = "cartesian"

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


class MultidimPlotComponent(Component):
    """Multidimensional plot visualization component."""

    def __init__(self, props: MultidimPlotProps):
        """Initialize the multidimensional plot component."""
        self.props = props
        self._mesh: Optional[QuadMesh] = None
        self._mirror_mesh: Optional[QuadMesh] = None
        self._initialized: bool = False
        self._linear_fields = [
            "velocity",
            "momentum",
            "gamma_beta",
        ]  # Fields that shouldn't use log scale
        self.last_x = np.array([])
        self.last_y = np.array([])

    def initialize(self, fig: Figure, ax: Axes) -> None:
        """Initialize the component with figure and axes."""
        self.fig = fig
        self.ax = ax
        self._initialized = True

    @property
    def initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized

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

    def draw_bodies(
        self, bodies: dict[str, Body], proj: tuple[int, ...]
    ) -> None:
        # Clear any existing patches
        for patch in self.ax.patches:
            patch.remove()

        import matplotlib.patches as mpatches

        # Draw each immersed body
        for _, body in bodies.items():
            if body.accretion is not None:
                radius = body.accretion.accretion_radius
            else:
                radius = body.radius

            # if we're in 3D, then we must be
            # mindful to plot the circles projected
            # along the 2D plane. If the projection
            # is (1,2,3), then we are plotting in the
            # x1-x2 plane, and the z coordinate
            # is the third coordinate. If we are plotting
            # in the x2-x3 plane, then we need to
            # onto the x1. The body position is
            # always in cartesian coordinates.
            if proj == (1, 2, 3):
                projected_position = (
                    body.position[0],
                    body.position[1],
                )
            elif proj == (2, 3, 1):
                projected_position = (
                    body.position[1],
                    body.position[2],
                )
            else:
                projected_position = (
                    body.position[0],
                    body.position[2],
                )

            # Create circle
            circle = mpatches.Circle(
                projected_position,
                radius,
                color="black",
                linestyle="--",
                alpha=0.5,
            )

            # Add to plot
            self.ax.add_patch(circle)

    def render(self, data: PlotData, style: StyleConfig) -> Optional[QuadMesh]:
        """Render the multidimensional plot with data."""
        if not self._initialized or not hasattr(self, "ax"):
            raise RuntimeError(
                "Component not initialized. Call initialize() first."
            )

        # Get field data
        field = data.fields[self.props.field_index]

        # Prepare mesh data
        x, y = field.domain
        cmap = self.props.cmap
        crange = self.props.color_range

        values = field.values
        if self.props.plot_type == "polar":
            x, y = y, x
            values = field.values.T

        # if data.bodies:
        # self.draw_bodies(data.bodies, self.props.projection)
        # Create or update mesh
        if self._mesh is None:
            # Create new mesh
            self._mesh = self.ax.pcolormesh(
                x,
                y,
                values,
                cmap=cmap,
                shading=self.props.shading,
                alpha=self.props.alpha,
            )

            if self.ax.name == "polar":
                self._mirror_mesh = self.ax.pcolormesh(
                    -x[::-1],
                    y,
                    values,
                    cmap=cmap,
                    shading=self.props.shading,
                    alpha=self.props.alpha,
                )
        else:
            # Update existing mesh
            self._update_mesh(x, y, values)

        # Apply color normalization
        norm = create_color_normalization(
            field.values,
            crange,
            self.props.log_scale,
            self.props.power,
            self._linear_fields,
        )
        self._mesh.set_norm(norm)
        if self._mirror_mesh:
            self._mirror_mesh.set_norm(norm)

        format_multidim_plot_axes(
            self.ax,
            self.fig,
            self._mesh,
            data,
            self.props.field_index,
            style,
            self.props.plot_type == "polar",
        )
        self.last_x = x
        self.last_y = y

        return self._mesh

    def _update_mesh(self, x: Array, y: Array, values: Array) -> None:
        """Update existing mesh with new data."""
        if self._mesh is None:
            raise RuntimeError("Mesh is not initialized. Call render() first.")

        if not np.allclose(x, self.last_x) or not np.allclose(y, self.last_y):
            # If the coords changed, create a brand new mesh
            if self._mesh in self.ax.collections:
                self._mesh.remove()

            self._mesh = self.ax.pcolormesh(
                x,
                y,
                values,
                cmap=self.props.cmap,
                shading=self.props.shading,
                alpha=self.props.alpha,
            )
            if self.ax.name == "polar":
                self._mirror_mesh = self.ax.pcolormesh(
                    -x[::-1],
                    y,
                    values,
                    cmap=self.props.cmap,
                    shading=self.props.shading,
                    alpha=self.props.alpha,
                )

            self.last_x = x
            self.last_y = y
        else:
            # Just update the values if coordinates are the same
            self._mesh.set_array(values.ravel())
            if self._mirror_mesh:
                self._mirror_mesh.set_array(values.ravel())

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, "ax") and self._mesh is not None:
            if self._mesh in self.ax.collections:
                self._mesh.remove()
            self._mesh = None
