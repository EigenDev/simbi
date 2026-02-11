"""
Quad plot component for visualization.

Renders a single 2D FieldData object as a pcolormesh.
"""

from typing import Literal, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure

from simbi.reader.io import BodyCollection

from ..config import FigureConfig
from ..types import Array, FieldData, RenderResult
from .interface import Component
from .shared import ColormappedProps, create_color_normalization, draw_bodies


class QuadPlotProps(ColormappedProps):
    """Properties for a quad (pcolormesh) plot component."""

    shading: Literal["auto", "nearest", "gouraud", "flat"] = "auto"
    plot_type: Literal["polar", "cartesian"] = "cartesian"


class QuadPlotComponent(Component):
    """
    A simple renderer for a single 2D field.
    Expects 2D FieldData.
    """

    def __init__(
        self, props: QuadPlotProps, bodies: Optional[BodyCollection] = None
    ):
        self.props = props
        self._mesh: Optional[QuadMesh] = None
        self._mirror_mesh: Optional[QuadMesh] = None  # For polar plots
        self._initialized: bool = False
        self._first_render: bool = True
        self.last_x = np.array([])
        self.last_y = np.array([])
        self._mesh_lines: list = []
        self.bodies: Optional[BodyCollection] = bodies

    def initialize(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax = ax
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, props: QuadPlotProps) -> None:
        """Update component properties and restyle the mesh if it exists."""
        self.props = props
        if self._mesh and self._initialized:
            self._mesh.set_cmap(props.cmap)
            self._mesh.set_alpha(props.alpha)
            # Other style updates can go here

        # Handle mesh grid toggle
        if self.props.show_mesh_grid:
            self._draw_mesh_grid(self.last_x, self.last_y)
        else:
            self._clear_mesh_grid()

    def render(self, data: FieldData, style: FigureConfig) -> RenderResult:
        """
        Render the Quadensional plot with guaranteed 2D data.
        `data` is a *single* FieldData object.

        Returns:
            RenderResult containing the created matplotlib artists and optional metadata.
        """
        if not self._initialized:
            raise RuntimeError(
                "Component not initialized. Call initialize() first."
            )

        if data.ndim != 2:
            raise ValueError(
                f"QuadPlotComponent received data with ndim={data.ndim}."
                " It can only render 2D FieldData."
            )

        if self.ax.name == "polar":
            # (r, theta) -> (theta, r) for polar plot
            x, y = data.domain[0], data.domain[1]
            values = data.values.T
        else:
            x, y = data.domain[1], data.domain[0]
            values = data.values

        # compute domain bounds for axis limits
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        norm = create_color_normalization(
            values,
            self.props.color_range,
            self.props.log_scale,
            self.props.power,
        )

        if self._mesh is None:
            self._mesh = self.ax.pcolormesh(
                x,
                y,
                values,
                cmap=self.props.cmap,
                shading=self.props.shading,
                alpha=self.props.alpha,
                norm=norm,
            )
        else:
            self._update_mesh(x, y, values)
            self._mesh.set_norm(norm)  # Re-apply norm for animations

        if self.props.show_mesh_grid:
            self._draw_mesh_grid(x, y)

        if style.draw_bodies and self.bodies:
            self.draw_bodies(
                self.bodies,
                zorder=10,
                axes=data.axis_names if data.axis_names else ["x1", "x2"],
            )

        self.last_x = x
        self.last_y = y

        # set limits only on first render (preserves CLI limits and user zoom)
        if self._first_render:
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self._first_render = False

        self.ax.set_aspect("equal", adjustable="box")

        return RenderResult(
            artists={"mesh": self._mesh}, metadata={"mappable": self._mesh}
        )

    def _update_mesh(self, x: Array, y: Array, values: Array) -> None:
        """Update existing mesh with new data (for animation)."""
        if self._mesh is None:
            raise RuntimeError("Mesh is not initialized. Call render() first.")

        # Check if coordinates have changed
        if not np.allclose(x, self.last_x) or not np.allclose(y, self.last_y):
            # Coordinates changed: must remove and re-create mesh
            if self._mesh in self.ax.collections:
                self._mesh.remove()
            if self._mirror_mesh and self._mirror_mesh in self.ax.collections:
                self._mirror_mesh.remove()

            self._mesh = self.ax.pcolormesh(
                x,
                y,
                values,
                cmap=self.props.cmap,
                shading=self.props.shading,
                alpha=self.props.alpha,
            )
            # Handle polar mirror (if needed)
            if self.ax.name == "polar":
                self._mirror_mesh = self.ax.pcolormesh(
                    -x[::-1],
                    y,
                    values,  # Example mirror logic
                    cmap=self.props.cmap,
                    shading=self.props.shading,
                    alpha=self.props.alpha,
                )
            self.last_x = x
            self.last_y = y
        else:
            # Just update values
            self._mesh.set_array(values.ravel())
            if self._mirror_mesh:
                self._mirror_mesh.set_array(values.ravel())

    def draw_bodies(
        self, body_collection: BodyCollection, zorder: int, axes: Sequence[str]
    ) -> None:
        """Draw immersed bodies on the plot."""
        draw_bodies(self.ax, body_collection, zorder, axes)

    def _draw_mesh_grid(self, x: Array, y: Array) -> None:
        """Draw cell boundaries on the mesh."""
        self._clear_mesh_grid()
        for xi in x:
            line = self.ax.axvline(
                xi,
                color=self.props.mesh_color,
                alpha=self.props.mesh_alpha,
                linewidth=self.props.mesh_linewidth,
                zorder=5,
            )
            self._mesh_lines.append(line)
        for yi in y:
            line = self.ax.axhline(
                yi,
                color=self.props.mesh_color,
                alpha=self.props.mesh_alpha,
                linewidth=self.props.mesh_linewidth,
                zorder=5,
            )
            self._mesh_lines.append(line)

    def _clear_mesh_grid(self) -> None:
        """Remove all mesh grid lines."""
        for line in self._mesh_lines:
            line.remove()
        self._mesh_lines.clear()

    def cleanup(self) -> None:
        """Clean up resources."""
        self._clear_mesh_grid()
        if self._mesh and self._mesh in self.ax.collections:
            self._mesh.remove()
        if self._mirror_mesh and self._mirror_mesh in self.ax.collections:
            self._mirror_mesh.remove()
        self._mesh = None
        self._mirror_mesh = None
