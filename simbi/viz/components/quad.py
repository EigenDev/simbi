"""
Quadensional plot component for visualization.

This component is a "simple" renderer. It expects to be given
a single, 2D FieldData object and will render it as a pcolormesh.
"""

from typing import Literal, Optional, Sequence

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from pydantic import ValidationInfo, field_validator

from simbi.types.bodies import Body

from ..config import StyleConfig
from ..types import Array, ColorRange, FieldData, RenderResult
from .interface import Component, ComponentProps

LOGICAL_AXIS_MAP = {"x1": 0, "x2": 1, "x3": 2}


class QuadPlotProps(ComponentProps):
    """Properties for a *single* Quadensional plot component."""

    cmap: str = "viridis"
    color_range: ColorRange = ColorRange(min=None, max=None)
    log_scale: bool = False
    power: float = 1.0
    shading: Literal["auto", "nearest", "gouraud", "flat"] = "auto"
    alpha: float = 1.0
    plot_type: Literal["polar", "cartesian"] = "cartesian"

    # Mesh visualization (optional)
    show_mesh_grid: bool = False
    mesh_color: str = "white"
    mesh_alpha: float = 0.3
    mesh_linewidth: float = 0.1

    @field_validator("power")
    @classmethod
    def validate_power(cls, v: float, _: ValidationInfo) -> float:
        if v <= 0:
            raise ValueError(f"Power must be positive, got {v}")
        return v

    @field_validator("alpha", "mesh_alpha")
    @classmethod
    def validate_alpha(cls, v: float, _: ValidationInfo) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {v}")
        return v


def _create_color_normalization(
    values: Array,
    color_range: ColorRange,
    log_scale: bool = False,
    power: float = 1.0,
) -> mcolors.Normalize:
    """Create color normalization based on data and settings."""
    vmin = color_range.min if color_range.min is not None else np.nanmin(values)
    vmax = color_range.max if color_range.max is not None else np.nanmax(values)

    if np.allclose(vmin, vmax, rtol=1e-10):
        eps = max(float(abs(vmin) * 1e-2), 0.1)
        vmin -= eps
        vmax += eps

    if log_scale:
        if vmin <= 0:
            pos_min = (
                np.nanmin(values[values > 0]) if np.any(values > 0) else 1e-10
            )
            vmin = pos_min * 0.9
        return mcolors.LogNorm(vmin=float(vmin), vmax=float(vmax))
    else:
        return mcolors.PowerNorm(
            gamma=power, vmin=float(vmin), vmax=float(vmax)
        )


class QuadPlotComponent(Component):
    """
    A "simple" renderer for a single 2D field.
    Expects 2D FieldData.
    """

    def __init__(
        self, props: QuadPlotProps, bodies: Optional[dict[str, Body]] = None
    ):
        self.props = props
        self._mesh: Optional[QuadMesh] = None
        self._mirror_mesh: Optional[QuadMesh] = None  # For polar plots
        self._initialized: bool = False
        self.last_x = np.array([])
        self.last_y = np.array([])
        self._mesh_lines: list = []
        self.bodies: Optional[dict[str, Body]] = bodies

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

    def render(self, data: FieldData, style: StyleConfig) -> RenderResult:
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

        if self.props.plot_type == "polar":
            # (r, theta) -> (theta, r) for polar plot
            x, y = data.domain[0], data.domain[1]
            values = data.values.T
        else:
            x, y = data.domain[1], data.domain[0]
            values = data.values

        norm = _create_color_normalization(
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
        self, bodies: dict[str, Body], zorder: int, axes: Sequence[str]
    ) -> None:
        """Draw immersed bodies on the plot."""
        # This function can be simplified as it no longer
        # needs projection logic (that's handled by the slice)
        import matplotlib.patches as mpatches

        for patch in self.ax.patches:
            patch.remove()
        n_i = LOGICAL_AXIS_MAP[axes[0]]
        n_j = LOGICAL_AXIS_MAP[axes[1]]

        for _, body in bodies.items():
            radius = body.radius
            if body.accretion is not None:
                radius = body.accretion.accretion_radius
            position = (body.position[n_i], body.position[n_j])  # Assumes 2D

            circle = mpatches.Circle(
                position,
                radius,
                color="black",
                linestyle="--",
                alpha=0.5,
                zorder=zorder,
            )
            self.ax.add_patch(circle)

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
