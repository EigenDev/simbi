"""
Quadensional plot component for visualization.

This component is a simple renderer. It expects to be given
a single, 2D FieldData object and will render it as a pcolormesh.
"""

from typing import Literal, Optional

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.figure import Figure
from pydantic import ValidationInfo, field_validator


from ..config import FigureConfig
from ..types import Array, ColorRange, FieldData, RenderResult
from .interface import Component, ComponentProps
from .mesh_overlay import DEFAULT_MAX_LINES, mesh_segments


class QuadPlotProps(ComponentProps):
    """Properties for a *single* Quadensional plot component."""

    cmap: str = "viridis"
    color_range: ColorRange = ColorRange(min=None, max=None)
    log_scale: bool = False
    power: float = 1.0
    shading: Literal["auto", "nearest", "gouraud", "flat"] = "auto"
    alpha: float = 1.0
    plot_type: Literal["polar", "cartesian"] = "cartesian"

    # cell-edge overlay (optional)
    show_mesh_grid: bool = False
    mesh_color: str = "white"
    mesh_alpha: float = 0.3
    mesh_linewidth: float = 0.1
    # coordinate lines drawn per axis: 0 decimates a fine mesh to a readable
    # count, 1 draws every cell edge
    mesh_stride: int = 0
    mesh_max_lines: int = DEFAULT_MAX_LINES

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

    @field_validator("mesh_stride")
    @classmethod
    def validate_mesh_stride(cls, v: int, _: ValidationInfo) -> int:
        if v < 0:
            raise ValueError(f"mesh_stride must be non-negative, got {v}")
        return v

    @field_validator("mesh_max_lines")
    @classmethod
    def validate_mesh_max_lines(cls, v: int, _: ValidationInfo) -> int:
        if v < 2:
            raise ValueError(
                f"mesh_max_lines must leave room for both domain edges, got {v}"
            )
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
    A simple renderer for a single 2D field.
    Expects 2D FieldData.
    """

    def __init__(self, props: QuadPlotProps):
        self.props = props
        self._mesh: Optional[QuadMesh] = None
        self._initialized: bool = False
        self.last_x = np.array([])
        self.last_y = np.array([])
        self._mesh_edges: Optional[LineCollection] = None

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
            self._mesh.set_cmap(self._resolve_cmap())
            self._mesh.set_alpha(props.alpha)

        # handle mesh grid toggle
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

        norm = _create_color_normalization(
            values,
            self.props.color_range,
            self.props.log_scale,
            self.props.power,
        )

        # resolve the cmap spec against the norm, so a `join:`/`stack:` composite -- and a
        # `@DATA` split within it -- can be given inline in props.cmap (see colormaps.resolve_cmap).
        self._last_norm = norm
        cmap = self._resolve_cmap(norm)

        if self._mesh is None:
            self._mesh = self.ax.pcolormesh(
                x,
                y,
                values,
                cmap=cmap,
                shading=self.props.shading,
                alpha=self.props.alpha,
                norm=norm,
            )
        else:
            self._update_mesh(x, y, values)
            self._mesh.set_norm(norm)  # Re-apply norm for animations

        if self.props.show_mesh_grid:
            self._draw_mesh_grid(x, y)
        else:
            self._clear_mesh_grid()

        self.last_x = x
        self.last_y = y

        self.ax.set_aspect("equal", adjustable="box")

        # the axes is shared, and the extent drawn on it moves with a moving
        # mesh, so the view is the figure's to compose from what each component
        # reports each frame
        return RenderResult(
            artists={"mesh": self._mesh},
            mappable=self._mesh,
            colorbar_label=data.name,
            view_bounds=(x_min, x_max, y_min, y_max),
        )

    def _resolve_cmap(self, norm=None):
        """resolve props.cmap (a plain name or a `join:`/`stack:` composite spec) to a
        Colormap, against the given norm or the last one seen. see colormaps.resolve_cmap."""
        from ..colormaps import resolve_cmap

        return resolve_cmap(
            self.props.cmap,
            norm=norm if norm is not None else getattr(self, "_last_norm", None),
        )

    def _update_mesh(self, x: Array, y: Array, values: Array) -> None:
        """Update existing mesh with new data (for animation)."""
        if self._mesh is None:
            raise RuntimeError("Mesh is not initialized. Call render() first.")

        if self._coordinates_moved(x, y):
            # a QuadMesh owns its vertices, so a mesh whose vertices moved --
            # a homologous expansion, a shock-following mesh law -- has to be
            # rebuilt rather than refilled
            if self._mesh in self.ax.collections:
                self._mesh.remove()

            self._mesh = self.ax.pcolormesh(
                x,
                y,
                values,
                cmap=self._resolve_cmap(),
                shading=self.props.shading,
                alpha=self.props.alpha,
            )
            self.last_x = x
            self.last_y = y
        else:
            # just update values
            self._mesh.set_array(values.ravel())

    def _coordinates_moved(self, x: Array, y: Array) -> bool:
        """whether the vertex arrays differ from the ones the mesh was built on.

        a change of length is a change of mesh, and comparing values across
        two different lengths is not defined, so shape is tested first."""
        for new, old in ((x, self.last_x), (y, self.last_y)):
            if np.shape(new) != np.shape(old):
                return True
            if not np.allclose(new, old):
                return True
        return False

    def _draw_mesh_grid(self, x: Array, y: Array) -> None:
        """Draw the mesh cell edges over the field.

        the edges come from this frame's vertex arrays, so the overlay tracks a
        mesh that moves between checkpoints. they are one collection: a fine
        mesh drawn as one artist per line costs thousands of artists per frame.
        """
        segments = mesh_segments(
            x,
            y,
            curved=self.ax.name == "polar",
            stride=self.props.mesh_stride,
            max_lines=self.props.mesh_max_lines,
        )
        if not segments:
            self._clear_mesh_grid()
            return

        if self._mesh_edges is None:
            # above the field: the QuadMesh is rebuilt whenever the vertices
            # move, which re-adds it after the overlay in the collection list
            self._mesh_edges = LineCollection(
                segments,
                colors=self.props.mesh_color,
                linewidths=self.props.mesh_linewidth,
                alpha=self.props.mesh_alpha,
                zorder=5,
            )
            self.ax.add_collection(self._mesh_edges)
        else:
            self._mesh_edges.set_segments(segments)
            self._mesh_edges.set_color(self.props.mesh_color)
            self._mesh_edges.set_linewidth(self.props.mesh_linewidth)
            self._mesh_edges.set_alpha(self.props.mesh_alpha)

    def _clear_mesh_grid(self) -> None:
        """Remove the cell-edge overlay."""
        if self._mesh_edges is not None:
            self._mesh_edges.remove()
            self._mesh_edges = None

    def cleanup(self) -> None:
        """Clean up resources."""
        self._clear_mesh_grid()
        if self._mesh and self._mesh in self.ax.collections:
            self._mesh.remove()
        self._mesh = None
