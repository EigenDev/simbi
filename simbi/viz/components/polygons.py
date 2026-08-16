"""
Polygon plot component for visualization.

This component is a simple renderer. It expects a single PolygonData: one
quadrilateral per cell with one value each, which is how a level hierarchy is
drawn when a single quadmesh cannot hold cells of two different sizes.
"""

from typing import Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from pydantic import ValidationInfo, field_validator


from ..config import FigureConfig
from ..types import Array, ColorRange, PolygonData, RenderResult
from .interface import Component, ComponentProps
from .mesh_overlay import mesh_segments
from .quad import _create_color_normalization


class PolygonPlotProps(ComponentProps):
    """Properties for a *single* polygon plot component."""

    cmap: str = "viridis"
    color_range: ColorRange = ColorRange(min=None, max=None)
    log_scale: bool = False
    power: float = 1.0
    alpha: float = 1.0

    # mesh visualization (optional)
    show_mesh_grid: bool = False
    mesh_color: str = "white"
    mesh_alpha: float = 0.3
    mesh_linewidth: float = 0.1

    # level bounds visualization (optional)
    show_level_bounds: bool = False
    level_color: str = "white"
    level_linewidth: float = 1.5
    level_alpha: float = 0.8

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


class PolygonPlotComponent(Component):
    """
    A simple renderer for 2D refined data as polygons.
    Expects 1D FieldData adhering to the "Polygon Contract".
    """

    def __init__(self, props: PolygonPlotProps):
        self.props = props
        self._poly_collection: Optional[PolyCollection] = None
        self._level_edges: Optional[LineCollection] = None
        self._initialized: bool = False

    def initialize(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax = ax
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, props: PolygonPlotProps) -> None:
        """Update component properties and restyle the collection."""
        self.props = props
        if self._poly_collection and self._initialized:
            self._poly_collection.set_cmap(props.cmap)
            self._poly_collection.set_alpha(props.alpha)
            # update edge colors based on mesh grid toggle
            edge_color = (
                self.props.mesh_color if self.props.show_mesh_grid else "none"
            )
            edge_width = (
                self.props.mesh_linewidth if self.props.show_mesh_grid else 0
            )
            self._poly_collection.set_edgecolors(edge_color)
            self._poly_collection.set_linewidths(edge_width)

    def render(self, data: PolygonData, style: FigureConfig) -> RenderResult:
        """
        Render one cell per polygon.
        `data` is a *single* PolygonData object.
        """
        if not self._initialized:
            raise RuntimeError(
                "Component not initialized. Call initialize() first."
            )

        if not isinstance(data, PolygonData):
            raise TypeError(
                "PolygonPlotComponent draws PolygonData (independent cells);"
                f" got {type(data).__name__}"
            )

        patches = self._to_axes_coordinates(np.asarray(data.patches, dtype=float))
        values = data.values

        # compute domain bounds for setting axis limits
        x_min, x_max = patches[..., 0].min(), patches[..., 0].max()
        y_min, y_max = patches[..., 1].min(), patches[..., 1].max()

        # create color normalization
        norm = _create_color_normalization(
            values,
            self.props.color_range,
            self.props.log_scale,
            self.props.power,
        )

        if self._poly_collection is None:
            edge_color = (
                self.props.mesh_color if self.props.show_mesh_grid else "none"
            )
            edge_width = (
                self.props.mesh_linewidth if self.props.show_mesh_grid else 0
            )

            # create the new PolyCollection
            self._poly_collection = PolyCollection(
                patches,
                array=values,
                cmap=self.props.cmap,
                norm=norm,
                edgecolors=edge_color,
                linewidths=edge_width,
                alpha=self.props.alpha,
            )
            self.ax.add_collection(self._poly_collection)
        else:
            # update existing collection
            self._poly_collection.set_verts(patches)
            self._poly_collection.set_array(values)
            self._poly_collection.set_norm(norm)

        self.ax.set_aspect("equal", adjustable="box")

        if self.props.show_level_bounds and data.level_bounds:
            self._draw_level_bounds(data.level_bounds)
        else:
            self._clear_level_bounds()

        # the axes is shared, and the cells of a moving mesh sit at new
        # positions in each checkpoint, so the view is the figure's to compose
        # from what each component reports each frame
        return RenderResult(
            artists={"collection": self._poly_collection},
            mappable=self._poly_collection,
            colorbar_label=data.name,
            view_bounds=(x_min, x_max, y_min, y_max),
        )

    def _to_axes_coordinates(self, vertices: Array) -> Array:
        """map vertices from mesh order (x1, x2) to the order the axes draws in.

        a polar axes reads a vertex as (angle, radius), which is the reverse of
        the (radius, angle) order a spherical mesh stores its axes in."""
        if self.ax.name == "polar":
            return vertices[..., ::-1]
        return vertices

    def _draw_level_bounds(
        self, level_bounds: Sequence[tuple[float, float, float, float]]
    ) -> None:
        """Outline each refined level's bounding box.

        the box follows the chart: on a polar axes its constant-radius sides
        are drawn as arcs, so they hug the wedge along its true boundary."""
        polar = self.ax.name == "polar"

        segments: list[Array] = []
        # level 0 spans the whole domain, so its outline is the plot frame
        for bounds in level_bounds[1:]:
            x0, x1, y0, y1 = bounds
            if polar:
                # the bounds span (x1, x2) = (radius, angle); a polar axes
                # plots angle horizontally
                x0, x1, y0, y1 = y0, y1, x0, x1
            segments += mesh_segments(
                [x0, x1], [y0, y1], curved=polar, stride=1
            )

        if not segments:
            self._clear_level_bounds()
            return

        if self._level_edges is None:
            self._level_edges = LineCollection(
                segments,
                colors=self.props.level_color,
                linewidths=self.props.level_linewidth,
                alpha=self.props.level_alpha,
                zorder=5,
            )
            self.ax.add_collection(self._level_edges)
        else:
            self._level_edges.set_segments(segments)

    def _clear_level_bounds(self) -> None:
        if self._level_edges is not None:
            self._level_edges.remove()
            self._level_edges = None

    def cleanup(self) -> None:
        if self._poly_collection:
            self._poly_collection.remove()
        self._poly_collection = None
        self._clear_level_bounds()
