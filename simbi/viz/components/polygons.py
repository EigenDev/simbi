"""
Polygon plot component for visualization.

Renders 2D AMR data as a PolyCollection. Expects 1D FieldData
where domain is a list of patches and values are cell scalars.
"""

from typing import Optional, Sequence

from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from simbi.reader.io import BodyCollection

from ..config import FigureConfig
from ..types import FieldData, RenderResult
from .interface import Component
from .shared import ColormappedProps, create_color_normalization, draw_bodies


class PolygonPlotProps(ColormappedProps):
    """Properties for a polygon (AMR) plot component."""

    show_level_bounds: bool = False
    level_color: str = "white"
    level_linewidth: float = 1.5
    level_alpha: float = 0.8


class PolygonPlotComponent(Component):
    """
    A simple renderer for 2D refined data as polygons.
    Expects 1D FieldData adhering to the "Polygon Contract".
    """

    def __init__(
        self, props: PolygonPlotProps, bodies: Optional[BodyCollection] = None
    ):
        self.props = props
        self._poly_collection: Optional[PolyCollection] = None
        self._level_artists: list = []
        self._initialized: bool = False
        self._first_render: bool = True
        self.bodies: Optional[BodyCollection] = bodies

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
            # Update edge colors based on mesh grid toggle
            edge_color = (
                self.props.mesh_color if self.props.show_mesh_grid else "none"
            )
            edge_width = (
                self.props.mesh_linewidth if self.props.show_mesh_grid else 0
            )
            self._poly_collection.set_edgecolors(edge_color)
            self._poly_collection.set_linewidths(edge_width)

    def render(self, data: FieldData, style: FigureConfig) -> RenderResult:
        """
        Render the polygons with guaranteed 1D polygon data.
        `data` is a *single* FieldData object.
        """
        if not self._initialized:
            raise RuntimeError(
                "Component not initialized. Call initialize() first."
            )

        if data.ndim != 1 or not data.name.endswith("_polygons"):
            raise ValueError(
                "PolygonPlotComponent received invalid data. "
                "Expected 1D FieldData with '_polygons' suffix."
            )

        # Extract data from the "Polygon Contract"
        patches = data.domain
        values = data.values

        # compute domain bounds for setting axis limits
        import numpy as np

        all_x = [pt[0] for patch in patches for pt in patch]
        all_y = [pt[1] for patch in patches for pt in patch]
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)

        # Create color normalization
        norm = create_color_normalization(
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

            # Create the new PolyCollection
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

        # set limits only on first render (preserves CLI limits and user zoom)
        if self._first_render:
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self._first_render = False

        self.ax.set_aspect("equal", adjustable="box")

        if style.draw_bodies and self.bodies:
            self.draw_bodies(
                self.bodies,
                zorder=10,
                axes=data.axis_names if data.axis_names else ["x1", "x2"],
            )

        if self.props.show_level_bounds and data.level_bounds:
            self._draw_level_bounds(data.level_bounds)

        return RenderResult(
            artists={"collection": self._poly_collection},
            metadata={"mappable": self._poly_collection},
        )

    def draw_bodies(
        self, body_collection: BodyCollection, zorder: int, axes: Sequence[str]
    ) -> None:
        """Draw immersed bodies on the plot."""
        draw_bodies(self.ax, body_collection, zorder, axes)

    def _draw_level_bounds(
        self, level_bounds: Sequence[tuple[float, float, float, float]]
    ) -> None:
        """Draw rectangles around each AMR level's bounding box."""
        import matplotlib.patches as mpatches

        # clear old level rectangles
        for artist in self._level_artists:
            artist.remove()
        self._level_artists = []

        # skip level 0 (coarsest) - only show refined level boundaries
        for bounds in level_bounds[1:]:
            x0, x1, y0, y1 = bounds
            rect = mpatches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=self.props.level_color,
                linewidth=self.props.level_linewidth,
                alpha=0.2,  # self.props.level_alpha,
                zorder=5,
            )
            self.ax.add_patch(rect)
            self._level_artists.append(rect)

    def cleanup(self) -> None:
        if self._poly_collection:
            self._poly_collection.remove()
        self._poly_collection = None
        for artist in self._level_artists:
            artist.remove()
        self._level_artists = []
