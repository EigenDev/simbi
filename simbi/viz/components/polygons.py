"""
Polygon plot component for visualization.

This component is a "simple" renderer. It expects to be given
a single, 1D FieldData object where the domain is a list of patches
and the values are a list of corresponding colors.
"""

from typing import Any, Optional, Sequence

from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from pydantic import ValidationInfo, field_validator

from simbi.core.types.bodies import Body

from ..config import StyleConfig
from ..types import ColorRange, FieldData
from .interface import Component, ComponentProps
from .quad import _create_color_normalization

LOGICAL_AXIS_MAP = {"x1": 0, "x2": 1, "x3": 2}


class PolygonPlotProps(ComponentProps):
    """Properties for a *single* polygon plot component."""

    cmap: str = "viridis"
    color_range: ColorRange = ColorRange(min=None, max=None)
    log_scale: bool = False
    power: float = 1.0
    alpha: float = 1.0

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


class PolygonPlotComponent(Component):
    """
    A "simple" renderer for 2D FMR data as polygons.
    Expects 1D FieldData adhering to the "Polygon Contract".
    """

    def __init__(
        self, props: PolygonPlotProps, bodies: Optional[dict[str, Body]] = None
    ):
        self.props = props
        self._poly_collection: Optional[PolyCollection] = None
        self._initialized: bool = False
        self.bodies: Optional[dict[str, Body]] = bodies

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

    def render(self, data: FieldData, style: StyleConfig) -> dict[str, Any]:
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

        # Create color normalization
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

        if style.draw_bodies and self.bodies:
            self.draw_bodies(
                self.bodies,
                zorder=10,
                axes=data.axis_names if data.axis_names else ["x1", "x2"],
            )

        self._poly_collection.set_array(values)
        self._poly_collection.set_norm(norm)

        self.ax.add_collection(self._poly_collection)
        self.ax.autoscale_view()
        self.ax.set_aspect("equal", adjustable="box")

        return {"collection": self._poly_collection}

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

    def cleanup(self) -> None:
        if self._poly_collection:
            self._poly_collection.remove()
        self._poly_collection = None
