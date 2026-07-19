"""
Stream plot component for visualization.

This component is a simple renderer. It expects to be given
a list of two 2D FieldData objects (U, V) and will render them
as a stream plot.
"""

from typing import List, Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.streamplot import StreamplotSet
from pydantic import ValidationInfo, field_validator

from ..config import FigureConfig
from ..types import Array, FieldData, RenderResult
from .interface import Component, ComponentProps


class StreamPlotProps(ComponentProps):
    """Properties for a stream plot component."""

    color: str = "white"
    linewidth: float = 0.5
    density: float | Tuple[float, float] = 1.0
    arrowstyle: str = "->"
    arrowsize: float = 1.0
    alpha: float = 0.6

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float, _: ValidationInfo) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {v}")
        return v


class StreamPlotComponent(Component):
    """
    A simple renderer for a 2D stream plot.
    Expects list[FieldData] with [U, V] components.
    """

    def __init__(self, props: StreamPlotProps):
        self.props = props
        self._streamplot: Optional[StreamplotSet] = None
        self._initialized: bool = False

    def initialize(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax = ax
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, props: StreamPlotProps) -> None:
        """Update component properties."""
        self.props = props
        # stream plots must be fully redrawn on render

    def _validate_and_prep_data(
        self, data: List[FieldData]
    ) -> tuple[Array, Array, Array, Array]:
        """Validate data and prepare it for plotting."""
        if not data or len(data) < 2:
            raise ValueError(
                "StreamPlotComponent requires at least two fields (U, V)."
            )

        u_field, v_field = data[0], data[1]

        if u_field.ndim != 2 or v_field.ndim != 2:
            raise ValueError("StreamPlotComponent fields must be 2D.")

        # streamplot needs 1D coordinate arrays.
        # the pipeline provides pcolormesh-style "edge" coordinates
        x, y = u_field.domain
        u_values, v_values = u_field.values, v_field.values

        # plotted on cell *centers*.
        # x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        # y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # streamplot expects U, V to be (N, M)
        # and x, y to be (M,) and (N,)
        # the data is (N, M) and the centers are (M,) and (N,)
        # this matches the matplotlib convention.

        return x, y, u_values, v_values

    def render(
        self, data: List[FieldData], style: FigureConfig
    ) -> RenderResult:
        """
        Render the stream plot.
        `data` is a *list* of FieldData objects [U, V].
        """

        if not self._initialized:
            raise RuntimeError(
                "Component not initialized. Call initialize() first."
            )

        x, y, u, v = self._validate_and_prep_data(data)

        # stream plots are static and must be cleared and redrawn
        self.cleanup()
        x = 0.5 * (x[1:] + x[:-1])
        y = 0.5 * (y[1:] + y[:-1])

        self._streamplot = self.ax.streamplot(
            x,
            y,
            u,
            v,
            color=self.props.color,
            linewidth=self.props.linewidth,
            density=self.props.density,
            arrowstyle=self.props.arrowstyle,
            arrowsize=self.props.arrowsize,
        )
        self._streamplot.lines.set_alpha(self.props.alpha)

        return RenderResult(
            artists={"streamplot": self._streamplot},
            metadata={"is_vector": True},
        )

    def cleanup(self) -> None:
        if self._streamplot:
            # streamplot returns a composite object
            # remove its lines
            if hasattr(self._streamplot, "lines") and self._streamplot.lines:
                self.ax.collections.remove(self._streamplot.lines)

            # and its arrows (if they exist)
            if hasattr(self._streamplot, "arrows") and self._streamplot.arrows:
                self._streamplot.arrows.remove()

        self._streamplot = None
