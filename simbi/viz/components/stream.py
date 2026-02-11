"""
Stream plot component for visualization.

This component is a simple renderer. It expects to be given
a list of two 2D FieldData objects (U, V) and will render them
as a stream plot.
"""

from typing import List, Optional, Tuple

import matplotlib.colors as mcolors
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
    alpha: float = 0.35

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
        # Stream plots must be fully redrawn on render

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

        # streamplot needs 1D coordinate arrays, not 2D meshgrids.
        # we assume the pipeline gives us pcolormesh-style "edge" coordinates
        x, y = u_field.domain
        u_values, v_values = u_field.values, v_field.values

        # We plot on cell *centers*.
        # x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        # y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # streamplot expects U, V to be (N, M)
        # and x, y to be (M,) and (N,)
        # Our data is (N, M) and our centers are (M,) and (N,)
        # This matches the matplotlib convention.

        return x, y, u_values, v_values

    def _crop_to_limits(self, x, y, u, v, style: FigureConfig):
        """crop velocity data to axis limits for denser streamlines in zoomed views."""
        import numpy as np

        xlims = style.xlims
        ylims = style.ylims

        if not xlims and not ylims:
            return x, y, u, v, False

        xmin = xlims.min if xlims and xlims.min is not None else x[0]
        xmax = xlims.max if xlims and xlims.max is not None else x[-1]
        ymin = ylims.min if ylims and ylims.min is not None else y[0]
        ymax = ylims.max if ylims and ylims.max is not None else y[-1]

        xi = np.where((x >= xmin) & (x <= xmax))[0]
        yi = np.where((y >= ymin) & (y <= ymax))[0]

        if len(xi) < 2 or len(yi) < 2:
            return x, y, u, v, False

        x_crop = x[xi]
        y_crop = y[yi]
        u_crop = u[np.ix_(yi, xi)]
        v_crop = v[np.ix_(yi, xi)]

        return x_crop, y_crop, u_crop, v_crop, True

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

        # Stream plots are static and must be cleared and redrawn
        self.cleanup()
        x = 0.5 * (x[1:] + x[:-1])
        y = 0.5 * (y[1:] + y[:-1])

        x, y, u, v, is_cropped = self._crop_to_limits(x, y, u, v, style)
        alpha = self.props.alpha * 0.5 if is_cropped else self.props.alpha

        # bake alpha into the color tuple so both lines and arrows
        # inherit it — streamplot has no alpha kwarg and post-hoc
        # set_alpha is overridden by explicit facecolors

        rgba = mcolors.to_rgba(self.props.color, alpha=alpha)

        self._streamplot = self.ax.streamplot(
            x,
            y,
            u,
            v,
            color=rgba,
            linewidth=self.props.linewidth,
            density=self.props.density,
            arrowstyle=self.props.arrowstyle,
            arrowsize=self.props.arrowsize,
        )

        return RenderResult(
            artists={"streamplot": self._streamplot},
            metadata={"is_vector": True},
        )

    def cleanup(self) -> None:
        if self._streamplot:
            # streamplot returns a composite object
            # We must remove its lines
            if hasattr(self._streamplot, "lines") and self._streamplot.lines:
                self.ax.collections.remove(self._streamplot.lines)

            # And its arrows (if they exist)
            if hasattr(self._streamplot, "arrows") and self._streamplot.arrows:
                self._streamplot.arrows.remove()

        self._streamplot = None
