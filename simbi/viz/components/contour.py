# =============================================================================
# contour.py
#
# contour plot component for overlay visualization.
# renders contour lines or filled contours over 2d field data.
#
# usage:
#   props = ContourPlotProps(levels=[1.0], color="white")
#   component = ContourPlotComponent(props)
#   component.initialize(fig, ax)
#   result = component.render(field_data, style)
#
# notes:
#   - designed for overlay use (e.g., mach=1 contour over schlieren)
#   - handles both edge coordinates and cell centers
#   - supports animation via cleanup/re-render cycle
# =============================================================================
from typing import Optional, Sequence, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure

from ..config import FigureConfig
from ..types import Array, FieldData, RenderResult
from .interface import ComponentProps


class ContourPlotProps(ComponentProps):
    """properties for contour overlay component."""

    levels: Sequence[float] = (1.0,)
    colors: Optional[Sequence[str]] = None
    color: str = "white"
    linewidths: Union[float, Sequence[float]] = 1.5
    linestyles: Union[str, Sequence[str]] = "-"
    alpha: float = 1.0
    filled: bool = False
    label_contours: bool = False
    label_fontsize: int = 8
    label_inline: bool = True
    zorder: int = 10


class ContourPlotComponent:
    """
    renders contour lines over 2d data.

    this component is intended for overlay use - adding contour lines
    on top of an existing 2d visualization (e.g., mach=1 surface over
    schlieren plot).
    """

    def __init__(self, props: ContourPlotProps):
        self.props = props
        self._contour_set: Optional[QuadContourSet] = None
        self._initialized: bool = False
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None

    def initialize(self, fig: Figure, ax: Axes) -> None:
        """initialize the component with figure and axes."""
        self.fig = fig
        self.ax = ax
        self._initialized = True

    @property
    def initialized(self) -> bool:
        """check if the component has been initialized."""
        return self._initialized

    def update(self, props: ContourPlotProps) -> None:
        """update component properties."""
        self.props = props

    def render(self, data: FieldData, style: FigureConfig) -> RenderResult:
        """
        render contour lines on 2d field data.

        handles coordinate conversion from edges to centers if needed.
        cleans up previous contours for animation support.
        """
        if not self._initialized:
            raise RuntimeError(
                "component not initialized. call initialize() first."
            )

        if data.ndim != 2:
            raise ValueError(
                f"ContourPlotComponent requires 2d data, got ndim={data.ndim}"
            )

        # cleanup previous contours for animation
        self._remove_contours()

        # extract coordinates and values
        # domain is [y_coords, x_coords] for 2d data
        x, y = data.domain[1], data.domain[0]
        values = data.values

        # convert edges to centers if coordinates are edge-based
        x_centers = _edges_to_centers(x, values.shape[1])
        y_centers = _edges_to_centers(y, values.shape[0])

        # create meshgrid for contour
        X, Y = np.meshgrid(x_centers, y_centers)

        # determine colors
        colors = self.props.colors if self.props.colors else self.props.color

        if self.props.filled:
            self._contour_set = self.ax.contourf(
                X,
                Y,
                values,
                levels=list(self.props.levels),
                colors=colors,
                alpha=self.props.alpha,
                zorder=self.props.zorder,
            )
        else:
            self._contour_set = self.ax.contour(
                X,
                Y,
                values,
                levels=list(self.props.levels),
                colors=colors,
                linewidths=self.props.linewidths,
                linestyles=self.props.linestyles,
                alpha=self.props.alpha,
                zorder=self.props.zorder,
            )

        if self.props.label_contours and self._contour_set:
            self.ax.clabel(
                self._contour_set,
                inline=self.props.label_inline,
                fontsize=self.props.label_fontsize,
            )

        return RenderResult(
            artists={"contour": self._contour_set},
            metadata={
                "is_contour": True,
                "levels": list(self.props.levels),
                "is_overlay": True,
            },
        )

    def _remove_contours(self) -> None:
        """remove existing contour artists from axes."""
        if self._contour_set is not None:
            # matplotlib >= 3.8 removed collections attribute; use remove() directly
            try:
                self._contour_set.remove()
            except (ValueError, AttributeError):
                # fallback for older matplotlib versions
                if hasattr(self._contour_set, "collections"):
                    for coll in self._contour_set.collections:
                        try:
                            coll.remove()
                        except ValueError:
                            pass
            self._contour_set = None

    def cleanup(self) -> None:
        """clean up resources."""
        self._remove_contours()


def _edges_to_centers(coords: Array, expected_size: int) -> Array:
    """
    convert edge coordinates to cell centers if needed.

    if coords has one more element than expected_size, assumes edge-based
    and computes midpoints. otherwise returns coords unchanged.
    """
    if len(coords) == expected_size + 1:
        return 0.5 * (coords[:-1] + coords[1:])
    return coords
