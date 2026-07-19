"""
Quiver plot component for visualization.

This component is a simple renderer. It expects to be given
a list of two 2D FieldData objects (U, V) and will render them
as a quiver plot.
"""

from typing import List, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
from pydantic import ValidationInfo, field_validator

from ..config import FigureConfig
from ..types import Array, FieldData, RenderResult
from .interface import Component, ComponentProps


class QuiverPlotProps(ComponentProps):
    """Properties for a quiver plot component."""

    color: str = "white"
    scale: Optional[float] = None
    width: Optional[float] = 0.002
    alpha: float = 1.0
    skip: int = 5  # plot every 'skip' vector

    @field_validator("skip")
    @classmethod
    def validate_skip(cls, v: int, _: ValidationInfo) -> int:
        if v < 1:
            raise ValueError(f"skip must be >= 1, got {v}")
        return int(v)

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float, _: ValidationInfo) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {v}")
        return v


class QuiverPlotComponent(Component):
    """
    A simple renderer for a 2D quiver plot.
    Expects list[FieldData] with [U, V] components.
    """

    def __init__(self, props: QuiverPlotProps):
        self.props = props
        self._quiver: Optional[Quiver] = None
        self._initialized: bool = False

    def initialize(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax = ax
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, props: QuiverPlotProps) -> None:
        """Update component properties and restyle the quiver."""
        self.props = props
        # quiver styling is complex to update live.
        # render() rebuilds the quiver on redraw.

    def _validate_and_prep_data(
        self, data: List[FieldData]
    ) -> tuple[Array, Array, Array, Array]:
        """Validate data and prepare it for plotting."""
        if not data or len(data) < 2:
            raise ValueError(
                "QuiverPlotComponent requires at least two fields (U, V)."
            )

        u_field, v_field = data[0], data[1]

        if u_field.ndim != 2 or v_field.ndim != 2:
            raise ValueError("QuiverPlotComponent fields must be 2D.")

        # the pipeline provides pcolormesh-style "edge" coordinates
        x_edges, y_edges = u_field.domain
        u_values, v_values = u_field.values, v_field.values

        # quiver plots on cell *centers*.
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        X, Y = np.meshgrid(x_centers, y_centers)

        # apply the 'skip' prop for downsampling
        sl = slice(None, None, self.props.skip)

        X_sparse, Y_sparse = X[sl, sl], Y[sl, sl]
        U_sparse, V_sparse = u_values[sl, sl], v_values[sl, sl]

        return X_sparse, Y_sparse, U_sparse, V_sparse

    def render(
        self, data: List[FieldData], style: FigureConfig
    ) -> RenderResult:
        """
        Render the quiver plot and return a RenderResult.
        'data' is a list of FieldData objects [U, V].
        """
        if not self._initialized:
            raise RuntimeError(
                "Component not initialized. Call initialize() first."
            )

        X, Y, U, V = self._validate_and_prep_data(data)

        # quiver is slow to update. it's cleaner to remove and redraw.
        if self._quiver:
            self._quiver.remove()

        self._quiver = self.ax.quiver(
            X,
            Y,
            U,
            V,
            color=self.props.color,
            scale=self.props.scale,
            width=self.props.width,
            alpha=self.props.alpha,
        )

        return RenderResult(
            artists={"quiver": self._quiver}, metadata={"is_vector": True}
        )

    def cleanup(self) -> None:
        if self._quiver:
            self._quiver.remove()
        self._quiver = None
