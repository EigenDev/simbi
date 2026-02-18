from typing import List, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pydantic import ValidationInfo, field_validator

from simbi.viz.utility import get_field_str

from ..config import FigureConfig
from ..types import Array, FieldData, RenderResult
from .interface import Component, ComponentProps


def _calculate_moving_average(values: Array, window_size: int = 5) -> Array:
    """Calculate moving average of values."""
    if window_size <= 1 or len(values) <= window_size:
        return np.array([])
    weights = np.ones(window_size) / window_size
    return np.convolve(values, weights, mode="valid")


def _calculate_trend(times: Array, values: Array, degree: int = 1) -> Array:
    """Calculate polynomial trend line."""
    if len(times) < degree + 1:
        return np.zeros_like(times)
    coeffs = np.polyfit(times, values, degree)
    return np.polyval(coeffs, times)


class TimeSeriesPlotProps(ComponentProps):
    """Properties for time_series plot component."""

    label: Optional[str] = None
    linestyle: str = "-"
    linewidth: float = 2.0
    marker: Optional[str] = None
    marker_size: float = 6.0
    alpha: float = 0.6
    normalization: Optional[float] = None

    show_moving_average: bool = False
    moving_average_window: int = 5
    show_trend: bool = False
    trend_degree: int = 1

    @field_validator("linewidth", "marker_size", "alpha")
    @classmethod
    def validate_positive_float(cls, v: float, info: ValidationInfo) -> float:
        if v <= 0:
            field_name = info.field_name or "Value"
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v

    @field_validator("moving_average_window", "trend_degree")
    @classmethod
    def validate_positive_int(cls, v: int, info: ValidationInfo) -> int:
        if v <= 0:
            field_name = info.field_name or "Value"
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v


class TimeSeriesPlotComponent(Component[TimeSeriesPlotProps, FieldData]):
    """
    A "smart" renderer for 1D or 2D time series data.

    Expects FieldData:
    - 1D: (N_times,) -> Renders 1 line
    - 2D: (N_times, N_bodies) -> Renders N_bodies lines
    """

    def __init__(self, props: TimeSeriesPlotProps):
        self.props = props
        self._main_lines: List[Line2D] = []
        self._ma_lines: List[Line2D] = []
        self._trend_lines: List[Line2D] = []
        self._initialized: bool = False

    def initialize(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax = ax
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, props: TimeSeriesPlotProps) -> None:
        self.props = props
        # In a full impl, this would update styles or rebuild lines
        pass

    def _get_labels(self, num_lines: int) -> list[str]:
        """Get labels for one or more lines."""
        labels: list[str] = []
        if not self.data.name.startswith("$"):
            base_label = get_field_str(self.props.label or self.data.name)
        else:
            base_label = self.data.name

        if num_lines == 1:
            labels = [base_label]
        elif self.data.body_names and len(self.data.body_names) == num_lines:
            labels = [f"${name}$" for name in self.data.body_names]
        else:
            labels = [f"{base_label}_{i}" for i in range(num_lines)]
        return labels

    def render(self, data: FieldData, style: FigureConfig) -> RenderResult:
        """Render the time_series plot with 1D or 2D data and return a RenderResult."""
        if not self._initialized:
            raise RuntimeError("Component not initialized.")

        self.data = data  # Store data for helper access

        times = data.domain[0]
        values = data.values

        if style.time_scale:
            times = times / style.time_scale

        if values.ndim == 1:
            values_2d = values.reshape(-1, 1)
        elif values.ndim == 2:
            values_2d = values
        else:
            raise ValueError(
                "TimeSeriesPlotComponent expects 1D or 2D FieldData."
            )

        num_lines = values_2d.shape[1]
        labels = self._get_labels(num_lines)

        all_rendered_lines: List[Line2D] = []

        for i in range(num_lines):
            line_values = values_2d[:, i]
            line_label = labels[i]
            norm = self.props.normalization or 1.0
            main_line = self.ax.plot(
                times,
                line_values / norm,
                label=line_label,
                linewidth=self.props.linewidth,
                alpha=self.props.alpha,
            )[0]
            if norm != 1:
                self.ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
            all_rendered_lines.append(main_line)

        # render percentile bands if present
        if data.bands is not None:
            p_lo, p_hi = data.bands
            norm = self.props.normalization or 1.0
            color = all_rendered_lines[-1].get_color() if all_rendered_lines else "C0"
            self.ax.fill_between(
                times, p_lo / norm, p_hi / norm,
                alpha=0.2, color=color, label="10th-90th percentile",
            )

            # --- Render Decorators (Moving Avg / Trend) ---
            if self.props.show_moving_average:
                ma_values = _calculate_moving_average(
                    line_values, self.props.moving_average_window
                )
                if ma_values.any():
                    offset = (self.props.moving_average_window - 1) // 2
                    ma_times = times[offset : offset + len(ma_values)]
                    ma_line = self.ax.plot(
                        ma_times, ma_values, label=f"{line_label} (Avg)"
                    )[0]
                    all_rendered_lines.append(ma_line)

            if self.props.show_trend:
                trend_values = _calculate_trend(
                    times, line_values, self.props.trend_degree
                )
                trend_line = self.ax.plot(
                    times, trend_values, label=f"{line_label} (Trend)"
                )[0]
                all_rendered_lines.append(trend_line)

        # return a RenderResult containing all created line artists and labels metadata
        return RenderResult(
            artists={"lines": all_rendered_lines}, metadata={"labels": labels}
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        all_lines = self._main_lines + self._ma_lines + self._trend_lines
        if hasattr(self, "ax"):
            for line in all_lines:
                if line and line in self.ax.lines:
                    line.remove()
        self._main_lines = []
        self._ma_lines = []
        self._trend_lines = []
