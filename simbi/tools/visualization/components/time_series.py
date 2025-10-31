"""time_series plot component for visualization."""

from itertools import cycle
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pydantic import ValidationInfo, field_validator

from ...utility import get_field_str
from ..core.config import StyleConfig
from ..core.types import Array, FieldData, PlotData
from ..formatters.time_series import (
    format_time_series_plot_axes,
)
from .interface import Component, ComponentProps


def extract_time_data(field: FieldData) -> Array:
    """Extract time data from field.

    For time_series plots, we expect the x-axis domain to represent time.
    """
    # Use first domain array as time coordinates
    if field.domain:
        return field.domain[0]

    # Fallback to index array if no domain provided
    return np.arange(field.values.size, dtype=np.floating)


def extract_value_data(field: FieldData) -> Array:
    """Extract values from field for plotting."""
    values = field.values

    # Handle dimensionality - ensure we have a 1D array
    if values.ndim > 1:
        # Either take first row/column or flatten
        if values.shape[0] == 1:
            values = values[0]
        elif values.shape[1] == 1:
            values = values[:, 0]
        else:
            # Take mean along appropriate axis or just flatten
            values = np.mean(values, axis=1)

    return values


def calculate_moving_average(values: Array, window_size: int = 5) -> Array:
    """Calculate moving average of values."""
    if window_size <= 1 or len(values) <= window_size:
        return values

    # Use numpy's convolve for efficient moving average
    weights = np.ones(window_size) / window_size
    return np.convolve(values, weights, mode="valid")


def calculate_trend(times: Array, values: Array, degree: int = 1) -> Array:
    """Calculate polynomial trend line."""
    # Skip if not enough data points
    if len(times) < degree + 1:
        return np.zeros_like(times)

    # Fit polynomial
    coeffs = np.polyfit(times, values, degree)

    # Generate trend values
    return np.polyval(coeffs, times)


def create_time_series_style(
    linewidth: float,
    alpha: float,
    color: Optional[str],
    linestyle: str,
    marker: Optional[str],
    marker_size: float,
) -> Dict[str, Any]:
    """Create styling for a time_series line."""
    style = {"linewidth": linewidth, "alpha": alpha, "linestyle": linestyle}

    # Add color if specified
    if color:
        style["color"] = color

    # Add marker if specified
    if marker:
        style["marker"] = marker
        style["markersize"] = marker_size

    return style


class time_seriesPlotProps(ComponentProps):
    """Properties for time_series plot component."""

    field_index: int = 0
    color: Optional[str] = None
    linestyle: str = "-"
    linewidth: float = 2.0
    marker: Optional[str] = None
    marker_size: float = 6.0
    alpha: float = 1.0
    label: Optional[str] = None
    show_moving_average: bool = False
    moving_average_window: int = 5
    show_trend: bool = False
    trend_degree: int = 1

    @field_validator("field_index")
    @classmethod
    def validate_field_index(cls, v: int, info: ValidationInfo) -> int:
        """Validate that field index is non-negative."""
        if v < 0:
            raise ValueError(f"Field index must be non-negative, got {v}")
        return v

    @field_validator("linewidth", "marker_size", "alpha")
    @classmethod
    def validate_positive_float(cls, v: float, info: ValidationInfo) -> float:
        """Validate that numeric values are positive."""
        if v <= 0:
            field_name = info.field_name or "Value"
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v

    @field_validator("moving_average_window", "trend_degree")
    @classmethod
    def validate_positive_int(cls, v: int, info: ValidationInfo) -> int:
        """Validate that integer values are positive."""
        if v <= 0:
            field_name = info.field_name or "Value"
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v


class time_seriesPlotComponent(Component):
    """time_series plot visualization component."""

    def __init__(self, props: time_seriesPlotProps):
        """Initialize the time_series plot component."""
        self.props = props
        self.times = np.array([])
        self.values = np.array([])
        self._main_line: Optional[Line2D] = None
        self._ma_line: Optional[Line2D] = None  # Moving average line
        self._trend_line: Optional[Line2D] = None  # Trend line
        self._initialized: bool = False

    def initialize(self, fig: Figure, ax: Axes) -> None:
        """Initialize the component with figure and axes."""
        self.fig = fig
        self.ax = ax
        self._initialized = True

        # Create empty lines
        empty_data = np.array([])

        # Main data line
        # self._main_line = self.ax.plot(
        #     empty_data,
        #     empty_data,
        #     **create_time_series_style(
        #         self.props.linewidth,
        #         self.props.alpha,
        #         self.props.color,
        #         self.props.linestyle,
        #         self.props.marker,
        #         self.props.marker_size,
        #     ),
        #     label=get_field_str(self.props.label),
        # )[0]

        # Moving average line (if enabled)
        if self.props.show_moving_average:
            self._ma_line = self.ax.plot(
                empty_data,
                empty_data,
                linestyle="--",
                color="orange" if not self.props.color else self.props.color,
                alpha=0.7,
                linewidth=1.5,
                label="Moving Avg",
            )[0]

        # Trend line (if enabled)
        if self.props.show_trend:
            self._trend_line = self.ax.plot(
                empty_data,
                empty_data,
                linestyle="-.",
                color="red" if not self.props.color else self.props.color,
                alpha=0.7,
                linewidth=1.5,
                label=f"Trend (deg={self.props.trend_degree})",
            )[0]

    @property
    def initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized

    def update(self, props: time_seriesPlotProps) -> None:
        """Update component properties."""
        prev_props = self.props
        self.props = props

        # Handle changes that require reinitializing lines
        needs_reinit = (
            prev_props.show_moving_average != props.show_moving_average
        ) or (prev_props.show_trend != props.show_trend)

        if needs_reinit and hasattr(self, "ax"):
            self.cleanup()
            self.initialize(self.fig, self.ax)
        else:
            # Update existing lines
            if self._main_line:
                style = create_time_series_style(
                    props.linewidth,
                    props.alpha,
                    props.color,
                    props.linestyle,
                    props.marker,
                    props.marker_size,
                )

                for key, value in style.items():
                    setter = getattr(self._main_line, f"set_{key}", None)
                    if setter:
                        setter(value)

                # Update label
                if prev_props.label != props.label:
                    self._main_line.set_label(props.label)

    def render(self, data: PlotData, style: StyleConfig) -> None:
        """Render the time_series plot with data."""
        if not self._initialized or not hasattr(self, "ax"):
            raise RuntimeError(
                "Component not initialized. Call initialize() first."
            )

        # Update main line
        for idx, field in enumerate(data.fields):
            if field.values.ndim > 1:
                ls = cycle(["-", "--", ":", "-."])
                colormap = plt.get_cmap(next(style.cmap))
                colors = cycle(
                    [
                        colormap(c)
                        for c in np.linspace(0.25, 0.75, field.values.ndim)
                    ]
                )
                if style.value_scale:
                    SCALE = style.value_scale[0]
                else:
                    SCALE = 1.0

                for vidx, vals in enumerate(field.values.T):
                    if field.name in ["mdot", "maccr"]:
                        label = f"$M_{vidx + 1}$"
                    else:
                        label = get_field_str(field.name)

                    times = field.domain[idx].copy()
                    if style.time_scale:
                        times /= style.time_scale
                    self.ax.plot(
                        times,
                        vals / SCALE,
                        label=label,
                        linestyle=next(ls),
                        color=next(colors),
                    )

                def compute_orbital_averages(time, mdot, time_scale):
                    """Compute averages over orbital periods.

                    Args:
                        time: array of time values
                        mdot: array of mdot values
                        time_scale: orbital period (e.g. 2π)

                    Returns:
                        t_bins: array of time bin centers
                        mdot_avg: array of averaged mdot values
                    """
                    n_orbits = (time[-1] - time[0]) / time_scale
                    bins = np.linspace(time[0], time[-1], int(n_orbits) + 1)
                    t_bins = (bins[1:] + bins[:-1]) / 2  # bin centers
                    mdot_avg = np.array(
                        [
                            np.mean(
                                mdot[(time >= bins[i]) & (time < bins[i + 1])]
                            )
                            for i in range(len(bins) - 1)
                        ]
                    )
                    return t_bins / time_scale, mdot_avg

                if field.name == "mdot" and field.values.shape[1] > 1:
                    ma_times, ma_total_mdot = compute_orbital_averages(
                        field.domain[idx],
                        np.sum(field.values, axis=1),
                        time_scale=style.time_scale or 2 * np.pi,
                    )

                    self.ax.plot(
                        ma_times,
                        ma_total_mdot / SCALE,
                        label=r"$M_{\rm total}$ (Orbital Average)",
                        linestyle="solid",
                        color="black",
                        linewidth=1.5,
                        marker="o",
                    )
            else:
                self.ax.plot(
                    field.domain, field.values, label=get_field_str(field.name)
                )

        # Update moving average line if enabled
        # if self.props.show_moving_average and self._ma_line:
        #     ma_values = calculate_moving_average(
        #         values, self.props.moving_average_window
        #     )

        #     # Adjust times for valid moving average values
        #     ma_times = times
        #     if len(ma_values) < len(times):
        #         # Handle edge trimming from convolution
        #         offset = (self.props.moving_average_window - 1) // 2
        #         ma_times = times[offset : offset + len(ma_values)]

        #     self._ma_line.set_data(ma_times, ma_values)

        # # Update trend line if enabled
        # if self.props.show_trend and self._trend_line:
        #     trend_values = calculate_trend(
        #         times, values, self.props.trend_degree
        #     )
        #     self._trend_line.set_data(times, trend_values)

        # Show legend if any line has a label
        # if any(
        #     line.get_label() and not str(line.get_label()).startswith("_")
        #     for line in self._get_active_lines()
        # ):
        #     self.ax.legend()

        format_time_series_plot_axes(
            self.ax, data, self.props.field_index, style
        )
        # return self._get_active_lines()

    def _get_active_lines(self) -> List[Line2D]:
        """Get all active (non-None) lines."""
        lines = []
        if self._main_line:
            lines.append(self._main_line)
        if self._ma_line:
            lines.append(self._ma_line)
        if self._trend_line:
            lines.append(self._trend_line)
        return lines

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, "ax"):
            for line in self._get_active_lines():
                if line in self.ax.lines:
                    line.remove()

            self._main_line = None
            self._ma_line = None
            self._trend_line = None
