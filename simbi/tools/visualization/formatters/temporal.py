"""Formatting functions for temporal plots."""

from typing import Any, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from simbi.tools.utility import get_field_str
from simbi.tools.visualization.core.config import StyleConfig

from ..core.types import PlotData
from .common import (
    apply_basic_style,
    calculate_margins,
    set_axis_scales,
    set_axis_title,
)


def format_temporal_plot_axes(
    ax: Axes,
    data: PlotData,
    field_index: int,
    style: StyleConfig,
    time_units: Optional[str] = None,
    value_units: Optional[str] = None,
) -> None:
    """Format axes for a temporal plot."""
    apply_basic_style(ax)
    set_axis_title(ax, style.setup, time=None)
    apply_temporal_labels(
        ax,
        data,
        field_index,
        style.x_label,
        style.y_label,
        time_units,
        value_units,
    )

    set_axis_scales(ax, log_x=style.semilogy, log_y=style.semilogy)

    if style.xlims or style.ylims:
        if style.xlims:
            ax.set_xlim(style.xlims.min, style.xlims.max)
        if style.ylims:
            ax.set_ylim(style.ylims.min, style.ylims.max)
    # else:
    # auto_scale_temporal_axes(ax, data, field_index)


def apply_temporal_labels(
    ax: Axes,
    data: PlotData,
    field_index: int,
    x_label: Optional[str],
    y_label: Optional[str],
    time_units: Optional[str],
    value_units: Optional[str],
) -> None:
    """Apply axis labels for temporal plot."""
    # Set x-label (time axis)
    if x_label:
        ax.set_xlabel(x_label)
    else:
        time_label = "$t$"
        if time_units:
            time_label += f" [{time_units}]"
        ax.set_xlabel(time_label)

    # Set y-label (value axis)
    if y_label:
        ax.set_ylabel(y_label)
    elif field_index < len(data.fields):
        field = data.fields[field_index]
        value_label = get_field_str(field.name)
        if value_units:
            value_label += f" [{value_units}]"
        ax.set_ylabel(value_label)


def auto_scale_temporal_axes(
    ax: Axes, data: PlotData, field_index: int
) -> None:
    """Automatically set axis limits for temporal plot."""
    # Skip if field index is out of range
    if field_index >= len(data.fields):
        return

    field = data.fields[field_index]

    # Extract data
    times = field.domain[0] if field.domain else np.arange(field.values.size)
    values = field.values

    # Handle dimensionality
    if values.ndim > 1:
        if values.shape[0] == 1:
            values = values[0]
        elif values.shape[1] == 1:
            values = values[:, 0]
        else:
            values = np.mean(values, axis=1)

    # Calculate limits
    if times.size > 0:
        t_min, t_max = float(np.min(times)), float(np.max(times))
        t_margin_left, t_margin_right = calculate_margins(t_min, t_max)
        ax.set_xlim(t_min - t_margin_left, t_max + t_margin_right)

    if values.size > 0:
        v_min, v_max = float(np.nanmin(values)), float(np.nanmax(values))
        v_margin_bottom, v_margin_top = calculate_margins(v_min, v_max)
        ax.set_ylim(v_min - v_margin_bottom, v_max + v_margin_top)


def format_time_ticks(
    ax: Axes,
    major_spacing: Optional[float] = None,
    minor_spacing: Optional[float] = None,
    date_format: Optional[str] = None,
) -> None:
    """Format time axis ticks."""
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    # Set major ticks if spacing provided
    if major_spacing:
        ax.xaxis.set_major_locator(MultipleLocator(major_spacing))

    # Set minor ticks if spacing provided
    if minor_spacing:
        ax.xaxis.set_minor_locator(MultipleLocator(minor_spacing))
    else:
        # Otherwise use auto minor locator
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    # Format dates if needed
    if date_format:
        from matplotlib.dates import DateFormatter

        ax.xaxis.set_major_formatter(DateFormatter(date_format))

    # Make sure ticks are inside
    ax.tick_params(which="both", direction="in")
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)


def add_annotations(
    ax: Axes,
    times: np.ndarray,
    values: np.ndarray,
    labels: list[str],
    offsets: Optional[list[tuple[float, float]]] = None,
    arrow_props: Optional[dict[str, Any]] = None,
) -> None:
    """Add annotations to points on a temporal plot."""
    from matplotlib.pyplot import annotate

    # Default offset and arrow properties if not provided
    default_offset = (10, 10)  # points
    default_arrow_props = dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")

    # Use provided or default arrow properties
    arrow_props = arrow_props or default_arrow_props

    # Add annotations
    for i, (t, v, label) in enumerate(zip(times, values, labels)):
        # Get offset for this annotation
        offset = offsets[i] if offsets and i < len(offsets) else default_offset

        # Create annotation
        annotate(
            label,
            xy=(t, v),
            xytext=offset,
            textcoords="offset points",
            arrowprops=arrow_props,
        )


def add_event_lines(
    ax: Axes,
    times: list[float],
    labels: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
    linestyles: Optional[list[str]] = None,
    alpha: float = 0.7,
) -> list[Line2D]:
    """Add vertical lines marking events on the temporal plot."""
    lines = []
    default_color = "gray"
    default_linestyle = "--"

    # Create event lines
    for i, t in enumerate(times):
        color = colors[i] if colors and i < len(colors) else default_color
        linestyle = (
            linestyles[i]
            if linestyles and i < len(linestyles)
            else default_linestyle
        )
        label = labels[i] if labels and i < len(labels) else None

        line = ax.axvline(
            x=t, color=color, linestyle=linestyle, alpha=alpha, label=label
        )
        lines.append(line)

    # Update legend if any line has a label
    if labels and any(labels):
        ax.legend()

    return lines
