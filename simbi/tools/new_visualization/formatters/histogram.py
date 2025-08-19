"""Formatting functions for histogram plots."""

from typing import Optional
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ..core.types import PlotData
from .common import (
    apply_basic_style,
    set_axis_title,
    set_axis_grid,
    set_axis_scales,
)


def format_histogram_plot_axes(
    ax: Axes,
    data: PlotData,
    field_index: int,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    auto_scale: bool = True,
    grid: bool = True,
    log_x: bool = True,
    log_y: bool = True,
    x_lim: Optional[tuple[float, float]] = None,
    y_lim: Optional[tuple[float, float]] = None,
    x_units: Optional[str] = None,
    y_units: Optional[str] = None,
) -> None:
    """Format axes for a histogram plot."""
    # Apply basic styling
    apply_basic_style(ax)

    # Set title if provided
    set_axis_title(ax, title)

    # Apply labels
    apply_histogram_labels(ax, data, field_index, x_label, y_label, x_units, y_units)

    # Set scales (histograms often use log-log scales)
    set_axis_scales(ax, log_x=log_x, log_y=log_y)

    # Set grid
    set_axis_grid(ax, grid, linestyle=":", alpha=0.5)

    # Set limits
    apply_histogram_limits(ax, x_lim, y_lim, auto_scale)


def apply_histogram_labels(
    ax: Axes,
    data: PlotData,
    field_index: int,
    x_label: Optional[str],
    y_label: Optional[str],
    x_units: Optional[str],
    y_units: Optional[str],
) -> None:
    """Apply axis labels for histogram plot."""
    # Get field if available
    field_name = ""
    if field_index < len(data.fields):
        field_name = data.fields[field_index].name

    # Set x-label
    if x_label:
        ax.set_xlabel(x_label)
    else:
        # Default x-label based on field
        x_label = field_name
        if x_units:
            x_label += f" [{x_units}]"
        ax.set_xlabel(x_label)

    # Set y-label
    if y_label:
        ax.set_ylabel(y_label)
    else:
        # Default y-label
        y_label = "Count"
        if y_units:
            y_label += f" [{y_units}]"
        ax.set_ylabel(y_label)


def apply_histogram_limits(
    ax: Axes,
    x_lim: Optional[tuple[float, float]],
    y_lim: Optional[tuple[float, float]],
    auto_scale: bool = True,
) -> None:
    """Apply axis limits for histogram plot."""
    # Apply user-defined limits if provided
    if x_lim:
        ax.set_xlim(x_lim)

    if y_lim:
        ax.set_ylim(y_lim)

    # For histograms, we usually don't auto-scale if limits are manually set
    # The render_histogram function typically handles this properly


def format_histogram_ticks(
    ax: Axes,
    log_x: bool = True,
    log_y: bool = True,
    x_major_ticks: Optional[list[float]] = None,
    y_major_ticks: Optional[list[float]] = None,
) -> None:
    """Format ticks for histogram axes."""
    from matplotlib.ticker import LogLocator, AutoMinorLocator

    # X-axis ticks
    if log_x:
        if x_major_ticks:
            ax.set_xticks(x_major_ticks)
        else:
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    else:
        if x_major_ticks:
            ax.set_xticks(x_major_ticks)
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    # Y-axis ticks
    if log_y:
        if y_major_ticks:
            ax.set_yticks(y_major_ticks)
        else:
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    else:
        if y_major_ticks:
            ax.set_yticks(y_major_ticks)
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def add_distribution_fit(
    ax: Axes,
    x_data: np.ndarray,
    fit_type: str = "normal",
    color: str = "red",
    linestyle: str = "--",
    linewidth: float = 2.0,
    label: Optional[str] = None,
    num_points: int = 100,
) -> Line2D:
    """
    Add a statistical distribution fit to a histogram.

    Args:
        ax: Matplotlib axes to plot on
        x_data: Data to fit
        fit_type: Type of distribution ('normal', 'lognormal', 'exponential', etc.)
        color: Line color
        linestyle: Line style
        linewidth: Line width
        label: Line label
        num_points: Number of points to use for plotting the fit

    Returns:
        The plotted line
    """
    from scipy import stats

    # Remove any NaN values
    x_data = x_data[~np.isnan(x_data)]

    # Get distribution function based on fit_type
    if fit_type == "normal":
        dist = stats.norm
    elif fit_type == "lognormal":
        dist = stats.lognorm
    elif fit_type == "exponential":
        dist = stats.expon
    else:
        raise ValueError(f"Unsupported distribution type: {fit_type}")

    # Fit distribution to data
    params = dist.fit(x_data)

    # Get the PDF values
    if fit_type == "lognormal":
        # Handle special case for lognorm which has different parameters
        x_min, x_max = x_data.min(), x_data.max()
        x = np.linspace(x_min, x_max, num_points)
        pdf = dist.pdf(x, *params)
    else:
        x_min, x_max = x_data.min(), x_data.max()
        x = np.linspace(x_min, x_max, num_points)
        pdf = dist.pdf(x, *params)

    # Scale PDF to match histogram height
    y_lim = ax.get_ylim()[1]
    pdf_scaled = pdf * (y_lim / pdf.max())

    # Plot the fit
    line = ax.plot(
        x,
        pdf_scaled,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        label=label if label else f"{fit_type.capitalize()} Fit",
    )[0]

    return line


def add_vertical_markers(
    ax: Axes,
    positions: list[float],
    labels: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
    linestyles: Optional[list[str]] = None,
    linewidths: Optional[list[float]] = None,
    alpha: float = 0.7,
    show_in_legend: bool = True,
) -> list[Line2D]:
    """Add vertical lines at specific positions on histogram."""
    lines = []
    default_color = "black"
    default_linestyle = "--"
    default_linewidth = 1.5

    for i, pos in enumerate(positions):
        color = colors[i] if colors and i < len(colors) else default_color
        linestyle = (
            linestyles[i] if linestyles and i < len(linestyles) else default_linestyle
        )
        linewidth = (
            linewidths[i] if linewidths and i < len(linewidths) else default_linewidth
        )
        label = labels[i] if labels and i < len(labels) else None

        line = ax.axvline(
            x=pos,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label if show_in_legend else None,
        )
        lines.append(line)

    # Update legend if any line has a label and should be in legend
    if show_in_legend and labels and any(labels):
        ax.legend()

    return lines
