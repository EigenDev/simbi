"""Common formatting utilities shared across different plot types."""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def apply_basic_style(ax: Axes) -> None:
    """Apply basic styling to an axis that's common across plot types."""
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_axis_title(
    ax: Axes,
    title: Optional[str],
    time: Optional[float],
    fig: Optional[Figure] = None,
    title_pos: Optional[float] = None,
) -> None:
    """Set axis title if provided."""
    if title:
        title = f"{title} {time:.2f}" if time is not None else f"{title}"
    else:
        title = f"Untitled Simbi Simulation: {time:.2f}"

    if ax.name == "polar" and fig:
        fig.suptitle(title, y=title_pos)
    else:
        ax.set_title(title)


def set_axis_grid(
    ax: Axes, show_grid: bool, linestyle: str = "--", alpha: float = 0.7
) -> None:
    """Set grid visibility and style."""
    ax.grid(show_grid, linestyle=linestyle, alpha=alpha)


def set_axis_scales(ax: Axes, log_x: bool = False, log_y: bool = False) -> None:
    """Set axis scales (linear or logarithmic)."""
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")


def calculate_margins(
    min_val: float, max_val: float, margin_percent: float = 0.05
) -> tuple[float, float]:
    """Calculate axis margins based on data range."""
    if not (np.isfinite(min_val) and np.isfinite(max_val)):
        return 0.0, 0.0

    if np.allclose(min_val, max_val, rtol=1e-10):
        # Handle case where min == max
        margin = 0.1 * abs(max_val) if max_val != 0 else 0.1
    else:
        margin = margin_percent * (max_val - min_val)

    return margin, margin


def set_axis_limits(
    ax: Axes,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    margin_percent: float = 0.05,
) -> None:
    """Set axis limits with margins."""
    # Calculate margins
    x_margin_left, x_margin_right = calculate_margins(
        x_min, x_max, margin_percent
    )
    y_margin_bottom, y_margin_top = calculate_margins(
        y_min, y_max, margin_percent
    )

    # Set limits
    if np.isfinite(x_min) and np.isfinite(x_max):
        ax.set_xlim(x_min - x_margin_left, x_max + x_margin_right)

    if np.isfinite(y_min) and np.isfinite(y_max):
        ax.set_ylim(y_min - y_margin_bottom, y_max + y_margin_top)


def set_plot_style(
    ax: Axes,
    frame_visibility: bool = True,
    tick_direction: str = "in",
    tick_position: str = "both",
    minor_ticks: bool = True,
) -> None:
    """Set general plot style properties."""
    # Frame visibility
    for spine in ax.spines.values():
        spine.set_visible(frame_visibility)

    # Tick direction
    ax.tick_params(which="both", direction=tick_direction)

    # Tick position
    if tick_position == "both":
        ax.tick_params(
            which="both", top=True, bottom=True, left=True, right=True
        )
    elif tick_position == "outside":
        ax.tick_params(
            which="both", top=False, bottom=True, left=True, right=False
        )

    # Minor ticks
    if minor_ticks:
        from matplotlib.ticker import AutoMinorLocator

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def format_date_axis(
    ax: Axes,
    date_format: str = "%Y-%m-%d",
    rotation: float = 45,
    align_labels: bool = True,
) -> None:
    """Format an axis for date display."""
    import matplotlib.dates as mdates

    # Set formatter
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

    # Rotate labels for better readability
    if rotation:
        plt.setp(ax.get_xticklabels(), rotation=rotation)

    # Align rotated labels
    if align_labels:
        plt.setp(ax.get_xticklabels(), ha="right")

    # Make sure figure adjusts to accommodate labels
    ax.figure.autofmt_xdate()


def add_text_annotations(
    ax: Axes,
    texts: list[str],
    positions: list[tuple[float, float]],
    colors: Optional[list[str]] = None,
    fontsize: Optional[float] = None,
    bbox_props: Optional[dict[str, Any]] = None,
) -> list[Any]:
    """Add text annotations to a plot."""
    annotations = []
    default_color = "black"

    for i, (text, pos) in enumerate(zip(texts, positions)):
        color = colors[i] if colors and i < len(colors) else default_color

        annotation = ax.text(
            pos[0],
            pos[1],
            text,
            color=color,
            fontsize=fontsize,
            bbox=bbox_props,
            transform=ax.transData,
        )
        annotations.append(annotation)

    return annotations
