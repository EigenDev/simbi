"""Formatting functions for line plots"""

from typing import Optional

from matplotlib.axes import Axes

from simbi.tools.utility import get_field_str
from simbi.tools.visualization.core.config import StyleConfig

from ..core.types import FieldData, PlotData
from .common import apply_basic_style, set_axis_title


def format_line_plot_axes(
    ax: Axes, data: PlotData, field_index: int, style: StyleConfig
) -> None:
    """Format axes for a line plot."""
    apply_basic_style(ax)
    set_axis_title(ax, style.setup, data.time)
    apply_labels(ax, data, field_index, style)
    apply_line_limits(ax, data, field_index)


def apply_labels(
    ax: Axes, data: PlotData, field_index: int, style: StyleConfig
) -> None:
    """Apply axis labels based on data and configuration."""
    # Get coordinate system information
    if data.coord_system is None:
        coord_system = "cartesian"
    else:
        coord_system = data.coord_system.value

    # Set x-label
    if style.x_label:
        ax.set_xlabel(style.x_label)
    else:
        # Default x-label based on coordinate system
        coord_labels = {
            "cartesian": "$x$",
            "spherical": "$r$",
            "cylindrical": "$r$",
        }
        ax.set_xlabel(coord_labels.get(coord_system, "$x$"))

    # Set y-label
    if style.y_label:
        ax.set_ylabel(style.y_label)
    else:
        # if the user is only plotting
        # one field, then use the field name as the y-label
        # otherwise, the field names will appear in the legend
        if len(data.fields) == 1:
            field = data.fields[field_index]
            ax.set_ylabel(get_field_str(field.name))
        else:
            ax.legend(loc=style.legend_loc)


def auto_scale_line_axes(ax: Axes, data: PlotData, field_index: int) -> None:
    """Auto-scale axes for line plot."""
    if field_index >= len(data.fields):
        return

    if len(data.fields) > 1:
        return

    field: FieldData = data.fields[field_index]

    # Auto-scale x-axis based on coordinate data
    if field.domain is not None and len(field.domain) > 0:
        x_data = field.domain[0]
        ax.set_xlim(float(min(x_data) * 0.95), float(max(x_data)))

    # Auto-scale y-axis based on field data
    if field.values is not None and len(field.values) > 0:
        y_data = field.values
        ax.set_ylim(float(min(y_data) * 0.95), float(max(y_data) * 1.05))


def apply_line_limits(
    ax: Axes,
    data: PlotData,
    field_index: int,
    x_lim: Optional[tuple[float, float]] = None,
    y_lim: Optional[tuple[float, float]] = None,
    auto_scale: bool = True,
) -> None:
    """Apply axis limits for line plot."""
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)

    if auto_scale and not (x_lim and y_lim):
        auto_scale_line_axes(ax, data, field_index)
