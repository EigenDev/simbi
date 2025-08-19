"""Formatting functions for multidimensional plots."""

from typing import Optional
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar

from ..core.types import PlotData, FieldData
from .common import apply_basic_style, set_axis_title, set_axis_grid


def format_multidim_plot_axes(
    ax: Axes,
    data: PlotData,
    field_index: int,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    equal_aspect: bool = True,
    grid: bool = False,
    show_colorbar: bool = True,
) -> None:
    """Format axes for a multidimensional plot."""
    # Apply basic styling
    apply_basic_style(ax)

    # Set title if provided
    set_axis_title(ax, title)

    # Apply labels
    apply_labels(ax, data, field_index, x_label, y_label)

    # Set aspect ratio
    if equal_aspect:
        ax.set_aspect("equal")

    # Set grid
    set_axis_grid(ax, grid)


def apply_labels(
    ax: Axes,
    data: PlotData,
    field_index: int,
    x_label: Optional[str],
    y_label: Optional[str],
) -> None:
    """Apply axis labels based on data and configuration."""
    # Get coordinate system information
    coord_system = data.coord_system.value

    # Set x-label
    if x_label:
        ax.set_xlabel(x_label)
    else:
        # Default x-label based on coordinate system
        coord_labels = {
            "cartesian": "$x$",
            "spherical": "$r$",
            "cylindrical": "$r$",
            "planar_cylindrical": "$r$",
            "axis_cylindrical": "$z$",
        }
        ax.set_xlabel(coord_labels.get(coord_system, "$x$"))

    # Set y-label
    if y_label:
        ax.set_ylabel(y_label)
    else:
        # Default y-label based on coordinate system
        coord_labels = {
            "cartesian": "$y$",
            "spherical": r"$\theta$",
            "cylindrical": "$z$",
            "planar_cylindrical": r"$\phi$",
            "axis_cylindrical": "$r$",
        }
        ax.set_ylabel(coord_labels.get(coord_system, "$y$"))


def add_colorbar(
    fig: Figure,
    ax: Axes,
    mesh: QuadMesh,
    field: FieldData,
    position: str = "right",
    size: str = "5%",
    pad: float = 0.05,
    label: Optional[str] = None,
) -> Colorbar:
    """Add a colorbar to the plot."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create axes for colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)

    # Create colorbar
    cbar = fig.colorbar(mesh, cax=cax)

    # Set label
    if label:
        cbar.set_label(label)
    elif field.name:
        cbar.set_label(field.name)

    return cbar


def format_polar_axes(
    ax: Axes,
    data: PlotData,
    field_index: int,
    r_label: Optional[str] = None,
    theta_label: Optional[str] = None,
) -> None:
    """Format axes for polar coordinate plots."""
    # Set polar-specific properties
    ax.grid(True)

    # Set labels if provided
    if r_label:
        ax.set_ylabel(r_label)

    if theta_label:
        ax.set_xlabel(theta_label)

    # Set r-axis limits if needed
    if data.fields and field_index < len(data.fields):
        field = data.fields[field_index]
        if field.domain and len(field.domain) >= 1:
            r_max = float(np.max(field.domain[0]))
            ax.set_ylim(0, r_max * 1.05)  # Add small margin


def apply_multidim_limits(
    ax: Axes,
    data: PlotData,
    field_index: int,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    auto_scale: bool = True,
) -> None:
    """Apply axis limits for multidimensional plot."""
    # Apply user-defined limits if provided
    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    # Auto-scale if requested and no user limits
    if auto_scale and not (xlim and ylim) and field_index < len(data.fields):
        field = data.fields[field_index]
        if len(field.domain) >= 2:
            x_data = field.domain[0]
            y_data = field.domain[1]

            # Get domain extents
            if not xlim and x_data.size > 0:
                x_min = float(np.min(x_data))
                x_max = float(np.max(x_data))
                x_margin = 0.05 * (x_max - x_min) if x_max > x_min else 0.1 * abs(x_max)
                ax.set_xlim(x_min - x_margin, x_max + x_margin)

            if not ylim and y_data.size > 0:
                y_min = float(np.min(y_data))
                y_max = float(np.max(y_data))
                y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1 * abs(y_max)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
