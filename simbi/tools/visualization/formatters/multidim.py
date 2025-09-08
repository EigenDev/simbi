"""Formatting functions for multidimensional plots."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from simbi.tools.utility import get_field_str
from simbi.tools.visualization.core.config import StyleConfig

from ..core.types import FieldData, PlotData
from .common import set_axis_title


@dataclass
class CBar:
    color_bar: Optional[Colorbar] = None


cbar_state = CBar()


def format_multidim_plot_axes(
    ax: Axes,
    fig: Figure,
    mesh: QuadMesh,
    data: PlotData,
    field_index: int,
    style: StyleConfig,
) -> None:
    """Format axes for a multidimensional plot."""

    set_axis_title(ax, style.setup, data.time)
    apply_labels(ax, data, style)
    if not cbar_state.color_bar:
        cbar_state.color_bar = add_colorbar(
            fig, ax, mesh, data.fields[field_index]
        )

    # Set aspect ratio
    if style.equal_aspect:
        ax.set_aspect("equal")


def apply_labels(ax: Axes, data: PlotData, style: StyleConfig) -> None:
    """Apply axis labels based on data and configuration."""
    # Get coordinate system information
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
            "planar_cylindrical": "$r$",
            "axis_cylindrical": "$z$",
        }
        ax.set_xlabel(coord_labels.get(coord_system, "$x$"))

    # Set y-label
    if style.y_label:
        ax.set_ylabel(style.y_label)
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
        cbar.set_label(get_field_str(field.name))

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
                x_margin = (
                    0.05 * (x_max - x_min)
                    if x_max > x_min
                    else 0.1 * abs(x_max)
                )
                ax.set_xlim(x_min - x_margin, x_max + x_margin)

            if not ylim and y_data.size > 0:
                y_min = float(np.min(y_data))
                y_max = float(np.max(y_data))
                y_margin = (
                    0.05 * (y_max - y_min)
                    if y_max > y_min
                    else 0.1 * abs(y_max)
                )
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
