"""Formatting functions for multidimensional plots."""

from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from ...utility import get_field_str
from ..core.config import StyleConfig
from ..core.types import Bounds, FieldData, PlotData
from .common import set_axis_title


class ColorbarFormatter:
    """Handles colorbar creation and formatting"""

    initialized: bool = False

    @staticmethod
    def add_cartesian_colorbar(
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

    @staticmethod
    def add_polar_colorbar(
        fig: Figure, ax: Axes, mesh: QuadMesh, data: PlotData
    ) -> Colorbar:
        """Add a colorbar to a polar plot"""
        field = data.fields[0]
        theta = field.domain[1]
        # Get polar extent
        max_angle = theta[-1]
        half_sphere = max_angle == 0.5 * np.pi

        # Determine orientation
        orientation = "horizontal" if half_sphere else "vertical"

        # Get the position of the current polar axis
        polar_pos = ax.get_position()
        nfields = len(data.fields)
        # Position colorbar based on orientation
        if orientation == "horizontal":
            # Center horizontally under the polar plot
            width = min(0.6, 0.78 / nfields)  # Cap width for better appearance
            x = polar_pos.x0 + (polar_pos.width - width) / 2 - 0.01
            cax = fig.add_axes([x, 0.2, width, 0.03])
        else:
            # Center vertically to the right of the polar plot
            height = 0.8 / (2 if max_angle < np.pi else 1)
            x = (
                polar_pos.x0 + polar_pos.width + 0.05
            )  # Right side with small padding
            y = polar_pos.y0 + (polar_pos.height - height) / 2
            cax = fig.add_axes([x, y, 0.03, height])

        # Create colorbar
        cbar = fig.colorbar(mesh, cax=cax, orientation=orientation)

        field_label = get_field_str(field.name)
        cbar.set_label(field_label)
        return cbar


def format_multidim_plot_axes(
    ax: Axes,
    fig: Figure,
    mesh: QuadMesh,
    data: PlotData,
    field_index: int,
    style: StyleConfig,
    is_polar: bool = False,
) -> None:
    """Format axes for a multidimensional plot."""
    title_pos = None
    if ax.name == "polar":
        theta_max = data.fields[0].domain[1][-1]
        title_pos = 0.95 if theta_max == np.pi else 0.92
    set_axis_title(
        ax,
        style.setup,
        data.time,
        fig=fig,
        title_pos=title_pos,
        time_scale=style.time_scale,
        time_units=style.time_units,
    )
    if is_polar:
        format_polar_axes(ax, data, field_index, style.xlims)
    else:
        apply_labels(ax, data, style)

    if not ColorbarFormatter.initialized:
        if is_polar:
            ColorbarFormatter.add_polar_colorbar(fig, ax, mesh, data)
            ColorbarFormatter.initialized = True
        else:
            ColorbarFormatter.add_cartesian_colorbar(
                fig, ax, mesh, data.fields[0]
            )
            ColorbarFormatter.initialized = True

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
    xlims: Optional[Bounds] = None,
) -> None:
    """Format axes for polar coordinate plots."""
    field = data.fields[field_index]
    r = field.domain[0]
    theta = field.domain[1]
    half_sphere = theta[-1] == np.pi * 0.5
    if half_sphere:
        theta_min = -90
        theta_max = 90
    else:
        theta_min = 0
        theta_max = 360

    if xlims:
        xmax = xlims.max
    else:
        xmax = None

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # remove r labels
    ax.set_thetamin(theta_min)
    ax.set_thetamax(theta_max)
    ax.set_yticklabels([])  # Remove radial labels
    ax.set_xticklabels([])  # Remove angular labels
    ax.set_rmin(r[0])
    ax.set_rmax(xmax or r[-1])
    ax.grid(False)
    if half_sphere:
        ax.set_position([0.1, -0.45, 0.8, 2])


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
