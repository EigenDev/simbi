from matplotlib.collections import QuadMesh
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from numpy.typing import NDArray


class AxisFormatter:
    """Formats plot axes based on data and configuration"""

    def format_polar_axis(self, ax, mesh, config, field_info):
        """Format a polar axis based on plot type"""
        half_sphere = mesh["x2max"] <= np.pi / 2
        nfields = len(config["plot"]["fields"])
        if half_sphere and nfields <= 2:
            theta_min = -90
            theta_max = 90
        else:
            theta_min = 0
            theta_max = 360
        ax.set_theta_zero_location("N")  # Set zero at the top
        ax.set_theta_direction(-1)
        # remove r labels
        ax.set_thetamin(theta_min)
        ax.set_thetamax(theta_max)
        ax.set_yticklabels([])  # Remove radial labels
        ax.set_xticklabels([])  # Remove angular labels
        ax.set_rmin(mesh["x1min"])
        ax.set_rmax(config["style"]["xmax"] or mesh["x1max"])
        if half_sphere:
            ax.set_position([0.1, -0.45, 0.8, 2])

    def format_cartesian_axis(self, ax, setup, config, field_info):
        """Format a Cartesian axis based on plot type"""
        plot_type = config.get("plot", {}).get("plot_type", "line")

        # Call the appropriate formatter based on plot type
        if plot_type == "line":
            self._format_line_plot(ax, setup, config, field_info)
        elif plot_type == "multidim":
            self._format_multidim_plot(ax, setup, config, field_info)
        elif plot_type == "histogram":
            self._format_histogram_plot(ax, setup, config, field_info)
        elif plot_type == "temporal":
            self._format_temporal_plot(ax, setup, config, field_info)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def _format_line_plot(self, ax, setup, config, field_info):
        """Format a line plot axis"""
        # Set labels
        ax.set_xlabel(f"${config['style']['xlabel']}$")
        if len(config["plot"]["fields"]) == 1:
            ax.set_ylabel(f"{field_info}")

        # Set title
        setup_name = config.get("plot", {}).get("setup", "Simulation")
        ax.set_title(f"{setup_name} t = {setup.get('time', 0.0):.2f}")

        # Set limits if provided
        xlims = config.get("style", {}).get("xlims", (None, None))
        ylims = config.get("style", {}).get("ylims", (None, None))
        if any(xlims):
            ax.set_xlim(xlims)
        if any(ylims):
            ax.set_ylim(ylims)

        # Show legend if needed
        if config["style"]["legend"] and len(ax.get_lines()) > 1:
            ax.legend()

    def _format_multidim_plot(self, ax, setup, config, field_info):
        """Format a multidimensional plot axis"""
        # Set labels
        proj = config["multidim"]["projection"]
        if proj == (1, 2, 3):
            xlabel = "x"
            ylabel = "y"
        elif proj == (1, 3, 2):
            xlabel = "x"
            ylabel = "z"
        else:
            xlabel = "y"
            ylabel = "z"
        ax.set_xlabel(f"${xlabel}$")
        ax.set_ylabel(f"${ylabel}$")

        # Set title with time information
        time = setup.get("time", 0.0)
        time_unit = ""

        # Apply orbital time scaling if configured
        if config.get("style", {}).get("orbital_params"):
            p = config.get("style", {}).get("orbital_params", {})
            if "separation" in p and "mass" in p:
                import math

                orbital_period = (
                    2.0
                    * math.pi
                    * math.sqrt(float(p["separation"]) ** 3 / float(p["mass"]))
                )
                time = setup["time"] / orbital_period
                time_unit = "orbit(s)"

        setup_name = config.get("plot", {}).get("setup", "Simulation")
        title = f"{setup_name} t = {time:.2f} {time_unit}"
        ax.set_title(title)

        # Set aspect ratio for 2D plots
        ax.set_aspect("equal")

        # Set limits if provided
        xlims = config.get("style", {}).get("xlims", (None, None))
        ylims = config.get("style", {}).get("ylims", (None, None))
        if any(xlims):
            ax.set_xlim(xlims)
        if any(ylims):
            ax.set_ylim(ylims)

    def _format_histogram_plot(self, ax, setup, config, field_info):
        """Format a histogram plot axis"""
        # Set labels
        ax.set_xlabel(f"${config.get('style', {}).get('xlabel', 'x')}$")
        ax.set_ylabel(f"${config.get('style', {}).get('ylabel', 'y')}$")

        # Set title with time information
        time = setup.get("time", 0.0)
        setup_name = config.get("plot", {}).get("setup", "Simulation")
        ax.set_title(f"{setup_name} t = {time:.2f}")

        # Set limits if provided
        xlims = config.get("style", {}).get("xlims", (None, None))
        ylims = config.get("style", {}).get("ylims", (None, None))
        if any(xlims):
            ax.set_xlim(xlims)
        if any(ylims):
            ax.set_ylim(ylims)

    def _format_temporal_plot(self, ax, setup, config, field_info):
        """Format a temporal plot axis"""
        # Set labels
        ax.set_xlabel(f"${config.get('style', {}).get('xlabel', 'x')}$")
        ax.set_ylabel(f"${config.get('style', {}).get('ylabel', 'y')}$")

        # Set title with time information
        time = setup.get("time", 0.0)
        setup_name = config.get("plot", {}).get("setup", "Simulation")
        ax.set_title(f"{setup_name}")

        # Set limits if provided
        xlims = config.get("style", {}).get("xlims", (None, None))
        ylims = config.get("style", {}).get("ylims", (None, None))
        if any(xlims):
            ax.set_xlim(xlims)
        if any(ylims):
            ax.set_ylim(ylims)


class ColorbarFormatter:
    """Handles colorbar creation and formatting"""

    @staticmethod
    def add_cartesian_colorbar(
        fig: plt.Figure,
        ax: plt.Axes,
        mesh: QuadMesh,
        field: str,
        config: dict[str, Any],
    ) -> None:
        """Add a colorbar to a Cartesian plot"""
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Determine side based on field index
        fields = config.get("plot", {}).get("fields", ["rho"])
        field_idx = fields.index(field) if field in fields else 0
        side = "right" if field_idx == 0 else "left"

        # Create colorbar axes
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(side, size="5%", pad=0.05)

        # Create colorbar with proper orientation
        orientation = config["style"]["colorbar_orientation"]
        cbar = fig.colorbar(mesh, cax=cax, orientation=orientation)

        # Set label
        from ...utility import get_field_str

        field_label = get_field_str(field)
        cbar.set_label(field_label)

        return cbar

    @staticmethod
    def add_polar_colorbar(
        fig: plt.Figure,
        ax: plt.Axes,
        mesh: QuadMesh,
        field: str,
        config: dict[str, Any],
        setup: dict[str, Any],
    ) -> plt.colorbar:
        """Add a colorbar to a polar plot"""
        import numpy as np

        # Get field information
        fields = config["plot"]["fields"]
        field_idx = fields.index(field) if field in fields else 0
        nfields = len(fields)

        # Get polar extent
        max_angle = setup.get("x2max", np.pi)
        half_sphere = max_angle <= np.pi / 2

        # Determine orientation
        orientation = "horizontal" if half_sphere else "vertical"

        # Get the position of the current polar axis
        polar_pos = ax.get_position()

        # Position colorbar based on orientation
        if orientation == "horizontal":
            # Center horizontally under the polar plot
            width = min(0.6, 0.78 / nfields)  # Cap width for better appearance
            x = polar_pos.x0 + (polar_pos.width - width) / 2 - 0.01
            cax = fig.add_axes([x, 0.2, width, 0.03])
        else:
            # Center vertically to the right of the polar plot
            height = 0.8 / (2 if max_angle < np.pi else 1)
            x = polar_pos.x0 + polar_pos.width + 0.05  # Right side with small padding
            y = polar_pos.y0 + (polar_pos.height - height) / 2
            cax = fig.add_axes([x, y, 0.03, height])

        # Create colorbar
        cbar = fig.colorbar(mesh, cax=cax, orientation=orientation)

        # Set label
        from ...utility import get_field_str

        field_label = get_field_str(field)
        cbar.set_label(field_label)

        return cbar
