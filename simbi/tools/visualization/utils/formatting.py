from typing import Optional, List, Tuple
from dataclasses import dataclass
from matplotlib.collections import QuadMesh
import matplotlib.colors as mcolors
from matplotlib.colorbar import Colorbar, ColorbarBase
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from ..config import Config
from matplotlib.axes import Axes
from typing import Any
from cycler import cycler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import pi
from ...utility import get_field_str


@dataclass
class PlotTextStyle:
    """Text style configuration"""

    fontsize: int = 12
    color: str = "black"
    fontweight: str = "bold"
    fontname: str = "Times New Roman"
    alpha: float = 1.0
    use_tex: bool = False
    font_family: str = "serif"

    def __init__(self, config: dict[str, "Config"]) -> None:
        """Initialize text style with optional overrides"""
        # Apply overrides if provided
        for key, value in vars(config["style"]).items():
            if hasattr(self, key):
                setattr(self, key, value)

        colormap = plt.get_cmap(next(config["style"].color_maps))
        if config["plot"].nplots == 1:
            nind_curves = max(
                len(config["plot"].fields),
                len(config["plot"].files),
                # len(config['plot'].cutoffs),
                len(config["multidim"].coords["xj"].split(","))
                * len(config["multidim"].coords["xk"].split(",")),
            )
        else:
            nind_curves = config["plot"].nplots // len(config["plot"].files)

        colors = np.array([colormap(k) for k in np.linspace(0.1, 0.9, nind_curves)])
        linestyles = [
            x[0]
            for x in zip(
                cycle(["-", "--", ":", "-."]), range(len(config["plot"].fields))
            )
        ]
        default_cycler = cycler(color=colors) * (cycler(linestyle=linestyles))

        # Update matplotlib params
        plt.rcParams.update(
            {
                "font.size": self.fontsize,
                "text.color": self.color,
                "font.family": self.fontname,
                "text.usetex": self.use_tex,
                "font.family": self.font_family,
                "legend.fontsize": self.fontsize,
                "axes.labelsize": self.fontsize,
                "axes.titlesize": self.fontsize,
                "xtick.labelsize": self.fontsize,
                "ytick.labelsize": self.fontsize,
                "figure.titlesize": self.fontsize,
                "xtick.color": self.color,
                "ytick.color": self.color,
                "axes.labelcolor": self.color,
                "axes.edgecolor": self.color,
                "axes.facecolor": "white",
                "figure.facecolor": "white",
                "axes.prop_cycle": default_cycler,
            }
        )


@dataclass
class PlotStyle:
    """Plot style configuration"""

    spines: List[str] = ("top", "right")
    grid: bool = False
    axis_below: bool = True


class PlotFormatter:
    """Handles plot formatting"""

    def setup__plot(self, ax: Axes, config: Config) -> None:
        """Apply plot configuration and style"""
        if not config.grid.cartesian:
            self.setup_polar_axes(ax, config['style'].show_ticks)
        else:
            self.setup_axis_style(ax, PlotStyle())
        self.setup_color_map()

    def setup_color_map(self) -> None:
        self.color_maps = [
            (
                plt.get_cmap(cmap).reversed()
                if self.config['style'].rcmap
                else plt.get_cmap(cmap)
            )
            for cmap in self.config['style'].cmap
        ]
        self.color_maps = cycle(self.color_maps)

    @staticmethod
    def setup_axis_style(
        ax: Axes,
        style: PlotStyle = PlotStyle(),
        xlim: Optional[tuple[float, float]] = (None, None),
        ylim: Optional[tuple[float, float]] = (None, None),
    ) -> None:
        """Configure axis style"""
        for spine in style.spines:
            ax.spines[spine].set_visible(False)
        ax.grid(style.grid)
        ax.set_axisbelow(style.axis_below)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    @staticmethod
    def set_axes_properties(
        fig: plt.Figure, ax: Axes, setup: dict[str, Any], config: dict[str, Any]
    ) -> None:
        """Set axes properties"""
        field_string = get_field_str(config["plot"].fields)
        if ax.name == "polar":
            half_sphere = setup["x2max"] == 0.5 * pi
            if half_sphere:
                theta_min = -90
                theta_max = +90
            else:
                theta_min = 0
                theta_max = (360 / pi) * setup["x2max"]
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_thetamin(theta_min)
            ax.set_thetamax(theta_max)
            ax.set_rmin(setup["x1min"])
            ax.set_rmax(config["style"].xmax or setup["x1max"])

            kwargs = {"y": 0.8 if half_sphere else 1.03}
            fig.suptitle(f"{config['plot'].setup} t = {setup['time']:.2f}", **kwargs)
        else:
            if config["plot"].plot_type == "temporal":
                weight_string = get_field_str(
                    [config["plot"].weight], normalized=False
                ).replace(r"$", "")
                ax.set_xlabel(r"$t$")
                if len(config["plot"].fields) == 1:
                    ax.set_ylabel(
                        rf"$\langle$ {field_string}$~\rangle_{weight_string}$"
                    )
                else:
                    ax.legend()
                ax.set_title(f"{config['plot'].setup}")
            elif setup["dimensions"] == 1 or config["multidim"].slice_along:
                ax.set_xlabel(f"${config['style'].xlabel}$")
                if len(config["plot"].fields) == 1:
                    ax.set_ylabel(f"{field_string}")
                else:
                    ax.legend()
                ax.set_title(f"{config['plot'].setup} t = {setup['time']:.2f}")
            elif config["plot"].plot_type == "histogram":
                ax.set_xlabel(rf"$\Gamma\beta$")
                if config["plot"].hist_type == "kinetic":
                    ax.set_ylabel(r"$E_k (> \Gamma\beta)$")
                elif config["plot"].hist_type == "enthalpy":
                    ax.set_ylabel(r"$h (> \Gamma\beta)$")
                else:
                    ax.set_ylabel(r"$M (> \Gamma\beta)$")
                ax.set_title(f"{config['plot'].setup} t = {setup['time']:.2f}")
                # ax.legend()
            else:
                ax.set_xlabel(rf"${config['style'].xlabel}$")
                ax.set_ylabel(rf"${config['style'].ylabel}$")
                ax.set_title(f"{config['plot'].setup} t = {setup['time']:.2f}")

    @staticmethod
    def format_cartesian_colorbar(
        fig: plt.Figure,
        plot: QuadMesh,
        ax: Axes,
        orientation: str = "vertical",
        label: str = "",
        side: str = "right",
    ) -> Colorbar:
        """Create and format colorbar"""
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(side, size="5%", pad=0.05)

        cbar = fig.colorbar(plot, cax=cax, orientation=orientation)
        cbar.set_label(label)
        return cbar

    @staticmethod
    def format_polar_colorbar(
        fig: plt.Figure,
        plot: QuadMesh,
        nbars: int,
        max_angle: float,
        idx: int,
        label: str = "",
    ) -> ColorbarBase:
        """Setup polar colorbar based on angular extent"""
        # Determine orientation based on extent
        half_sphere = max_angle <= np.pi / 2
        orientation = "horizontal" if half_sphere else "vertical"

        if orientation == "horizontal":
            # Place under half-circle plot
            width = 0.78 / nbars
            x = 0.05 + ((nbars - 1 - idx) * (width + 0.1))
            cax = fig.add_axes([x, 0.2, width, 0.03])
        else:
            # Place beside full circle plot
            height = 0.8 / (2 if max_angle < np.pi else 1)
            x = 0.90 if idx == 0 else 0.09
            cax = fig.add_axes([x, 0.1, 0.03, height])

        cbar = fig.colorbar(plot, cax=cax, orientation=orientation)
        cbar.set_label(label)
        return cbar

    @staticmethod
    def create_label(
        field: str, coords: Optional[Tuple[float, float]] = None, units: bool = False
    ) -> str:
        """Create formatted label"""
        label = field
        if coords:
            xj, xk = coords
            label += f", $x_j={xj:.1f}$"
            if xk is not None:
                label += f", $x_k={xk:.1f}$"
        if units:
            # Add units based on field type
            pass
        return label

    @staticmethod
    def set_scale(ax: Axes, log_x: bool = False, log_y: bool = False) -> None:
        """Set axis scales"""
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

    @staticmethod
    def set_ticks(ax: Axes, ticks: Optional[List[float]] = None) -> None:
        """Set axis ticks"""
        if ticks:
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        else:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    @staticmethod
    def setup_polar_axes(ax: Axes, show_ticks: bool = True) -> None:
        """Setup polar axes"""
        ax.grid(False)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        if not show_ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
