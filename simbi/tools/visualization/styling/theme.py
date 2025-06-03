from dataclasses import dataclass, field
from typing import Any, Sequence, Optional
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from itertools import cycle


@dataclass
class Theme:
    """Central theme configuration for visualization styling"""

    # Text styling
    font_family: str = "serif"
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    text_color: str = "black"

    # Line styling
    line_styles: Sequence[str] = field(default_factory=lambda: ["-", "--", ":", "-."])
    line_width: float = 1.5

    # Color styling
    color_maps: Sequence[str] = field(default_factory=lambda: ["viridis"])
    color_cycle: Sequence[str] = field(
        default_factory=lambda: np.array(mpl.cm.viridis(np.linspace(0, 1, 4))).tolist()
    )

    # Axis styling
    hide_spines: Sequence[str] = field(default_factory=lambda: ["top", "right"])
    grid: bool = False
    axis_below: bool = True
    axis_equal: bool = False

    # Figure styling
    fig_size: tuple[float, float] = (8, 6)
    dpi: int = 100
    transparent: bool = False

    # Special styling
    polar_style: dict[str, Any] = field(default_factory=dict)
    colorbar_style: dict[str, Any] = field(default_factory=dict)
    use_tex: bool = False
    # Background colors
    background_colors: dict[str, str] = field(
        default_factory=lambda: {
            "figure": "#ffffff",
            "axes": "#ffffff",
        }
    )

    def apply(
        self,
        nfiles: int = 1,
        nfields: int = 1,
        user_fig_size: Optional[tuple[float, float]] = None,
    ):
        """Apply theme to matplotlib global settings"""
        plt.style.use("default")  # Reset to defaults

        colormap = plt.get_cmap(next(cycle(self.color_maps)))
        nlines = nfields
        nind_curves = nfields * nfiles
        colors = np.array([colormap(k) for k in np.linspace(0.1, 0.9, 2)])
        linestyles = [x[0] for x in zip(cycle(["-", "--", ":", "-."]), range(nlines))]
        if len(colors) == len(linestyles):
            default_cycler = cycler(color=colors) + (cycler(linestyle=linestyles))
        else:
            default_cycler = cycler(linestyle=linestyles) * cycler(color=colors)

        plt.rcParams.update(
            {
                # Font settings
                "font.family": self.font_family,
                "font.size": self.font_size,
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                # Color settings
                "text.color": self.text_color,
                "axes.labelcolor": self.text_color,
                "xtick.color": self.text_color,
                "ytick.color": self.text_color,
                # Line settings
                "lines.linewidth": self.line_width,
                "axes.prop_cycle": default_cycler,
                # Figure settings
                # "figure.figsize": user_fig_size or self.fig_size,
                # "figure.dpi": self.dpi,
                "savefig.transparent": self.transparent,
                # Text rendering settings
                "text.usetex": self.use_tex,
            }
        )

    def style_axis(self, ax):
        """Apply styling to a specific axis"""
        # Hide specified spines
        for spine in self.hide_spines:
            ax.spines[spine].set_visible(False)

        # Set grid and axis below
        ax.grid(self.grid)
        ax.set_axisbelow(self.axis_below)

        # Make axis equal if specified
        if self.axis_equal:
            ax.set_aspect("equal", adjustable="box")

    def style_polar_axis(self, ax):
        """Apply styling to a polar axis"""
        ax.grid(self.polar_style.get("grid", False))
        ax.set_theta_zero_location(self.polar_style.get("zero_location", "N"))
        ax.set_theta_direction(self.polar_style.get("direction", -1))

        # Hide tick labels if specified
        if not self.polar_style.get("show_ticks", True):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
