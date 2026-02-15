# =============================================================================
# theme.py
#
# theme configuration for visualization styling.
# contains ThemeConfig dataclass and predefined theme instances.
#
# usage:
#   from simbi.viz.styling.theme import ThemeConfig, THEMES
#   theme = ThemeConfig.from_name("dark")
#   theme.apply(nfiles=2, nfields=3)
# =============================================================================
from dataclasses import dataclass, field, fields
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


@dataclass(frozen=True)
class ThemeConfig:
    """Central theme configuration for visualization styling"""

    # Text styling
    font_family: str = "serif"
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    text_color: str = "black"

    # Line styling
    line_styles: Sequence[str] = field(
        default_factory=lambda: ["-", "--", ":", "-."]
    )
    line_width: float = 1.5

    # Color styling
    color_map: str = "viridis"
    color_range: tuple[float, float] = (0.1, 0.9)
    color_indices: tuple[int, ...] = ()

    # Axis styling
    hide_spines: Sequence[str] = field(default_factory=lambda: ["top", "right"])
    grid: bool = False
    axis_below: bool = True
    axis_equal: bool = False

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

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ThemeConfig":
        """build a ThemeConfig from a dict, filtering to known fields only."""
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    @classmethod
    def from_name(cls, name: str) -> "ThemeConfig":
        """build a ThemeConfig from a predefined theme name."""
        if name not in THEMES:
            valid = ", ".join(sorted(THEMES.keys()))
            raise ValueError(f"unknown theme: '{name}'. valid themes: {valid}")
        return THEMES[name]

    def apply(
        self, nfiles: int = 1, nfields: int = 1, overlay_mode: bool = False
    ):
        """Apply theme to matplotlib global settings.

        Args:
            nfiles: number of files being plotted
            nfields: number of fields being plotted
            overlay_mode: if True, use color-only cycling for same field across files.
                          linestyles cycle through fields, colors cycle through files.
        """
        plt.style.use("default")

        base_linestyles = ["-", "--", "-.", ":"]
        colormap = plt.get_cmap(self.color_map)
        is_discrete = colormap.N <= 20

        if overlay_mode:
            # overlay mode: colors cycle through files, linestyles through fields
            # cycler multiplication gives outer product: linestyle is outer, color is inner
            n_colors = max(4, nfiles)
            n_linestyles = min(nfields, len(base_linestyles))

            if self.color_indices:
                colors = [
                    colormap(ii % colormap.N) for ii in self.color_indices
                ]
                # pad to n_colors if fewer indices than needed
                while len(colors) < n_colors:
                    colors.append(colors[len(colors) % len(self.color_indices)])
            elif is_discrete:
                colors = [colormap(ii % colormap.N) for ii in range(n_colors)]
            else:
                clo, chi = self.color_range
                colors = [colormap(k) for k in np.linspace(clo, chi, n_colors)]
            linestyles = base_linestyles[:n_linestyles]

            # linestyle * color means: for each linestyle, cycle through all colors
            default_cycler = cycler(linestyle=linestyles) * cycler(color=colors)
        else:
            # standard mode: all properties advance in lockstep
            nlines = nfields * nfiles
            base_markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

            n_base_colors = max(
                4, (nlines + len(base_linestyles) - 1) // len(base_linestyles)
            )
            if self.color_indices:
                base_colors = [
                    colormap(ii % colormap.N) for ii in self.color_indices
                ]
                while len(base_colors) < n_base_colors:
                    base_colors.append(
                        base_colors[len(base_colors) % len(self.color_indices)]
                    )
            elif is_discrete:
                base_colors = [
                    colormap(ii % colormap.N) for ii in range(n_base_colors)
                ]
            else:
                clo, chi = self.color_range
                base_colors = [
                    colormap(k) for k in np.linspace(clo, chi, n_base_colors)
                ]

            colors = []
            linestyles = []
            markers = []
            for i in range(nlines):
                linestyle_idx = i % len(base_linestyles)
                color_idx = i % len(base_colors)
                marker_idx = i % len(base_markers)

                linestyles.append(base_linestyles[linestyle_idx])
                colors.append(base_colors[color_idx])
                markers.append(base_markers[marker_idx])

            default_cycler = (
                cycler(color=colors) + cycler(linestyle=linestyles)
                # + cycler(marker=markers)
            )

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


# =============================================================================
# predefined themes
# =============================================================================

default_theme = ThemeConfig(
    font_family="serif",
    font_size=12,
    title_size=14,
    label_size=12,
    text_color="black",
    line_styles=["-", "--", ":", "-."],
    line_width=1.5,
    color_map="viridis",
    hide_spines=["top", "right"],
    grid=False,
    polar_style={
        "grid": False,
        "zero_location": "N",
        "direction": -1,
        "show_ticks": True,
    },
    use_tex=False,
)

dark_theme = ThemeConfig(
    font_family="sans-serif",
    font_size=12,
    title_size=14,
    label_size=12,
    text_color="white",
    line_styles=["-", "--", ":", "-."],
    line_width=1.8,
    color_map="plasma",
    hide_spines=[],
    grid=False,
    background_colors={
        "figure": "#1e1e1e",
        "axes": "#1e1e1e",
    },
    use_tex=False,
)

scientific_theme = ThemeConfig(
    font_family="Times New Roman",
    font_size=10,
    title_size=12,
    label_size=10,
    text_color="black",
    line_styles=["-", "--", ":", "-."],
    line_width=1.2,
    color_map="viridis",
    hide_spines=["top", "right"],
    grid=False,
    axis_below=True,
    use_tex=True,
)

# theme registry
THEMES: dict[str, ThemeConfig] = {
    "default": default_theme,
    "dark": dark_theme,
    "scientific": scientific_theme,
}
