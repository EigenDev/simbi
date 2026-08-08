from dataclasses import dataclass, field, fields
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


@dataclass(frozen=True)
class ThemeConfig:
    """Central theme configuration for visualization styling"""

    # text styling
    font_family: str = "serif"
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    text_color: str = "black"

    # line styling
    line_styles: Sequence[str] = field(
        default_factory=lambda: ["-", "--", ":", "-."]
    )
    line_width: float = 1.5

    # color styling
    color_map: str = "viridis"

    # axis styling
    hide_spines: Sequence[str] = field(default_factory=lambda: ["top", "right"])
    grid: bool = False
    axis_below: bool = True
    axis_equal: bool = False

    # figure styling
    fig_size: tuple[float, float] = (8, 6)
    dpi: int = 300
    transparent: bool = False

    # special styling
    polar_style: dict[str, Any] = field(default_factory=dict)
    colorbar_style: dict[str, Any] = field(default_factory=dict)
    use_tex: bool = False
    # background colors
    background_colors: dict[str, str] = field(
        default_factory=lambda: {
            "figure": "#ffffff",
            "axes": "#ffffff",
        }
    )

    @classmethod
    def from_mapping(cls, data: Any) -> "ThemeConfig":
        """Build a ThemeConfig from a mapping-like object.

        This helper accepts dicts, pydantic models, or other mapping-like objects
        and constructs a ThemeConfig by filtering only the dataclass fields that
        ThemeConfig defines. It avoids importing ThemeProps (or other component
        props types) into styling code paths and therefore helps prevent circular
        imports when callers pass their props/dicts directly.

        Examples:
            ThemeConfig.from_mapping({"font_family": "sans-serif", "font_size": 14})
            ThemeConfig.from_mapping(theme_props_instance)
        """
        # normalize to plain dict if possible
        if not isinstance(data, dict):
            if hasattr(data, "model_dump"):
                # pydantic v2
                data = data.model_dump()
            elif hasattr(data, "dict"):
                # pydantic v1 or other mapping-like objects
                data = data.dict()
            else:
                try:
                    data = dict(data)
                except Exception:
                    raise TypeError(
                        "unsupported data type for ThemeConfig.from_mapping"
                    )

        # only use keys that match ThemeConfig fields to avoid passing unknown keys
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def rc_params(
        self, nfiles: int = 1, nfields: int = 1, overlay_mode: bool = False
    ) -> dict:
        """The matplotlib settings this theme asks for.

        These are handed to a style context around the drawing, never pushed
        into the global rcParams: a theme mutated into the global state
        outlives the figure it was chosen for and restyles every later figure
        in the same session.

        Args:
            nfiles: number of files being plotted
            nfields: number of fields being plotted
            overlay_mode: if True, use color-only cycling for same field across files.
                          linestyles cycle through fields, colors cycle through files.
        """

        base_linestyles = ["-", "--", "-.", ":"]
        colormap = plt.get_cmap(self.color_map)

        if overlay_mode:
            # overlay mode: colors cycle through files, linestyles through fields
            # cycler multiplication gives outer product: linestyle is outer, color is inner
            n_colors = max(4, nfiles)
            n_linestyles = min(nfields, len(base_linestyles))

            colors = [colormap(k) for k in np.linspace(0.1, 0.9, n_colors)]
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
            base_colors = [
                colormap(k) for k in np.linspace(0.1, 0.9, n_base_colors)
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

        return {
            # font settings
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            # color settings
            "text.color": self.text_color,
            "axes.labelcolor": self.text_color,
            "xtick.color": self.text_color,
            "ytick.color": self.text_color,
            # line settings
            "lines.linewidth": self.line_width,
            "axes.prop_cycle": default_cycler,
            "savefig.transparent": self.transparent,
            # text rendering settings
            "text.usetex": self.use_tex,
        }

    def style_axis(self, ax):
        """Apply styling to a specific axis"""
        # hide specified spines
        for spine in self.hide_spines:
            ax.spines[spine].set_visible(False)

        # set grid and axis below
        ax.grid(self.grid)
        ax.set_axisbelow(self.axis_below)

        # make axis equal if specified
        if self.axis_equal:
            ax.set_aspect("equal", adjustable="box")

    def style_polar_axis(self, ax):
        """Apply styling to a polar axis"""
        ax.grid(self.polar_style.get("grid", False))
        ax.set_theta_zero_location(self.polar_style.get("zero_location", "N"))
        ax.set_theta_direction(self.polar_style.get("direction", -1))

        # hide tick labels if specified
        if not self.polar_style.get("show_ticks", True):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
