from dataclasses import field
from typing import Any, Sequence

from .interface import ComponentProps


class ThemeProps(ComponentProps):
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
