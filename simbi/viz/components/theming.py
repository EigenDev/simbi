from dataclasses import field
from typing import Any, Sequence

from .interface import ComponentProps


class ThemeProps(ComponentProps):
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

    # Axis styling
    hide_spines: Sequence[str] = field(default_factory=lambda: ["top", "right"])
    grid: bool = False
    axis_below: bool = True
    axis_equal: bool = False

    # Figure styling
    fig_size: tuple[float, float] = (8, 6)
    dpi: int = 300
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
