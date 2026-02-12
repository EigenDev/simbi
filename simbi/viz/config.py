# =============================================================================
# config.py
#
# configuration models for the visualization system.
# clean separation of concerns:
#   - FigureConfig: figure-level layout and axes
#   - PlotConfig: plot type and data selection
#   - RefinementConfig: AMR/refinement pipeline settings
#   - TimeSeriesConfig: time series specific settings
#   - AnimationConfig: animation settings
#
# component-specific styling (cmap, log_scale, alpha, etc.) is handled
# entirely by component props via --config and --props.
# =============================================================================
from argparse import Namespace
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator

# re-export for backward compatibility
from .registry import PLOT_TYPE_ALIASES as _PLOT_TYPE_TO_REGISTRY  # noqa: F401
from .styling.theme import ThemeConfig
from .types import Bounds


class FigureConfig(BaseModel):
    """Figure-level configuration (layout, axes, labels)."""

    fig_size: tuple[float, float] = (8, 6)
    dpi: int = 300
    xlims: Optional[Bounds] = None
    ylims: Optional[Bounds] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    xscale: Literal["linear", "log", "symlog", "asinh"] = "linear"
    yscale: Literal["linear", "log", "symlog", "asinh"] = "linear"
    title: Optional[str] = None
    draw_bodies: bool = False
    time_scale: Optional[float] = None
    time_units: str = ""
    transparent: bool = False

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class PlotConfig(BaseModel):
    """Plot type and data configuration."""

    plot_type: str
    fields: list[str]
    ndim: int = 1
    slice: Optional[dict[str, float]] = None

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @field_validator("plot_type")
    @classmethod
    def validate_plot_type(cls, v: str) -> str:
        valid = set(_PLOT_TYPE_TO_REGISTRY.keys())
        if v not in valid:
            raise ValueError(
                f"unknown plot type '{v}'. valid: {', '.join(sorted(valid))}"
            )
        return v

    @field_validator("ndim")
    @classmethod
    def validate_ndim(cls, v: int, info: ValidationInfo) -> int:
        if v < 1 or v > 3:
            raise ValueError(f"ndim must be between 1 and 3, got {v}")
        return v


class RefinementConfig(BaseModel):
    """Configuration for AMR/refinement visualization."""

    composite_view: bool = False
    active_levels: Optional[set[int]] = None
    render_mode: Literal["polygons", "pcolormesh"] = "polygons"

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class CoordinateConfig(BaseModel):
    """Configuration for coordinate binning plots."""

    n_bins: int = 64

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class TimeSeriesConfig(BaseModel):
    """Configuration for time series plots."""

    weight: Optional[str] = None

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class AnimationConfig(BaseModel):
    """Configuration for animations."""

    total_frames: int = 1
    frame_rate: int = 30
    save_all_frames: bool = False

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @field_validator("frame_rate")
    @classmethod
    def validate_frame_rate(cls, v: int, info: ValidationInfo) -> int:
        if v <= 0:
            raise ValueError(f"frame_rate must be positive, got {v}")
        return v


class OverlayConfig(BaseModel):
    """Configuration for a single overlay layer (e.g., contour lines)."""

    field: str
    component: str = "contour"
    levels: list[float] = Field(default_factory=lambda: [1.0])
    color: str = "lightgrey"
    linewidth: float = 1.5
    linestyle: str = "--"
    alpha: float = 1.0
    filled: bool = False
    label_contours: bool = False

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float, info: ValidationInfo) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"alpha must be between 0 and 1, got {v}")
        return v


def parse_overlay_spec(
    spec: str,
    default_color: str = "white",
    default_linewidth: float = 1.5,
) -> OverlayConfig:
    """
    parse overlay specification string.

    format: FIELD:COMPONENT:LEVELS
    examples:
        "mach:contour:1.0"
        "mach:contour:1.0,2.0,3.0"
    """
    parts = spec.split(":")

    if len(parts) < 1:
        raise ValueError(
            f"invalid overlay spec: '{spec}'. expected FIELD:COMPONENT:LEVELS"
        )

    field = parts[0]
    component = parts[1] if len(parts) > 1 else "contour"
    levels_str = parts[2] if len(parts) > 2 else "1.0"

    try:
        levels = [float(x.strip()) for x in levels_str.split(",")]
    except ValueError as e:
        raise ValueError(f"invalid levels in overlay spec '{spec}': {e}")

    return OverlayConfig(
        field=field,
        component=component,
        levels=levels,
        color=default_color,
        linewidth=default_linewidth,
    )


def overlays_from_args(args: Namespace) -> list[OverlayConfig]:
    """
    parse field overlay arguments into OverlayConfig list.

    handles --field-overlay flag which can be specified multiple times.
    """
    raw_overlays = getattr(args, "field_overlays", None)
    if not raw_overlays:
        return []

    default_color = getattr(args, "overlay_color", "white")
    default_linewidth = getattr(args, "overlay_linewidth", 1.5)

    overlays: list[OverlayConfig] = []

    for overlay_group in raw_overlays:
        for spec in overlay_group:
            overlays.append(
                parse_overlay_spec(spec, default_color, default_linewidth)
            )

    return overlays


class VisualizationConfig(BaseModel):
    """Complete visualization configuration."""

    plot: PlotConfig
    figure: FigureConfig = Field(default_factory=FigureConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    coordinate: CoordinateConfig = Field(default_factory=CoordinateConfig)
    time_series: TimeSeriesConfig = Field(default_factory=TimeSeriesConfig)
    animation: AnimationConfig = Field(default_factory=AnimationConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    overlays: list[OverlayConfig] = Field(default_factory=list)

    model_config = {"frozen": True, "arbitrary_types_allowed": True}
