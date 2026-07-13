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
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator

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

    model_config = {"frozen": True, "arbitrary_types_allowed": True, "extra": "forbid"}


class PlotConfig(BaseModel):
    """Plot type and data configuration."""

    plot_type: Literal["line", "multidim", "coordinate_bin", "time_series"]
    fields: list[str]
    ndim: int = 1
    slice: Optional[dict[str, float]] = None
    # field-value normalization: a numeric string (divide by the constant), or "max"/"min"
    # (divide by the field's own extremum). None = no normalization.
    norm: Optional[str] = None

    model_config = {"frozen": True, "arbitrary_types_allowed": True, "extra": "forbid"}

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

    model_config = {"frozen": True, "arbitrary_types_allowed": True, "extra": "forbid"}


class CoordinateConfig(BaseModel):
    """Configuration for coordinate binning plots."""

    n_bins: int = 64

    model_config = {"frozen": True, "arbitrary_types_allowed": True, "extra": "forbid"}


class TimeSeriesConfig(BaseModel):
    """Configuration for time series plots."""

    weight: Optional[str] = None

    model_config = {"frozen": True, "arbitrary_types_allowed": True, "extra": "forbid"}


class AnimationConfig(BaseModel):
    """Configuration for animations."""

    total_frames: int = 1
    frame_rate: int = 30
    save_all_frames: bool = False

    model_config = {"frozen": True, "arbitrary_types_allowed": True, "extra": "forbid"}

    @field_validator("frame_rate")
    @classmethod
    def validate_frame_rate(cls, v: int, info: ValidationInfo) -> int:
        if v <= 0:
            raise ValueError(f"frame_rate must be positive, got {v}")
        return v


class VisualizationConfig(BaseModel):
    """Complete visualization configuration."""

    plot: PlotConfig
    figure: FigureConfig = Field(default_factory=FigureConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    coordinate: CoordinateConfig = Field(default_factory=CoordinateConfig)
    time_series: TimeSeriesConfig = Field(default_factory=TimeSeriesConfig)
    animation: AnimationConfig = Field(default_factory=AnimationConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)

    model_config = {"frozen": True, "arbitrary_types_allowed": True, "extra": "forbid"}
