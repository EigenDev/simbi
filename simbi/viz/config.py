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
    time_scale: Optional[float] = None
    time_units: str = ""
    # when False the title carries no time stamp -- the print (publication)
    # rendering, where a figure caption owns the epoch and a baked-in
    # "t = 3.21" forces a regenerate for every draft.
    show_time: bool = True
    transparent: bool = False
    # overlay each immersed body's silhouette on the field plot (cartesian only; the
    # body signed-distance is cartesian, so it does not align with a polar/spherical plot).
    draw_bodies: bool = False
    # overlay the black-hole event horizon (and excision surface) of a curved-spacetime
    # run, read from the checkpoint metadata; a flat (minkowski) run draws nothing.
    draw_horizon: bool = False
    # scatter mass-transport tracers on the field plot; a run with no
    # `tracers` group draws nothing.
    draw_tracers: bool = False

    model_config = {"frozen": True, "arbitrary_types_allowed": True, "extra": "forbid"}

    @property
    def xlims_pinned(self) -> bool:
        """the horizontal view is fixed by the user rather than by the data.
        a mesh that moves between checkpoints tracks the data otherwise."""
        return self.xlims is not None and (
            self.xlims.min is not None or self.xlims.max is not None
        )

    @property
    def ylims_pinned(self) -> bool:
        """the vertical view is fixed by the user rather than by the data."""
        return self.ylims is not None and (
            self.ylims.min is not None or self.ylims.max is not None
        )


class PlotConfig(BaseModel):
    """Plot type and data configuration."""

    plot_type: Literal["line", "multidim", "coordinate_bin", "time_series"]
    fields: list[str]
    ndim: int = 1
    slice: Optional[dict[str, float]] = None
    # field-value normalization: a numeric string (divide by the constant), or "max"/"min"
    # (divide by the field's own extremum). None = no normalization.
    norm: Optional[str] = None
    # which data an animation's colour scale is taken from. "sequence" sweeps
    # every checkpoint and pins one scale, so colour means the same thing in
    # every frame; "frame" rescales to each frame's own extremes, which draws a
    # decaying quantity at full brightness throughout.
    color_scale: Literal["sequence", "first", "frame"] = "sequence"

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
    # the default renderer for every entry point: a quadmesh is one artist for
    # the whole field, where polygons are one per cell and cost a python loop
    # to build. refined data overrides it, since a quadmesh cannot carry cells
    # of two different sizes.
    render_mode: Literal["polygons", "pcolormesh"] = "pcolormesh"

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
