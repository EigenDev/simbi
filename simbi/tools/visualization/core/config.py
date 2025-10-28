"""Configuration models for the visualization system."""

from itertools import cycle
from typing import Iterator, Literal, Optional, Sequence

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from ..styling.theme import ThemeConfig
from .types import Bounds, ColorRange


class StyleConfig(BaseModel):
    """
    Styling configuration for visualizations.

    Attributes:
        fig_size: Figure size in inches (width, height)
        dpi: Dots per inch (resolution)
        xlims: Optional x-axis limits (min, max)
        ylims: Optional y-axis limits (min, max)
        color_range: Optional color range for visualizations
        legend: Whether to show legend
        cmap: Colormap name
        units: Whether to use physical units
        log: Whether to use logarithmic scale
    """

    fig_size: tuple[float, float] = (8, 6)
    dpi: int = 300
    xlims: Optional[Bounds] = None
    ylims: Optional[Bounds] = None
    color_range: Iterator[ColorRange] = cycle([ColorRange(min=None, max=None)])
    legend: bool = True
    legend_loc: Optional[str] = "upper left"
    cmap: Iterator[str] = cycle(["viridis"])
    units: bool = False
    log: bool = False
    semilogx: bool = False
    semilogy: bool = False
    setup: str = "Simulation"
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    equal_aspect: bool = True
    value_scale: Optional[Sequence[float]] = None
    draw_bodies: bool = False
    time_scale: Optional[float] = None
    time_units: str = ""

    model_config = {
        "frozen": True,  # Make instances immutable
        "arbitrary_types_allowed": True,
    }


class MultidimConfig(BaseModel):
    """
    Configuration for multidimensional plots.

    Attributes:
        slice_along: Axis to slice along (e.g., "x1", "x2", "x3")
        projection: Projection indices for 3D data (base-1)
        bipolar: Whether to use bipolar projection
        coords: Coordinates for slices
    """

    slice_along: Optional[str] = None
    slice_position: float = 0.0
    projection: tuple[int, int, int] = (1, 2, 3)
    bipolar: bool = False
    coords: dict[str, list[float]] = Field(default_factory=dict)
    composite_view: bool = False
    active_levels: Optional[set[int]] = None

    model_config = {
        "frozen": True,  # Make instances immutable
        "arbitrary_types_allowed": True,
    }

    @field_validator("projection")
    @classmethod
    def validate_projection(
        cls, v: tuple[int, int, int], info: ValidationInfo
    ) -> tuple[int, int, int]:
        """Validate projection indices are between 1 and 3."""
        for idx in v:
            if idx < 1 or idx > 3:
                raise ValueError(
                    f"Projection indices must be between 1 and 3, got {v}"
                )
        if len(set(v)) != 3:
            raise ValueError(f"Projection indices must be unique, got {v}")
        return v

    @field_validator("composite_view", "active_levels")
    @classmethod
    def validate_composite_and_levels(cls, v, info: ValidationInfo):
        """Validate that active_levels is set if composite_view is True."""
        composite_view = info.data.get("composite_view", False)
        active_levels = info.data.get("active_levels", None)

        if composite_view and (
            active_levels is None or len(active_levels) == 0
        ):
            raise ValueError(
                "active_levels must be set and non-empty when composite_view is True"
            )
        return v


class PlotConfig(BaseModel):
    """
    Plot configuration.

    Attributes:
        plot_type: Type of plot
        fields: Field names to visualize
        setup: Setup name/title
        ndim: Number of dimensions to visualize
    """

    plot_type: Literal["line", "multidim", "histogram", "time_series"]
    fields: Sequence[str]
    ndim: int = 1

    model_config = {
        "frozen": True,  # Make instances immutable
        "arbitrary_types_allowed": True,
    }

    @field_validator("ndim")
    @classmethod
    def validate_ndim(cls, v: int, info: ValidationInfo) -> int:
        """Validate ndim is between 1 and 3."""
        if v < 1 or v > 3:
            raise ValueError(f"ndim must be between 1 and 3, got {v}")
        return v


class HistogramConfig(BaseModel):
    """
    Configuration for histogram plots.

    Attributes:
        hist_type: Type of histogram
        nbins: Number of bins
        powerfit: Whether to fit a power law
    """

    hist_type: Literal["kinetic", "enthalpy", "mass", "energy"] = "kinetic"
    nbins: int = 128
    powerfit: bool = False

    model_config = {
        "frozen": True,  # Make instances immutable
        "arbitrary_types_allowed": True,
    }

    @field_validator("nbins")
    @classmethod
    def validate_nbins(cls, v: int, info: ValidationInfo) -> int:
        """Validate nbins is positive."""
        if v <= 0:
            raise ValueError(f"nbins must be positive, got {v}")
        return v


class time_seriesConfig(BaseModel):
    """
    Configuration for time_series plots.

    Attributes:
        weight: Field to use for weighting
        body_id: ID of body to plot (for accretion data)
        single_file_mode: Whether to use single file mode
    """

    weight: Optional[str] = None
    body_id: Optional[str] = None
    single_file_mode: bool = False

    model_config = {
        "frozen": True,  # Make instances immutable
        "arbitrary_types_allowed": True,
    }


class AnimationConfig(BaseModel):
    """
    Configuration for animations.

    Attributes:
        frame_rate: Frames per second
        save_all_frames: Whether to save all frames
    """

    frame_rate: int = 30
    save_all_frames: bool = False

    model_config = {
        "frozen": True,  # Make instances immutable
        "arbitrary_types_allowed": True,
    }

    @field_validator("frame_rate")
    @classmethod
    def validate_frame_rate(cls, v: int, info: ValidationInfo) -> int:
        """Validate frame rate is positive."""
        if v <= 0:
            raise ValueError(f"frame_rate must be positive, got {v}")
        return v


class VisualizationConfig(BaseModel):
    """
    Complete visualization configuration.

    Attributes:
        plot: Basic plot configuration
        style: Styling configuration
        multidim: Multidimensional plot configuration
        histogram: Histogram configuration
        time_series: time_series plot configuration
        animation: Animation configuration
    """

    plot: PlotConfig
    style: StyleConfig = Field(default_factory=StyleConfig)
    multidim: MultidimConfig = Field(default_factory=MultidimConfig)
    histogram: HistogramConfig = Field(default_factory=HistogramConfig)
    time_series: time_seriesConfig = Field(default_factory=time_seriesConfig)
    animation: AnimationConfig = Field(default_factory=AnimationConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)

    model_config = {
        "frozen": True,  # Make instances immutable
        "arbitrary_types_allowed": True,
    }
