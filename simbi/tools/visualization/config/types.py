from typing import Optional, Sequence, TypedDict
from typing_extensions import Literal, NotRequired


class PlotConfig(TypedDict):
    """Plot configuration type definition"""

    files: Sequence[str]
    fields: Sequence[str]
    plot_type: Literal["line", "multidim", "histogram", "temporal"]
    setup: str
    ndim: int

    # Optional fields with default values
    weight: NotRequired[Optional[str]]
    hist_type: NotRequired[str]
    powerfit: NotRequired[bool]


class StyleConfig(TypedDict):
    """Style configuration type definition"""

    cmap: NotRequired[Sequence[str]]
    log: NotRequired[bool]
    power: NotRequired[float]
    fig_dims: NotRequired[tuple[float, float]]
    dpi: NotRequired[int]
    legend: NotRequired[bool]
    labels: NotRequired[Sequence[str]]
    xlims: NotRequired[tuple[Optional[float], Optional[float]]]
    ylims: NotRequired[tuple[Optional[float], Optional[float]]]
    xlabel: NotRequired[str]
    ylabel: NotRequired[str]
    show_colorbar: NotRequired[bool]
    colorbar_orientation: NotRequired[str]
    draw_immersed_bodies: NotRequired[bool]
    orbital_params: NotRequired[dict[str, float]]


class MultidimConfig(TypedDict):
    """Multidimensional plot configuration"""

    projection: NotRequired[Sequence[int]]
    box_depth: NotRequired[float]
    bipolar: NotRequired[bool]
    slice_along: NotRequired[Optional[str]]
    coords: NotRequired[dict[str, Sequence[str]]]


class AnimationConfig(TypedDict):
    """Animation configuration"""

    frame_rate: NotRequired[int]


class VisualizationConfig(TypedDict):
    """Complete visualization configuration"""

    plot: PlotConfig
    style: NotRequired[StyleConfig]
    multidim: NotRequired[MultidimConfig]
    animation: NotRequired[AnimationConfig]
