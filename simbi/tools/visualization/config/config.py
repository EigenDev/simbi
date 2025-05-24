from dataclasses import dataclass, field
from typing import Optional, Sequence, Any
from pathlib import Path
from enum import Enum
from itertools import cycle


class PlotType(Enum):
    LINE = "line"
    MULTIDIM = "multidim"
    HISTOGRAM = "hist"
    TEMPORAL = "temporal"


@dataclass
class PlotGroup:
    """Basic plot configuration"""

    setup: str
    files: Sequence[Path]
    plot_type: PlotType
    fields: Sequence[str] = field(default_factory=lambda: ["rho"])
    ndim: int = 1
    cartesian: bool = False
    xmax: Optional[float] = None
    save_as: Optional[str] = None
    kind: str = "snapshot"
    nplots: int = 1
    powerfit: Optional[float] = None
    hist_type: str = "kinetic"
    weight: str = "rho"
    extension: Optional[str] = None


@dataclass
class StyleGroup:
    """Plot style configuration"""

    color_maps: Sequence[str] = field(default_factory=lambda: cycle(["viridis"]))
    log: bool = False
    semilogx: bool = False
    semilogy: bool = False
    units: bool = False
    fig_dims: tuple[float, float] = (10, 6)
    legend: bool = True
    legend_loc: Optional[str] = None
    labels: Optional[Sequence[str]] = None
    xlims: tuple[Optional[float], Optional[float]] = (None, None)
    ylims: tuple[Optional[float], Optional[float]] = (None, None)
    xmax: Optional[float] = None
    power: float = 1.0
    transparent: bool = False
    black_background: bool = False
    annotation_loc: str = "upper right"
    annotation_text: str = ""
    annotation_anchor: tuple[float, float] = (1.0, 1.0)
    bipolar: bool = False
    use_tex: bool = False
    color_range: list[tuple[float, float]] = cycle([(None, None)])
    print: bool = False
    pictorial: bool = False
    scale_downs: Sequence[float] = field(default_factory=lambda: [1.0])
    time_modulus: float = 1.0
    normalize: bool = False
    bbox_kind: str = "tight"
    font_color: str = "black"
    show_colorbar: bool = True
    reverse_colormap: bool = False
    colorbar_orientation: str = "vertical"
    split_into_subplots: bool = False
    xlabel: str = "x"
    ylabel: str = "y"
    dpi: int = 300
    orbital_params: Optional[dict[str, float]] = None
    nlinestyles: Optional[int] = None
    draw_immersed_bodies: bool = False


@dataclass
class MultidimGroup:
    """Multidimensional plot settings"""

    projection: tuple[int, int, int] = (1, 2, 3)
    box_depth: float = 0.0
    bipolar: bool = False
    patches: int = 1
    slice_along: Optional[str] = None
    coords: dict[str, list[str]] = field(
        default_factory=lambda: {"xj": ["0.0"], "xk": ["0.0"]}
    )


@dataclass
class AnimationGroup:
    """Animation configuration"""

    frame_rate: int = 30
    pan_speed: Optional[float] = None
    extent: Optional[float] = None


@dataclass
class Config:
    """Master configuration"""

    plot: PlotGroup
    style: StyleGroup
    multidim: MultidimGroup = field(default_factory=MultidimGroup)
    animation: AnimationGroup = field(default_factory=AnimationGroup)

    @classmethod
    def from_dict(cls, param_dict: dict[str, Any]):
        return cls(
            plot=PlotGroup(**param_dict),
            style=StyleGroup(**param_dict),
            multidim=MultidimGroup(**param_dict),
            animation=AnimationGroup(**param_dict),
        )

    @classmethod
    def from_parser(cls, parser):
        """Create config from parser arguments"""
        args = vars(parser.parse_args())
        return cls(
            plot=PlotGroup(**args),
            style=StyleGroup(**args),
            multidim=MultidimGroup(**args),
            animation=AnimationGroup(**args),
        )

    def validate(self) -> None:
        """Validate all configuration settings"""
        if self.plot.ndim < 1:
            raise ValueError("ndim must be >= 1")
        if self.plot.plot_type == PlotType.MULTIDIM and self.plot.ndim < 2:
            raise ValueError("Multidim plots require ndim >= 2")
        if self.animation.frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
