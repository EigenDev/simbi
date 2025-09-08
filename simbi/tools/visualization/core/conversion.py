"""Convert command line arguments to typed configuration objects."""

from argparse import Namespace
from itertools import cycle
from typing import Literal, Optional

from typing_extensions import Any

from ...utility import get_dimensionality
from ..components.multidim import MultidimPlotProps
from ..styling import ThemeManager
from ..styling.theme import ThemeConfig
from .config import (
    AnimationConfig,
    HistogramConfig,
    MultidimConfig,
    PlotConfig,
    StyleConfig,
    TemporalConfig,
    VisualizationConfig,
)
from .types import Bounds, ColorRange


def pair_to_bounds(pair: list[float] | None) -> Optional[Bounds]:
    """Convert a [min, max] pair to Bounds object."""
    if not pair or not any(x is not None for x in pair):
        return None
    return Bounds(min=pair[0], max=pair[1])


def tuple_to_color_range(
    pair: list[tuple[float, float]] | None,
) -> Optional[ColorRange]:
    """Convert a (min, max) tuple to ColorRange object."""
    if not pair or pair == (None, None):
        return None

    return ColorRange(min=pair[0][0], max=pair[0][1])


def first_from_cycle(value):
    """Get first value from a cycle object."""
    if hasattr(value, "__next__"):
        return next(value)
    return value


def validate_plot_type(
    plot_type: str | None, files: list[str]
) -> Literal["line", "multidim", "temporal", "histogram"]:
    """Validate and auto-detect plot type if not specified."""
    if plot_type:
        return plot_type  # type: ignore[return-value]

    try:
        from ....tools.utility import get_dimensionality

        ndim = get_dimensionality(files) if files else 1
        return "line" if ndim == 1 else "multidim"
    except ImportError:
        return "multidim"


def plot_config_from_args(args: Namespace) -> PlotConfig:
    """Build PlotConfig from command line arguments."""
    files = args.files
    plot_type = args.plot_type or None

    return PlotConfig(
        plot_type=validate_plot_type(plot_type, files),
        fields=getattr(args, "fields", ["rho"]),
        ndim=get_dimensionality(files),
    )


def style_config_from_args(args: Namespace) -> StyleConfig:
    """Build StyleConfig from command line arguments."""
    xlims = pair_to_bounds(getattr(args, "xlims", None))
    ylims = pair_to_bounds(getattr(args, "ylims", None))
    raw_color_range = getattr(args, "color_range", [(None, None)])
    color_range = cycle(
        list(map(lambda c: ColorRange(min=c[0], max=c[1]), raw_color_range))
    )
    cmap = cycle(getattr(args, "cmap", ["viridis"]))

    return StyleConfig(
        fig_size=getattr(args, "fig_size") or (5, 4),
        dpi=getattr(args, "dpi", 300),
        xlims=xlims,
        ylims=ylims,
        color_range=color_range,
        legend=not getattr(args, "no_legend", False),
        cmap=cmap,
        units=getattr(args, "units", False),
        log=getattr(args, "log", False),
        setup=getattr(args, "setup", "Unittled Simulation"),
        legend_loc=getattr(args, "legend_loc", "upper right"),
        semilogx=getattr(args, "semilogx", False),
        semilogy=getattr(args, "semilogy", False),
    )


def multidim_config_from_args(args: Namespace) -> MultidimConfig:
    """Build MultidimConfig from command line arguments."""
    coords = getattr(args, "coords", None)
    if coords is None:
        coords = {"xj": [0.0], "xk": [0.0]}

    return MultidimConfig(
        slice_along=getattr(args, "slice_along", None),
        slice_position=getattr(args, "slice_position", 0.0),
        projection=getattr(args, "projection", (1, 2, 3)),
        bipolar=getattr(args, "bipolar", False),
        coords=coords,
    )


def multidim_props_from_args(
    args: dict[str, Any], field_index: int, cmap: str, color_range: ColorRange
) -> MultidimPlotProps:
    return MultidimPlotProps(
        field_index=field_index,
        color_range=color_range,
        cmap=cmap,
        log_scale=args["log"],
        power=args["power"],
        shading=args.get("shading", "auto"),
        alpha=args.get("alpha", 1.0),
        projection=args["projection"],
    )


def histogram_config_from_args(args: Namespace) -> HistogramConfig:
    """Build HistogramConfig from command line arguments."""
    return HistogramConfig(
        hist_type=getattr(args, "hist_type", "kinetic"),
        nbins=getattr(args, "nbins", 128),
        powerfit=getattr(args, "powerfit", False),
    )


def temporal_config_from_args(args: Namespace) -> TemporalConfig:
    """Build TemporalConfig from command line arguments."""
    return TemporalConfig(
        weight=getattr(args, "weight", None),
        body_id=getattr(args, "body_id", None),
        single_file_mode=getattr(args, "single_file_mode", False),
    )


def animation_config_from_args(args: Namespace) -> AnimationConfig:
    """Build AnimationConfig from command line arguments."""
    return AnimationConfig(
        frame_rate=getattr(args, "frame_rate", 30),
        save_all_frames=getattr(args, "save_all_frames", False),
    )


def theme_config_from_args(args: Namespace) -> ThemeConfig:
    """Build ThemeConfig from command line arguments."""
    return ThemeManager.get_theme(getattr(args, "theme", "default"))


def config_from_args(args: Namespace) -> VisualizationConfig:
    """Build complete VisualizationConfig from command line arguments."""
    return VisualizationConfig(
        plot=plot_config_from_args(args),
        style=style_config_from_args(args),
        multidim=multidim_config_from_args(args),
        histogram=histogram_config_from_args(args),
        temporal=temporal_config_from_args(args),
        animation=animation_config_from_args(args),
        theme=theme_config_from_args(args),
    )


def is_animation_requested(args: Namespace) -> bool:
    """Check if animation was requested."""
    return (
        getattr(args, "animate", False)
        or getattr(args, "kind", "snapshot") == "movie"
    )


def should_show_plot(args: Namespace) -> bool:
    """Determine if plot should be displayed."""
    return not getattr(args, "no_show", False)


def get_save_path(args: Namespace) -> Optional[str]:
    """Get the save path if specified."""
    return getattr(args, "save_as", None)
