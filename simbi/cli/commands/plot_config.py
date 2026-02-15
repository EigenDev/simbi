# =============================================================================
# plot_config.py
#
# converts cli arguments (argparse.Namespace) to typed viz configuration
# objects. pure CLI glue — no rendering, no data loading.
# =============================================================================
from argparse import Namespace
from typing import Literal, Optional

from simbi.viz.components.interface import ComponentProps
from simbi.viz.config import (
    AnimationConfig,
    CoordinateConfig,
    FigureConfig,
    PlotConfig,
    RefinementConfig,
    TemporalSpectrumConfig,
    TimeSeriesConfig,
    VisualizationConfig,
    overlays_from_args,
)
from simbi.viz.styling import get_theme
from simbi.viz.styling.theme import ThemeConfig
from simbi.viz.types import Bounds
from simbi.viz.utility import get_dimensionality


def _pair_to_bounds(pair: list[float] | None) -> Optional[Bounds]:
    """convert a [min, max] pair to Bounds object."""
    if not pair or not any(x is not None for x in pair):
        return None
    return Bounds(min=pair[0], max=pair[1])


def _validate_plot_type(
    plot_type: str | None, files: list[str]
) -> Literal["line", "multidim", "time_series", "coordinate_bin"]:
    """validate and auto-detect plot type if not specified."""
    if plot_type:
        return plot_type  # type: ignore[return-value]

    ndim = get_dimensionality(files) if files else 1
    return "line" if ndim == 1 else "multidim"


def plot_config_from_args(args: Namespace) -> PlotConfig:
    """build PlotConfig from cli arguments."""
    import argparse

    files = args.files
    plot_type = getattr(args, "plot_type", None)

    raw_slice = getattr(args, "slice", None)
    slice_spec: Optional[dict[str, float]] = None
    if raw_slice:
        try:
            slice_spec = {key: float(value) for key, value in raw_slice.items()}
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"invalid slice value: {e}. slice values must be numbers."
            )

    return PlotConfig(
        plot_type=_validate_plot_type(plot_type, files),
        fields=getattr(args, "fields", ["rho"]),
        ndim=get_dimensionality(files),
        slice=slice_spec,
    )


def figure_config_from_args(args: Namespace) -> FigureConfig:
    """build FigureConfig from cli arguments."""
    return FigureConfig(
        fig_size=getattr(args, "fig_size", None) or (8, 6),
        dpi=getattr(args, "dpi", 300),
        xlims=_pair_to_bounds(getattr(args, "xlims", None)),
        ylims=_pair_to_bounds(getattr(args, "ylims", None)),
        xlabel=getattr(args, "xlabel", None),
        ylabel=getattr(args, "ylabel", None),
        xscale=getattr(args, "xscale", "linear"),
        yscale=getattr(args, "yscale", "linear"),
        title=getattr(args, "setup", None),
        draw_bodies=getattr(args, "draw_bodies", False),
        time_scale=getattr(args, "time_scale", None),
        time_units=getattr(args, "time_units", ""),
        transparent=getattr(args, "transparent", False),
    )


def refinement_config_from_args(args: Namespace) -> RefinementConfig:
    """build RefinementConfig from cli arguments."""
    raw_levels = getattr(args, "active_levels", None)

    active_levels: Optional[set[int]] = None
    if raw_levels:
        if len(raw_levels) == 1 and raw_levels[0].lower() == "all":
            active_levels = None
        else:
            try:
                active_levels = set(int(lvl) for lvl in raw_levels)
            except ValueError:
                raise ValueError(
                    f"--active-levels must be integers or 'all', got: {raw_levels}"
                )

    return RefinementConfig(
        composite_view=getattr(args, "composite_view", False),
        active_levels=active_levels,
        render_mode=getattr(args, "render_mode", "pcolormesh"),
    )


def coordinate_config_from_args(args: Namespace) -> CoordinateConfig:
    """build CoordinateConfig from cli arguments."""
    return CoordinateConfig(
        n_bins=getattr(args, "n_bins", 64),
    )


def time_series_config_from_args(args: Namespace) -> TimeSeriesConfig:
    """build TimeSeriesConfig from cli arguments."""
    return TimeSeriesConfig(
        weight=getattr(args, "weight", None),
    )


def temporal_spectrum_config_from_args(
    args: Namespace,
) -> TemporalSpectrumConfig:
    """build TemporalSpectrumConfig from cli arguments."""
    return TemporalSpectrumConfig(
        psd_method=getattr(args, "psd_method", "standard"),
        n_segments=getattr(args, "psd_segments", 8),
        overlap=getattr(args, "psd_overlap", 0.5),
        normalize_psd=getattr(args, "normalize_psd", False),
    )


def animation_config_from_args(args: Namespace) -> AnimationConfig:
    """build AnimationConfig from cli arguments."""
    return AnimationConfig(
        total_frames=len(getattr(args, "files", [])),
        frame_rate=getattr(args, "frame_rate", 30),
        save_all_frames=getattr(args, "save_all_frames", False),
    )


def theme_config_from_args(args: Namespace) -> ThemeConfig:
    """build ThemeConfig from cli arguments."""
    from simbi.viz.config_loader import parse_overrides

    global_overrides, _ = parse_overrides(getattr(args, "props", []))
    overrides = global_overrides.get("theme", {})
    color_cycle = getattr(args, "color_cycle", None)
    if color_cycle:
        overrides = dict(overrides) if overrides else {}
        overrides["color_map"] = color_cycle
    color_range = getattr(args, "color_range", None)
    if color_range:
        overrides = dict(overrides) if overrides else {}
        overrides["color_range"] = tuple(color_range)
    color_indices = getattr(args, "color_indices", None)
    if color_indices:
        overrides = dict(overrides) if overrides else {}
        overrides["color_indices"] = tuple(color_indices)
    return get_theme(getattr(args, "theme", "default"), overrides or None)


def config_from_args(args: Namespace) -> VisualizationConfig:
    """build complete VisualizationConfig from cli arguments."""
    return VisualizationConfig(
        plot=plot_config_from_args(args),
        figure=figure_config_from_args(args),
        refinement=refinement_config_from_args(args),
        coordinate=coordinate_config_from_args(args),
        time_series=time_series_config_from_args(args),
        temporal_spectrum=temporal_spectrum_config_from_args(args),
        animation=animation_config_from_args(args),
        theme=theme_config_from_args(args),
        overlays=overlays_from_args(args),
    )


def is_animation_requested(args: Namespace) -> bool:
    """check if animation was requested."""
    return (
        getattr(args, "animate", False)
        or getattr(args, "kind", "snapshot") == "movie"
    )


def should_show_plot(args: Namespace) -> bool:
    """determine if plot should be displayed."""
    return not getattr(args, "no_show", False)


def get_save_path(args: Namespace) -> Optional[str]:
    """get the save path if specified."""
    return getattr(args, "save_as", None)


def load_props_from_args(
    args: Namespace,
) -> tuple[dict[str, ComponentProps], dict]:
    """
    load component props from config file and/or cli overrides.

    returns (global_props, per_file_overrides).
    """
    from simbi.viz.config_loader import load_component_props

    config_path = getattr(args, "config", None)
    overrides = getattr(args, "props", [])

    if not config_path and not overrides:
        return {}, {}

    return load_component_props(config_path, overrides)


def handle_generate_config(args: Namespace) -> bool:
    """
    handle --generate-config flag. returns True if handled (should exit).
    """
    if getattr(args, "generate_config", False):
        from simbi.viz.config_loader import generate_example_config

        print(generate_example_config())
        return True
    return False
