# =============================================================================
# conversion.py
#
# converts cli arguments to typed configuration objects.
# clean separation: figure config is separate from component props.
# component styling is handled entirely by config_loader.
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
    TimeSeriesConfig,
    VisualizationConfig,
)
from simbi.viz.utility import get_dimensionality

from ..styling import ThemeManager
from ..styling.theme import ThemeConfig
from ..types import Bounds


def _pair_to_bounds(pair: list[float] | None) -> Optional[Bounds]:
    """Convert a [min, max] pair to Bounds object."""
    if not pair or not any(x is not None for x in pair):
        return None
    return Bounds(min=pair[0], max=pair[1])


def _validate_plot_type(
    plot_type: str | None, files: list[str]
) -> Literal["line", "multidim", "time_series", "coordinate_bin"]:
    """Validate and auto-detect plot type if not specified."""
    if plot_type:
        return plot_type  # type: ignore[return-value]

    ndim = get_dimensionality(files) if files else 1
    return "line" if ndim == 1 else "multidim"


def plot_config_from_args(args: Namespace) -> PlotConfig:
    """Build PlotConfig from cli arguments."""
    import argparse

    files = args.files
    plot_type = getattr(args, "plot_type", None)

    # parse slice spec
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
        norm=getattr(args, "norm", None),
    )


def figure_config_from_args(args: Namespace) -> FigureConfig:
    """Build FigureConfig from cli arguments."""
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
        time_scale=getattr(args, "time_scale", None),
        time_units=getattr(args, "time_units", ""),
        transparent=getattr(args, "transparent", False),
        draw_bodies=getattr(args, "draw_bodies", False),
        draw_tracers=getattr(args, "draw_tracers", False),
        draw_horizon=getattr(args, "draw_horizon", False),
    )


def refinement_config_from_args(args: Namespace) -> RefinementConfig:
    """Build RefinementConfig from cli arguments."""
    raw_levels = getattr(args, "active_levels", None)

    active_levels: Optional[set[int]] = None
    if raw_levels:
        if len(raw_levels) == 1 and raw_levels[0].lower() == "all":
            active_levels = None  # signals "all levels"
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
    """Build CoordinateConfig from cli arguments."""
    return CoordinateConfig(
        n_bins=getattr(args, "n_bins", 64),
    )


def time_series_config_from_args(args: Namespace) -> TimeSeriesConfig:
    """Build TimeSeriesConfig from cli arguments."""
    return TimeSeriesConfig(
        weight=getattr(args, "weight", None),
    )


def animation_config_from_args(args: Namespace) -> AnimationConfig:
    """Build AnimationConfig from cli arguments."""
    return AnimationConfig(
        total_frames=len(getattr(args, "files", [])),
        frame_rate=getattr(args, "frame_rate", 30),
        save_all_frames=getattr(args, "save_all_frames", False),
    )


def theme_config_from_args(args: Namespace) -> ThemeConfig:
    """Build ThemeConfig from cli arguments.

    Strategy:
      - prefer validated theme props loaded via the existing loader (file + overrides)
      - convert validated ThemeProps -> ThemeConfig using ThemeConfig.from_mapping
      - if no validated theme props are present, parse the CLI overrides to extract
        any `theme.*` overrides and pass them to ThemeManager.get_theme
      - ensure a ThemeConfig is returned in all code paths
    """
    # delegate parsing/validation to the config loader when possible to reuse
    # coercion and pydantic validation, avoiding ad-hoc parsing here.
    from simbi.viz.config_loader import load_theme_config, parse_overrides

    # try to load validated theme props (from config file and/or --props)
    theme_props = load_theme_config(
        getattr(args, "config", None), getattr(args, "props", [])
    )
    if theme_props:
        # theme_props is a pydantic ThemeProps instance (or mapping); convert safely
        return ThemeConfig.from_mapping(theme_props)

    # no validated ThemeProps provided: extract any theme.* overrides to pass to ThemeManager
    overrides = parse_overrides(getattr(args, "props", []))
    theme_override_dict = overrides.get("theme", {})

    theme_candidate = ThemeManager.get_theme(
        theme_name=getattr(args, "theme", "default"),
        theme_props=theme_override_dict,
    )

    # if ThemeManager already returned a ThemeConfig, use it directly
    if isinstance(theme_candidate, ThemeConfig):
        return theme_candidate

    # try to convert mapping-like candidate into ThemeConfig
    if hasattr(ThemeConfig, "from_mapping"):
        try:
            return ThemeConfig.from_mapping(theme_candidate)
        except Exception:
            pass

    # fallback: if dict-like, filter to dataclass fields
    if isinstance(theme_candidate, dict):
        allowed = set(ThemeConfig.__dataclass_fields__.keys())
        filtered = {k: v for k, v in theme_candidate.items() if k in allowed}
        return ThemeConfig(**filtered)

    # last resort: default ThemeConfig
    return ThemeConfig()


def config_from_args(args: Namespace) -> VisualizationConfig:
    """Build complete VisualizationConfig from cli arguments."""
    return VisualizationConfig(
        plot=plot_config_from_args(args),
        figure=figure_config_from_args(args),
        refinement=refinement_config_from_args(args),
        coordinate=coordinate_config_from_args(args),
        time_series=time_series_config_from_args(args),
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


def load_props_from_args(args: Namespace) -> dict[str, ComponentProps]:
    """
    Load component props from config file and/or cli overrides.

    Args:
        args: parsed cli arguments (expects --config and --props)

    Returns:
        dict mapping component names to validated props instances
    """
    from simbi.viz.config_loader import load_component_props

    config_path = getattr(args, "config", None)
    overrides = getattr(args, "props", [])

    if not config_path and not overrides:
        return {}

    return load_component_props(config_path, overrides)


def handle_generate_config(args: Namespace) -> bool:
    """
    Handle --generate-config flag. Returns True if handled (should exit).
    """
    if getattr(args, "generate_config", False):
        from simbi.viz.config_loader import generate_example_config

        print(generate_example_config())
        return True
    return False
