from typing import Any
from .visualization import api


def visualize(config: dict[str, Any]) -> None:
    """
    Legacy entry point for visualization.

    Translates old config-based calls to the new component-based API.
    """
    # Extract key configuration sections
    plot_config = config["plot"]
    style_config = config["style"]
    multidim_config = config["multidim"]
    animation_config = config["animation"]

    # Extract common parameters
    files = plot_config["files"]
    fields = plot_config["fields"]
    setup_name = plot_config["setup"]
    plot_type = plot_config["plot_type"]
    save_as = plot_config["save_as"]

    # Create kwargs from configuration
    kwargs = _build_api_kwargs(
        plot_config, style_config, multidim_config, animation_config
    )

    # Determine if we're creating an animation
    is_animation = plot_config["kind"] == "movie"

    # Call appropriate API function
    if is_animation:
        if plot_type == "line":
            api.animate_line(
                files, fields, save_as, True, animation_config["frame_rate"], **kwargs
            )
        elif plot_type == "multidim":
            api.animate_multidim(
                files, fields, save_as, True, animation_config["frame_rate"], **kwargs
            )
        elif plot_type == "histogram":
            api.animate_histogram(
                files, fields, save_as, True, animation_config["frame_rate"], **kwargs
            )
        elif plot_type == "temporal":
            api.animate_temporal(
                files, fields, save_as, True, animation_config["frame_rate"], **kwargs
            )
    else:
        if plot_type == "line":
            api.plot_line(files, fields, save_as, True, **kwargs)
        elif plot_type == "multidim":
            api.plot_multidim(files, fields, save_as, True, **kwargs)
        elif plot_type == "histogram":
            api.plot_histogram(files, fields, save_as, True, **kwargs)
        elif plot_type == "temporal":
            api.plot_temporal(files, fields, save_as, True, **kwargs)


def _build_api_kwargs(
    plot_config: dict[str, Any],
    style_config: dict[str, Any],
    multidim_config: dict[str, Any],
    animation_config: dict[str, Any],
) -> dict[str, Any]:
    """Convert config dict to kwargs for API functions"""
    kwargs = {}

    # Plot configuration
    kwargs["ndim"] = plot_config["ndim"]
    kwargs["setup"] = plot_config["setup"]

    if "hist_type" in plot_config:
        kwargs["hist_type"] = plot_config["hist_type"]
    if "powerfit" in plot_config:
        kwargs["powerfit"] = plot_config["powerfit"]
    if "weight" in plot_config:
        kwargs["weight"] = plot_config["weight"]

    # Style configuration
    kwargs["cmap"] = style_config["color_maps"]
    kwargs["log"] = style_config["log"]
    kwargs["power"] = style_config["power"]
    kwargs["fig_dims"] = style_config["fig_dims"]
    kwargs["dpi"] = style_config["dpi"]
    kwargs["legend"] = style_config["legend"]
    kwargs["xlims"] = style_config["xlims"]
    kwargs["ylims"] = style_config["ylims"]
    kwargs["draw_bodies"] = style_config["draw_immersed_bodies"]
    kwargs["color_range"] = style_config["color_range"]
    kwargs["units"] = style_config["units"]

    if "orbital_params" in style_config and style_config["orbital_params"]:
        kwargs["orbital_params"] = style_config["orbital_params"]

    # Multidim configuration
    kwargs["projection"] = multidim_config["projection"]
    kwargs["box_depth"] = multidim_config["box_depth"]
    kwargs["bipolar"] = multidim_config["bipolar"]
    kwargs["slice_along"] = multidim_config["slice_along"]
    kwargs["coords"] = multidim_config["coords"]

    # Remove None values
    return {k: v for k, v in kwargs.items() if v is not None}
