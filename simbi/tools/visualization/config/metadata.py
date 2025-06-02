from typing import Sequence, Any, Optional, Callable, TypeVar

T = TypeVar("T")


class ArgumentMapping:
    """Registry for argument-to-config mapping metadata"""

    # Map arguments to their config sections
    # Format: {arg_name: section_name}
    SECTION_MAP: dict[str, str] = {
        # Plot section arguments
        "files": "plot",
        "fields": "plot",
        "plot_type": "plot",
        "setup": "plot",
        "ndim": "plot",
        "weight": "plot",
        "hist_type": "plot",
        "powerfit": "plot",
        # Style section arguments
        "cmap": "style",
        "log": "style",
        "power": "style",
        "fig_size": "style",
        "dpi": "style",
        "legend": "style",
        "xlims": "style",
        "ylims": "style",
        "xlabel": "style",
        "ylabel": "style",
        "show_colorbar": "style",
        "draw_bodies": "style",
        "colorbar_orientation": "style",
        "color_range": "style",
        "orbital_params": "style",
        "theme": "style",
        "black_background": "style",
        "transparent": "style",
        "reverse_colormap": "style",
        "color_maps": "style",
        "semilogx": "style",
        "semilogy": "style",
        "units": "style",
        "legend_loc": "style",
        "labels": "style",
        "annotation_loc": "style",
        "annotation_text": "style",
        "annotation_anchor": "style",
        "pictorial": "style",
        "print": "style",
        "bbox_kind": "style",
        "font_color": "style",
        "scale_downs": "style",
        "time_modulus": "style",
        "normalize": "style",
        "nlinestyles": "style",
        "split_into_subplots": "style",
        "show": "style",
        "save_as": "style",
        "extension": "style",
        "xmax": "style",
        # Multidim section arguments
        "projection": "multidim",
        "box_depth": "multidim",
        "bipolar": "multidim",
        "slice_along": "multidim",
        "coords": "multidim",
        # Animation section arguments
        "frame_rate": "animation",
        "animate": "animation",
    }

    # Special case transformations
    # Format: {arg_name: transformation_function}
    TRANSFORMATIONS: dict[str, Callable[[Any], Any]] = {
        "orbital_params": lambda v: parse_key_value_pairs(v, float),
        "coords": lambda v: parse_key_value_pairs(
            v, lambda x: [x] if isinstance(x, str) else x
        ),
        "fig_dims": lambda v: tuple(v) if isinstance(v, list) else v,
        "xlims": lambda v: tuple(v) if isinstance(v, list) else v,
        "ylims": lambda v: tuple(v) if isinstance(v, list) else v,
        "projection": lambda v: tuple(v) if isinstance(v, tuple) else v,
    }

    # Arguments that should be renamed when moved to config sections
    # Format: {arg_name: new_name}
    RENAME_MAP: dict[str, str] = {
        "draw_bodies": "draw_immersed_bodies",
    }

    @classmethod
    def get_section(cls, arg_name: str) -> str:
        """Get the configuration section for an argument"""
        return cls.SECTION_MAP.get(arg_name, "plot")  # Default to plot section

    @classmethod
    def get_transform(cls, arg_name: str) -> Optional[Callable[[Any], Any]]:
        """Get transformation function for an argument if it exists"""
        return cls.TRANSFORMATIONS.get(arg_name)

    @classmethod
    def get_name(cls, arg_name: str) -> str:
        """Get the config name for an argument (applies renaming)"""
        return cls.RENAME_MAP.get(arg_name, arg_name)


def parse_key_value_pairs(
    pairs: dict[str, Any], value_converter: Callable[[str], T] = str
) -> dict[str, T]:
    """Parse key=value pairs into a dictionary"""
    if not pairs:
        return {}

    result = {}
    for key, value in pairs.items():
        # Handle existing keys
        if key in result:
            if isinstance(result[key], list):
                result[key].append(value_converter(value))
            else:
                result[key] = [result[key], value_converter(value)]
        else:
            result[key] = value_converter(value)

    return result
