from .api import (
    animate,
    animate_coordinate_profile,
    plot,
    plot_coordinate_profile,
    plot_coordinate_profile_overlay,
    plot_overlay,
    plot_time_series,
)

from . import colormaps  # noqa: F401  (import for side effect: registers the simbi_* composites)
from .cli import setup_parser as setup_viz_parser
from .config_loader import generate_example_config, load_component_props
from .pipeline import config_from_args
from .pipeline.conversion import handle_generate_config, load_props_from_args

__all__ = [
    "setup_viz_parser",
    "plot",
    "plot_overlay",
    "animate",
    "animate_coordinate_profile",
    "plot_coordinate_profile",
    "plot_coordinate_profile_overlay",
    "plot_time_series",
    "config_from_args",
    "load_component_props",
    "load_props_from_args",
    "handle_generate_config",
    "generate_example_config",
    # body diagnostics
    # "BodyTimeSeries",
    # "BinaryTimeSeries",
    # "SingleBodyTimeSeries",
    # "load_body_timeseries",
    # "compute_binary_dynamics",
    # "plot_forces",
    # "plot_torques",
    # "plot_separation",
    # "plot_accretion_rate",
    # "plot_orbital_elements",
    # "plot_radial_acceleration",
    # "plot_body_diagnostics_summary",
    # "plot_binary_diagnostics_summary",
]
