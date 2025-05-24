from typing import Any
from .types import VisualizationConfig


def validate_config(config: VisualizationConfig) -> None:
    """Validate configuration"""
    # Validate plot configuration
    if "plot" not in config:
        raise ValueError("Missing plot configuration")

    plot_config = config["plot"]

    # Required fields
    if "files" not in plot_config:
        raise ValueError("Missing files in plot configuration")
    if "plot_type" not in plot_config:
        raise ValueError("Missing plot_type in plot configuration")

    # Type-specific validation
    if plot_config["plot_type"] == "multidim" and plot_config.get("ndim", 0) < 2:
        raise ValueError("Multidim plots require ndim >= 2")

    # Animation validation
    if "animation" in config and config["animation"].get("frame_rate", 0) <= 0:
        raise ValueError("Animation frame_rate must be positive")


def create_default_config(plot_type: str) -> dict[str, Any]:
    """Create default configuration for a plot type"""
    config = {
        "plot": {
            "plot_type": plot_type,
            "files": [],
            "fields": ["rho"],
            "setup": "Simulation",
            "ndim": 1,
        },
        "style": {
            "cmap": ["viridis"],
            "log": False,
            "fig_dims": (10, 6),
            "dpi": 300,
            "legend": True,
            "show_colorbar": True,
        },
    }

    # Add plot type specific defaults
    if plot_type == "multidim":
        config["multidim"] = {
            "projection": [1, 2, 3],
            "box_depth": 0.0,
            "bipolar": False,
        }
        config["plot"]["ndim"] = 2
    elif plot_type == "histogram":
        config["plot"]["hist_type"] = "kinetic"
        config["style"]["log"] = True
    elif plot_type == "temporal":
        config["plot"]["weight"] = None

    # Add animation defaults if needed
    config["animation"] = {"frame_rate": 30}

    return config
