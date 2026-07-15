# =============================================================================
# props_registry.py
#
# central registry mapping component names to their props classes.
# single source of truth for config file parsing and CLI generation.
#
# usage:
#   from simbi.viz.props_registry import PROPS_REGISTRY
#   props_cls = PROPS_REGISTRY["polygon"]
#   props = props_cls(**config_dict)
# =============================================================================
from typing import Type

from .components.coord_binning import CoordinateProfileProps
from .components.interface import ComponentProps
from .components.line import LinePlotProps
from .components.polygons import PolygonPlotProps
from .components.quad import QuadPlotProps
from .components.quiver import QuiverPlotProps
from .components.stream import StreamPlotProps
from .components.theming import ThemeProps
from .components.time_series import TimeSeriesPlotProps

# maps config key -> props class
# keys are lowercase, no suffix - these match yaml section names
PROPS_REGISTRY: dict[str, Type[ComponentProps]] = {
    "polygon": PolygonPlotProps,
    "quad": QuadPlotProps,
    "line": LinePlotProps,
    "quiver": QuiverPlotProps,
    "stream": StreamPlotProps,
    "coordinate_profile": CoordinateProfileProps,
    "time_series": TimeSeriesPlotProps,
    "theme": ThemeProps,
}

# reverse lookup: props class -> config key
PROPS_TO_KEY: dict[Type[ComponentProps], str] = {
    v: k for k, v in PROPS_REGISTRY.items()
}


def get_props_class(name: str) -> Type[ComponentProps]:
    """Get props class by name. Raises KeyError if not found."""
    key = name.lower().replace("-", "_")
    if key not in PROPS_REGISTRY:
        valid = ", ".join(sorted(PROPS_REGISTRY.keys()))
        raise KeyError(f"unknown component '{name}'. valid: {valid}")
    return PROPS_REGISTRY[key]


def list_components() -> list[str]:
    """Return sorted list of registered component names."""
    return sorted(PROPS_REGISTRY.keys())
