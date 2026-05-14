# =============================================================================
# registry.py
#
# single source of truth for visualization components.
# components are grouped by category in separate dicts.
# each entry is a (component_cls, props_cls, props_key) tuple.
#
# usage:
#   from simbi.viz.registry import select_scalar_component
#   comp_cls, props_cls, key = select_scalar_component(field_data)
# =============================================================================
from __future__ import annotations

from typing import Any, Optional, Type

# component tuple: (component_cls, props_cls, props_key)
_ComponentEntry = tuple[Type, Type, str]

# lazy-loaded dicts
_SCALAR: Optional[dict[str, _ComponentEntry]] = None
_VECTOR: Optional[dict[str, _ComponentEntry]] = None
_OVERLAY: Optional[dict[str, _ComponentEntry]] = None
_ANALYSIS: Optional[dict[str, _ComponentEntry]] = None

# cli plot-type names -> internal registry keys
PLOT_TYPE_ALIASES: dict[str, str] = {
    "line": "line",
    "multidim": "quad",
    "coordinate_bin": "coordinate_profile",
    "time_series": "time_series",
    "power_spectrum": "power_spectrum",
    "temporal_spectrum": "temporal_spectrum",
    "angular_spectrum": "power_spectrum",
    "phase_fold": "time_series",
}


def _ensure_loaded() -> None:
    """lazy-load all component dicts to avoid circular imports."""
    global _SCALAR, _VECTOR, _OVERLAY, _ANALYSIS
    if _SCALAR is not None:
        return

    from .components.contour import ContourPlotComponent, ContourPlotProps
    from .components.coord_binning import (
        CoordinateProfileComponent,
        CoordinateProfileProps,
    )
    from .components.line import LinePlotComponent, LinePlotProps
    from .components.polygons import PolygonPlotComponent, PolygonPlotProps
    from .components.power_spectrum import (
        PowerSpectrumComponent,
        PowerSpectrumProps,
    )
    from .components.quad import QuadPlotComponent, QuadPlotProps
    from .components.quiver import QuiverPlotComponent, QuiverPlotProps
    from .components.stream import StreamPlotComponent, StreamPlotProps
    from .components.time_series import (
        TimeSeriesPlotComponent,
        TimeSeriesPlotProps,
    )

    _SCALAR = {
        "line": (LinePlotComponent, LinePlotProps, "line"),
        "quad": (QuadPlotComponent, QuadPlotProps, "quad"),
        "polygon": (PolygonPlotComponent, PolygonPlotProps, "polygon"),
    }

    _VECTOR = {
        "quiver": (QuiverPlotComponent, QuiverPlotProps, "quiver"),
        "stream": (StreamPlotComponent, StreamPlotProps, "stream"),
    }

    _OVERLAY = {
        "contour": (ContourPlotComponent, ContourPlotProps, "contour"),
    }

    _ANALYSIS = {
        "coordinate_profile": (
            CoordinateProfileComponent,
            CoordinateProfileProps,
            "coordinate_profile",
        ),
        "time_series": (
            TimeSeriesPlotComponent,
            TimeSeriesPlotProps,
            "time_series",
        ),
        "power_spectrum": (
            PowerSpectrumComponent,
            PowerSpectrumProps,
            "power_spectrum",
        ),
        # temporal spectrum reuses the power spectrum component
        "temporal_spectrum": (
            PowerSpectrumComponent,
            PowerSpectrumProps,
            "power_spectrum",
        ),
    }


def _all_entries() -> dict[str, _ComponentEntry]:
    """merged view of all component dicts."""
    _ensure_loaded()
    return {**_SCALAR, **_VECTOR, **_OVERLAY, **_ANALYSIS}


# =========================================================================
# public accessors
# =========================================================================


def get_valid_plot_types() -> list[str]:
    """plot types valid for --plot-type cli flag."""
    return sorted(PLOT_TYPE_ALIASES.keys())


def get_props_class(name: str) -> Type:
    """get props class by component name."""
    entries = _all_entries()
    key = name.lower().replace("-", "_")
    if key not in entries:
        valid = ", ".join(sorted(entries.keys()))
        raise KeyError(f"unknown component '{name}'. valid: {valid}")
    return entries[key][1]


def get_props_registry() -> dict[str, Type]:
    """get flat dict of name -> props_cls (used by config_loader)."""
    return {name: entry[1] for name, entry in _all_entries().items()}


def list_components() -> list[str]:
    """return sorted list of registered component names."""
    return sorted(_all_entries().keys())


# =========================================================================
# dispatch helpers
# =========================================================================


def select_scalar_component(
    field_data: Any, use_polygons: bool = False
) -> tuple[Type, Type, str]:
    """select component for a scalar field based on dimensionality."""
    _ensure_loaded()

    ndim = field_data.ndim
    is_polygon = field_data.name.endswith("_polygons") or use_polygons

    if ndim == 3:
        raise ValueError(
            f"field '{field_data.name}' is 3D. use --slice to reduce dimensionality."
        )

    if is_polygon:
        return _SCALAR["polygon"]

    if ndim == 1:
        return _SCALAR["line"]
    elif ndim == 2:
        return _SCALAR["quad"]

    raise ValueError(f"no component for ndim={ndim}, polygon={is_polygon}.")


def select_vector_component(
    vector_type: str = "quiver",
) -> tuple[Type, Type, str]:
    """select component for vector field visualization."""
    _ensure_loaded()

    if vector_type not in _VECTOR:
        valid = ", ".join(sorted(_VECTOR.keys()))
        raise ValueError(
            f"unknown vector_type: '{vector_type}'. valid: {valid}"
        )
    return _VECTOR[vector_type]


def select_overlay_component(
    overlay_type: str,
) -> tuple[Type, Type, str]:
    """select component for overlay visualization."""
    _ensure_loaded()

    if overlay_type not in _OVERLAY:
        valid = ", ".join(sorted(_OVERLAY.keys()))
        raise ValueError(
            f"unknown overlay_type: '{overlay_type}'. valid: {valid}"
        )
    return _OVERLAY[overlay_type]


def list_overlay_types() -> list[str]:
    """return list of available overlay component types."""
    _ensure_loaded()
    return sorted(_OVERLAY.keys())


def refinement_info(fields, config) -> tuple[int, bool]:
    """compute nlvls and use_polygons from field list."""
    nlvls = 1 + sum("_L" in f.name for f in fields)
    use_polygons = nlvls > 1 or config.refinement.render_mode == "polygons"
    return nlvls, use_polygons
