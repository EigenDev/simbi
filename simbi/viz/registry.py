# =============================================================================
# registry.py
#
# single source of truth for all visualization components.
# adding a new plot type means adding one entry here (plus the component file).
#
# usage:
#   from simbi.viz.registry import registry, get_valid_plot_types
#   entry = registry()["power_spectrum"]
#   component = entry.component_cls(entry.props_cls())
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Type

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PlotTypeEntry:
    """metadata for a registered plot type."""

    component_cls: Type
    props_cls: Type
    props_key: str
    category: str  # "scalar", "vector", "overlay", "analysis"
    data_dims: tuple[int, ...] = ()
    is_polygon: bool = False
    supports_animation: bool = False
    supports_overlay: bool = False


_REGISTRY: Optional[dict[str, PlotTypeEntry]] = None


def _build_registry() -> dict[str, PlotTypeEntry]:
    """build the component registry. lazy-loaded to avoid circular imports."""
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

    return {
        # scalar components (dispatched by ndim)
        "line": PlotTypeEntry(
            component_cls=LinePlotComponent,
            props_cls=LinePlotProps,
            props_key="line",
            category="scalar",
            data_dims=(1,),
            supports_animation=True,
            supports_overlay=True,
        ),
        "quad": PlotTypeEntry(
            component_cls=QuadPlotComponent,
            props_cls=QuadPlotProps,
            props_key="quad",
            category="scalar",
            data_dims=(2,),
            supports_animation=True,
        ),
        "polygon": PlotTypeEntry(
            component_cls=PolygonPlotComponent,
            props_cls=PolygonPlotProps,
            props_key="polygon",
            category="scalar",
            data_dims=(1, 2),
            is_polygon=True,
            supports_animation=True,
        ),
        # vector components
        "quiver": PlotTypeEntry(
            component_cls=QuiverPlotComponent,
            props_cls=QuiverPlotProps,
            props_key="quiver",
            category="vector",
            data_dims=(2,),
        ),
        "stream": PlotTypeEntry(
            component_cls=StreamPlotComponent,
            props_cls=StreamPlotProps,
            props_key="stream",
            category="vector",
            data_dims=(2,),
        ),
        # overlay components
        "contour": PlotTypeEntry(
            component_cls=ContourPlotComponent,
            props_cls=ContourPlotProps,
            props_key="contour",
            category="overlay",
            data_dims=(2,),
        ),
        # analysis components (custom pipelines, not dispatched by ndim)
        "coordinate_profile": PlotTypeEntry(
            component_cls=CoordinateProfileComponent,
            props_cls=CoordinateProfileProps,
            props_key="coordinate_profile",
            category="analysis",
            supports_animation=True,
            supports_overlay=True,
        ),
        "time_series": PlotTypeEntry(
            component_cls=TimeSeriesPlotComponent,
            props_cls=TimeSeriesPlotProps,
            props_key="time_series",
            category="analysis",
        ),
        "power_spectrum": PlotTypeEntry(
            component_cls=PowerSpectrumComponent,
            props_cls=PowerSpectrumProps,
            props_key="power_spectrum",
            category="analysis",
        ),
    }


def registry() -> dict[str, PlotTypeEntry]:
    """get the component registry (lazy-initialized)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


# =========================================================================
# convenience accessors
# =========================================================================
def get_valid_plot_types() -> list[str]:
    """plot types valid for --plot-type cli flag."""
    from .config import _PLOT_TYPE_TO_REGISTRY

    return sorted(_PLOT_TYPE_TO_REGISTRY.keys())


def get_props_class(name: str) -> Type:
    """get props class by component name."""
    r = registry()
    key = name.lower().replace("-", "_")
    if key not in r:
        valid = ", ".join(sorted(r.keys()))
        raise KeyError(f"unknown component '{name}'. valid: {valid}")
    return r[key].props_cls


def get_props_registry() -> dict[str, Type]:
    """get flat dict of name -> props_cls (backward compat for config_loader)."""
    return {name: entry.props_cls for name, entry in registry().items()}


def list_components() -> list[str]:
    """return sorted list of registered component names."""
    return sorted(registry().keys())


# =========================================================================
# dispatch helpers (replace dispatch.py)
# =========================================================================
def select_scalar_component(
    field_data: Any, use_polygons: bool = False
) -> tuple[Type, Type, str]:
    """
    select component for a scalar field based on dimensionality.

    returns:
        tuple of (component_class, props_class, registry_key)
    """
    ndim = field_data.ndim
    is_polygon = field_data.name.endswith("_polygons") or use_polygons

    if ndim == 3:
        raise ValueError(
            f"field '{field_data.name}' is 3D. use --slice to reduce dimensionality."
        )

    r = registry()

    if is_polygon:
        entry = r["polygon"]
        return entry.component_cls, entry.props_cls, entry.props_key

    if ndim == 1:
        entry = r["line"]
    elif ndim == 2:
        entry = r["quad"]
    else:
        raise ValueError(f"no component for ndim={ndim}, polygon={is_polygon}.")

    return entry.component_cls, entry.props_cls, entry.props_key


def select_vector_component(
    vector_type: str = "quiver",
) -> tuple[Type, Type, str]:
    """select component for vector field visualization."""
    r = registry()
    if vector_type not in r or r[vector_type].category != "vector":
        valid = [k for k, v in r.items() if v.category == "vector"]
        raise ValueError(
            f"unknown vector_type: '{vector_type}'. valid: {', '.join(valid)}"
        )
    entry = r[vector_type]
    return entry.component_cls, entry.props_cls, entry.props_key


def select_overlay_component(
    overlay_type: str,
) -> tuple[Type, Type, str]:
    """select component for overlay visualization."""
    r = registry()
    if overlay_type not in r or r[overlay_type].category != "overlay":
        valid = [k for k, v in r.items() if v.category == "overlay"]
        raise ValueError(
            f"unknown overlay_type: '{overlay_type}'. valid: {', '.join(valid)}"
        )
    entry = r[overlay_type]
    return entry.component_cls, entry.props_cls, entry.props_key


def list_overlay_types() -> list[str]:
    """return list of available overlay component types."""
    return sorted(k for k, v in registry().items() if v.category == "overlay")
