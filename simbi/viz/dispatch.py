# =============================================================================
# dispatch.py
#
# maps data characteristics to component types.
# single source of truth for component selection logic.
#
# usage:
#   comp_cls, props_cls, key = select_scalar_component(field, use_polygons=False)
#   comp_cls, props_cls, key = select_overlay_component("contour")
#   comp_cls, props_cls, key = select_vector_component("quiver")
# =============================================================================
from typing import Type

from .components.contour import ContourPlotComponent, ContourPlotProps
from .components.interface import ComponentProps
from .components.line import LinePlotComponent, LinePlotProps
from .components.polygons import PolygonPlotComponent, PolygonPlotProps
from .components.quad import QuadPlotComponent, QuadPlotProps
from .components.quiver import QuiverPlotComponent, QuiverPlotProps
from .components.stream import StreamPlotComponent, StreamPlotProps
from .types import FieldData

# dispatch key: (ndim, is_polygon)
# value: (component_class, props_class, registry_key)
SCALAR_DISPATCH: dict[
    tuple[int, bool], tuple[Type, Type[ComponentProps], str]
] = {
    (1, False): (LinePlotComponent, LinePlotProps, "line"),
    (1, True): (PolygonPlotComponent, PolygonPlotProps, "polygon"),
    (2, False): (QuadPlotComponent, QuadPlotProps, "quad"),
    (2, True): (PolygonPlotComponent, PolygonPlotProps, "polygon"),
}

VECTOR_DISPATCH: dict[str, tuple[Type, Type[ComponentProps], str]] = {
    "quiver": (QuiverPlotComponent, QuiverPlotProps, "quiver"),
    "stream": (StreamPlotComponent, StreamPlotProps, "stream"),
}

OVERLAY_DISPATCH: dict[str, tuple[Type, Type[ComponentProps], str]] = {
    "contour": (ContourPlotComponent, ContourPlotProps, "contour"),
}


def select_scalar_component(
    field: FieldData,
    use_polygons: bool = False,
) -> tuple[Type, Type[ComponentProps], str]:
    """
    select component for a scalar field based on dimensionality.

    args:
        field: the field data to visualize
        use_polygons: force polygon rendering (for AMR visualization)

    returns:
        tuple of (component_class, props_class, registry_key)

    raises:
        ValueError: if no component matches the field characteristics
    """
    ndim = field.ndim
    is_polygon = field.name.endswith("_polygons") or use_polygons

    if ndim == 3:
        raise ValueError(
            f"field '{field.name}' is 3D. use --slice to reduce dimensionality."
        )

    key = (ndim, is_polygon)

    if key not in SCALAR_DISPATCH:
        raise ValueError(
            f"no component for ndim={ndim}, polygon={is_polygon}. "
            f"valid combinations: {list(SCALAR_DISPATCH.keys())}"
        )

    return SCALAR_DISPATCH[key]


def select_vector_component(
    vector_type: str = "quiver",
) -> tuple[Type, Type[ComponentProps], str]:
    """
    select component for vector field visualization.

    args:
        vector_type: "quiver" or "stream"

    returns:
        tuple of (component_class, props_class, registry_key)

    raises:
        ValueError: if vector_type is not recognized
    """
    if vector_type not in VECTOR_DISPATCH:
        valid = ", ".join(sorted(VECTOR_DISPATCH.keys()))
        raise ValueError(
            f"unknown vector_type: '{vector_type}'. valid: {valid}"
        )

    return VECTOR_DISPATCH[vector_type]


def select_overlay_component(
    overlay_type: str,
) -> tuple[Type, Type[ComponentProps], str]:
    """
    select component for overlay visualization.

    args:
        overlay_type: type of overlay (e.g., "contour")

    returns:
        tuple of (component_class, props_class, registry_key)

    raises:
        ValueError: if overlay_type is not recognized
    """
    if overlay_type not in OVERLAY_DISPATCH:
        valid = ", ".join(sorted(OVERLAY_DISPATCH.keys()))
        raise ValueError(
            f"unknown overlay_type: '{overlay_type}'. valid: {valid}"
        )

    return OVERLAY_DISPATCH[overlay_type]


def list_overlay_types() -> list[str]:
    """return list of available overlay component types."""
    return sorted(OVERLAY_DISPATCH.keys())
