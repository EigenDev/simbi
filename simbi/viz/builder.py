# =============================================================================
# builder.py
#
# composable figure builder + shared dispatch functions.
# SimFigure provides a fluent, layered composition api for programmatic users.
# dispatch functions are the single source of truth for wiring fields to
# components — used by both api.py and grid.py.
#
# usage:
#   from simbi.viz.builder import SimFigure
#   from simbi.viz.pipeline import load_data, create_plot_data
#
#   sim = load_data("checkpoint.h5")
#   data = create_plot_data(sim, ["rho"], config)
#   fig = SimFigure(config).add_scalar(data).render()
#   fig.show()
# =============================================================================
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt

from .components.interface import Component, ComponentProps
from .config import OverlayConfig, VisualizationConfig
from .figure import Figure, prepare_figure
from .pipeline import create_plot_data
from .pipeline.transforms import _compose_pcolormesh, compose_fields_for_render
from .registry import (
    refinement_info,
    select_overlay_component,
    select_scalar_component,
    select_vector_component,
)
from .types import CoordSystem, FieldData, PlotData

# ---------------------------------------------------------------------------
# shared dispatch functions (extracted from api.py)
# ---------------------------------------------------------------------------


def get_props(
    component_props: Optional[dict[str, ComponentProps]],
    key: str,
    default_factory,
) -> ComponentProps:
    """get props from dict or create default."""
    if component_props and key in component_props:
        return component_props[key]
    return default_factory()


def detect_projection(fields: Sequence[FieldData], coord_system: str) -> str:
    """determine projection from fields and coordinate system."""
    if not fields:
        return "cartesian"
    is_2d = fields[0].ndim == 2 or fields[0].name.endswith("_polygons")
    if is_2d and coord_system == "spherical":
        return "polar"
    return "cartesian"


def init_component(
    figure: Figure, component: Component, field_data, is_overlay: bool = False
) -> None:
    """initialize a component and attach it to the figure."""
    if figure.fig is None:
        raise RuntimeError("figure not initialized")
    component.initialize(figure.fig, figure.axes["main"])
    figure.add_component(component, field_data, is_overlay=is_overlay)


def dispatch_scalar_components(
    figure: Figure,
    final_fields: Sequence[FieldData],
    component_props: Optional[dict[str, ComponentProps]],
    use_polygons: bool,
    bodies=None,
) -> None:
    """create and attach scalar components to figure based on field dimensionality."""
    for field_data in final_fields:
        comp_cls, props_cls, props_key = select_scalar_component(
            field_data, use_polygons
        )
        props = get_props(component_props, props_key, props_cls)
        if props_key in ("polygon", "quad"):
            component = comp_cls(props, bodies)
        else:
            component = comp_cls(props)
        init_component(figure, component, field_data)


def dispatch_vector_components(
    figure: Figure,
    sim_data,
    vector_fields: Sequence[str],
    config: VisualizationConfig,
    component_props: Optional[dict[str, ComponentProps]],
    vector_type: str = "quiver",
) -> None:
    """create and attach vector field components (quiver or streamplot)."""
    vector_plot_data = create_plot_data(sim_data, vector_fields, config)

    vi_levels = [
        f
        for f in vector_plot_data.fields
        if f.name.startswith(vector_fields[0])
    ]
    vj_levels = [
        f
        for f in vector_plot_data.fields
        if f.name.startswith(vector_fields[1])
    ]

    v1_field = _compose_pcolormesh(vi_levels)
    v2_field = _compose_pcolormesh(vj_levels)

    comp_cls, props_cls, props_key = select_vector_component(vector_type)
    props = get_props(component_props, props_key, props_cls)
    init_component(figure, comp_cls(props), [v1_field, v2_field])


def dispatch_overlay_components(
    figure: Figure,
    sim_data,
    overlays: Sequence[OverlayConfig],
    config: VisualizationConfig,
) -> None:
    """create and attach overlay components (e.g., contour lines)."""
    for overlay in overlays:
        overlay_plot_data = create_plot_data(sim_data, [overlay.field], config)
        if not overlay_plot_data.fields:
            continue

        overlay_field = _compose_pcolormesh(list(overlay_plot_data.fields))
        if overlay_field.ndim != 2:
            continue

        comp_cls, props_cls, _ = select_overlay_component(overlay.component)
        props = props_cls(
            levels=tuple(overlay.levels),
            color=overlay.color,
            linewidths=overlay.linewidth,
            linestyles=overlay.linestyle,
            alpha=overlay.alpha,
            filled=overlay.filled,
            label_contours=overlay.label_contours,
        )
        init_component(figure, comp_cls(props), overlay_field, is_overlay=True)


def create_scalar_component(
    field_data: FieldData,
    component_props: Optional[dict[str, ComponentProps]],
    use_polygons: bool,
    bodies=None,
) -> tuple[Component, str]:
    """create a scalar component (without attaching to a figure).

    returns (component_instance, props_key) so the caller can do its own
    initialize/render against a specific (fig, ax) pair.
    """
    comp_cls, props_cls, props_key = select_scalar_component(
        field_data, use_polygons
    )
    props = get_props(component_props, props_key, props_cls)
    if props_key in ("polygon", "quad"):
        return comp_cls(props, bodies), props_key
    return comp_cls(props), props_key


# ---------------------------------------------------------------------------
# SimFigure — composable builder
# ---------------------------------------------------------------------------


class SimFigure:
    """composable figure builder for simbi visualization."""

    def __init__(
        self,
        config: VisualizationConfig,
        component_props: Optional[dict[str, ComponentProps]] = None,
    ):
        self._config = config
        self._component_props = component_props or {}
        self._figure: Optional[Figure] = None
        self._pending: list[tuple] = []
        self._scalar_plot_data: Optional[PlotData] = None
        self._sim_data = None

    # -- fluent add methods --------------------------------------------------

    def add_scalar(self, plot_data: PlotData) -> SimFigure:
        """auto-dispatches line/quad/polygon based on field dimensionality."""
        self._scalar_plot_data = plot_data
        self._pending.append(("scalar", plot_data))
        return self

    def add_vector(
        self, sim_data, vector_fields: Sequence[str], vector_type: str = "quiver"
    ) -> SimFigure:
        """add vector field overlay (quiver or streamplot)."""
        self._sim_data = sim_data
        self._pending.append(("vector", sim_data, vector_fields, vector_type))
        return self

    def add_overlay(self, overlay: OverlayConfig, sim_data) -> SimFigure:
        """add contour overlay."""
        self._sim_data = sim_data
        self._pending.append(("overlay", overlay, sim_data))
        return self

    # -- terminal operations -------------------------------------------------

    def render(self) -> SimFigure:
        """create figure and render all components."""
        self._ensure_figure()
        self._figure.render()
        return self

    def animate(self, files: Sequence[str], fps: int = 30) -> SimFigure:
        """create animation from file sequence."""
        self._ensure_figure()
        self._figure.animate(files, fps=fps)
        return self

    def save(self, path: str, **kwargs):
        """save figure to disk."""
        self._figure.save(path, **kwargs)

    def show(self):
        """display the figure."""
        plt.show()

    # -- accessors -----------------------------------------------------------

    @property
    def figure(self) -> Optional[Figure]:
        """the underlying Figure object (None until render/animate)."""
        return self._figure

    @property
    def fig(self):
        """the matplotlib figure (None until render/animate)."""
        return self._figure.fig if self._figure else None

    @property
    def ax(self):
        """the main matplotlib axes (None until render/animate)."""
        return self._figure.axes.get("main") if self._figure else None

    # -- internals -----------------------------------------------------------

    def _ensure_figure(self) -> None:
        """lazy-create mpl figure, then execute all pending add_* operations."""
        if self._figure is not None:
            return

        plot_data = self._scalar_plot_data
        if plot_data is None:
            raise RuntimeError("no scalar data added — call add_scalar() first")

        final_fields = compose_fields_for_render(plot_data.fields, self._config)
        nlvls, use_polygons = refinement_info(plot_data.fields, self._config)
        coord_sys = plot_data.coord_system or CoordSystem.CARTESIAN
        projection = detect_projection(final_fields, coord_sys.value)

        self._figure = prepare_figure(
            self._config,
            nfiles=1,
            projection=projection,
            nlvls=nlvls,
            coord_system=coord_sys,
        )

        # replay all pending operations
        for entry in self._pending:
            kind = entry[0]
            if kind == "scalar":
                pd = entry[1]
                fields = compose_fields_for_render(pd.fields, self._config)
                _, poly = refinement_info(pd.fields, self._config)
                dispatch_scalar_components(
                    self._figure,
                    fields,
                    self._component_props,
                    poly,
                    bodies=pd.body_collection,
                )
            elif kind == "vector":
                _, sim_data, vfields, vtype = entry
                dispatch_vector_components(
                    self._figure,
                    sim_data,
                    vfields,
                    self._config,
                    self._component_props,
                    vector_type=vtype,
                )
            elif kind == "overlay":
                _, overlay, sim_data = entry
                dispatch_overlay_components(
                    self._figure,
                    sim_data,
                    [overlay],
                    self._config,
                )

        self._pending.clear()
