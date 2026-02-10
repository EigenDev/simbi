# =============================================================================
# api.py
#
# public api for the visualization system.
# component styling comes entirely from component_props dict.
# figure-level config (axes, limits, theme) is separate from component props.
# =============================================================================
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt

from .components import (
    CoordinateProfileComponent,
    CoordinateProfileProps,
    LinePlotComponent,
    LinePlotProps,
    PowerSpectrumComponent,
    PowerSpectrumProps,
    TimeSeriesPlotComponent,
    TimeSeriesPlotProps,
)
from .components.interface import Component, ComponentProps
from .config import OverlayConfig, VisualizationConfig
from .dispatch import (
    select_overlay_component,
    select_scalar_component,
    select_vector_component,
)
from .figure import Figure, prepare_figure
from .pipeline import create_plot_data, load_data
from .pipeline.coord_binning import create_coordinate_profile_data
from .pipeline.power_spectrum import create_power_spectrum_data
from .pipeline.time_series import create_time_series_data
from .pipeline.transforms import _compose_pcolormesh, compose_fields_for_render
from .types import CoordSystem, FieldData

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _get_props(
    component_props: Optional[dict[str, ComponentProps]],
    key: str,
    default_factory,
) -> ComponentProps:
    """get props from dict or create default."""
    if component_props and key in component_props:
        return component_props[key]
    return default_factory()


def _refinement_info(
    fields: Sequence[FieldData], config: VisualizationConfig
) -> tuple[int, bool]:
    """compute nlvls and use_polygons from field list."""
    nlvls = 1 + sum("_L" in f.name for f in fields)
    use_polygons = nlvls > 1 or config.refinement.render_mode == "polygons"
    return nlvls, use_polygons


def _detect_projection(fields: Sequence[FieldData], coord_system: str) -> str:
    """determine projection from fields and coordinate system."""
    if not fields:
        return "cartesian"
    is_2d = fields[0].ndim == 2 or fields[0].name.endswith("_polygons")
    if is_2d and coord_system == "spherical":
        return "polar"
    return "cartesian"


def _dispatch_scalar_components(
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

        props = _get_props(component_props, props_key, props_cls)

        # some components accept bodies parameter
        if props_key in ("polygon", "quad"):
            component = comp_cls(props, bodies)
        else:
            component = comp_cls(props)

        if figure.fig is None:
            raise RuntimeError("figure not initialized")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, field_data)


def _dispatch_vector_components(
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
    vector_data = [v1_field, v2_field]

    comp_cls, props_cls, props_key = select_vector_component(vector_type)
    props = _get_props(component_props, props_key, props_cls)
    component = comp_cls(props)

    if figure.fig is None:
        raise RuntimeError("figure not initialized")

    component.initialize(figure.fig, figure.axes["main"])
    figure.add_component(component, vector_data)


def _dispatch_overlay_components(
    figure: Figure,
    sim_data,
    overlays: Sequence[OverlayConfig],
    config: VisualizationConfig,
) -> None:
    """create and attach overlay components (e.g., contour lines)."""
    for overlay in overlays:
        # load overlay field data
        overlay_plot_data = create_plot_data(sim_data, [overlay.field], config)

        if not overlay_plot_data.fields:
            continue

        # for contour overlays, we need 2d data (pcolormesh format), not polygons
        # use _compose_pcolormesh directly to ensure we get 2d output
        overlay_field = _compose_pcolormesh(list(overlay_plot_data.fields))

        if overlay_field.ndim != 2:
            # can't render contours on non-2d data
            continue

        # select overlay component
        comp_cls, props_cls, _ = select_overlay_component(overlay.component)

        # build props from overlay config
        props = props_cls(
            levels=tuple(overlay.levels),
            color=overlay.color,
            linewidths=overlay.linewidth,
            linestyles=overlay.linestyle,
            alpha=overlay.alpha,
            filled=overlay.filled,
            label_contours=overlay.label_contours,
        )

        component = comp_cls(props)

        if figure.fig is None:
            raise RuntimeError("figure not initialized")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, overlay_field, is_overlay=True)


def _save_and_show(figure: Figure, save_as: Optional[str], show: bool) -> None:
    """save and/or display the figure."""
    if save_as:
        figure.save(save_as)
    if show:
        plt.show()


def _init_component(figure: Figure, component: Component, field_data) -> None:
    """initialize a component and attach it to the figure."""
    if figure.fig is None:
        raise RuntimeError("figure not initialized")
    component.initialize(figure.fig, figure.axes["main"])
    figure.add_component(component, field_data)


# ---------------------------------------------------------------------------
# public api
# ---------------------------------------------------------------------------


def plot(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    vector_fields: Optional[Sequence[str]] = None,
    overlays: Optional[Sequence[OverlayConfig]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """
    create a visualization from checkpoint file(s).

    args:
        config: visualization configuration
        files: checkpoint file(s) to visualize
        fields: field(s) to visualize
        vector_fields: optional vector field components (e.g., ["v1", "v2"])
        overlays: optional overlay specifications (e.g., contour lines)
        save_as: optional path to save the figure
        show: whether to display the figure
        component_props: optional component-specific props overrides

    returns:
        the created Figure object
    """
    if isinstance(files, str):
        files = [files]

    # merge overlays from config and function argument
    all_overlays = list(config.overlays)
    if overlays:
        all_overlays.extend(overlays)

    sim_data = load_data(files[0])
    scalar_plot_data = create_plot_data(sim_data, fields, config)
    final_fields = compose_fields_for_render(scalar_plot_data.fields, config)
    nlvls, use_polygons = _refinement_info(scalar_plot_data.fields, config)
    projection = _detect_projection(
        final_fields, sim_data.metadata.coord_system
    )

    figure = prepare_figure(
        config,
        len(files),
        projection=projection,
        nlvls=nlvls,
        coord_system=CoordSystem(sim_data.metadata.coord_system),
    )

    _dispatch_scalar_components(
        figure,
        final_fields,
        component_props,
        use_polygons,
        bodies=scalar_plot_data.body_collection,
    )

    if vector_fields:
        _dispatch_vector_components(
            figure,
            sim_data,
            vector_fields,
            config,
            component_props,
            vector_type=kwargs.get("vector_type", "quiver"),
        )

    # add overlay components
    if all_overlays:
        _dispatch_overlay_components(figure, sim_data, all_overlays, config)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def animate(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    overlays: Optional[Sequence[OverlayConfig]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """
    create animation from ordered sequence of checkpoint files.

    overlays animate together with the primary field.
    """
    if isinstance(files, str):
        files = [files]

    if len(files) < 2:
        raise ValueError("animation requires at least 2 files")

    # merge overlays from config and function argument
    all_overlays = list(config.overlays)
    if overlays:
        all_overlays.extend(overlays)

    sim_data = load_data(files[0])
    vector_fields = kwargs.get("vector_fields")

    scalar_plot_data = create_plot_data(sim_data, fields, config)
    final_fields = compose_fields_for_render(scalar_plot_data.fields, config)
    nlvls, use_polygons = _refinement_info(scalar_plot_data.fields, config)
    projection = _detect_projection(
        final_fields, sim_data.metadata.coord_system
    )

    figure = prepare_figure(
        config,
        len(files),
        projection=projection,
        nlvls=nlvls,
        coord_system=CoordSystem(sim_data.metadata.coord_system),
    )

    _dispatch_scalar_components(
        figure,
        final_fields,
        component_props,
        use_polygons,
        bodies=scalar_plot_data.body_collection,
    )

    if vector_fields:
        _dispatch_vector_components(
            figure,
            sim_data,
            vector_fields,
            config,
            component_props,
            vector_type=kwargs.get("vector_type", "quiver"),
        )

    # add overlay components
    if all_overlays:
        _dispatch_overlay_components(figure, sim_data, all_overlays, config)

    fps = kwargs.get("fps") or config.animation.frame_rate
    figure.animate(
        files,
        output_path=save_as or "animation.mp4",
        fps=fps,
        save_all_frames=config.animation.save_all_frames,
    )

    _save_and_show(figure, save_as, show)
    return figure


def animate_coordinate_profile(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create animation of coordinate-binned profiles from multiple files."""
    if isinstance(files, str):
        files = [files]

    if len(files) < 2:
        raise ValueError("animation requires at least 2 files")

    sim_data = load_data(files[0])
    plot_data = create_coordinate_profile_data(sim_data, fields, config)

    if not plot_data.fields:
        raise ValueError("no coordinate profiles generated")

    figure = prepare_figure(config, len(files), projection="cartesian", nlvls=4)

    for field_data in plot_data.fields:
        props = _get_props(
            component_props, "coordinate_profile", CoordinateProfileProps
        )
        _init_component(figure, CoordinateProfileComponent(props), field_data)

    fps = kwargs.get("fps") or config.animation.frame_rate
    figure.animate_coordinate_profile(
        files,
        fields,
        config,
        output_path=save_as or "animation.mp4",
        fps=fps,
        save_all_frames=config.animation.save_all_frames,
    )

    _save_and_show(figure, save_as, show)
    return figure


def plot_coordinate_profile(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create coordinate-binned profile plot."""
    if isinstance(files, str):
        files = [files]

    sim_data = load_data(files[0])
    plot_data = create_coordinate_profile_data(sim_data, fields, config)

    if not plot_data.fields:
        raise ValueError("no coordinate profiles generated")

    figure = prepare_figure(config, len(files), projection="cartesian", nlvls=4)

    for field_data in plot_data.fields:
        props = _get_props(
            component_props, "coordinate_profile", CoordinateProfileProps
        )
        _init_component(figure, CoordinateProfileComponent(props), field_data)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def plot_time_series(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create time series plot from multiple checkpoint files."""
    if isinstance(files, str):
        raise ValueError("time series requires multiple files")

    time_series_data = create_time_series_data(files, fields, config)

    if not time_series_data.fields:
        raise ValueError("no time series data generated")

    nlines = time_series_data.count_plot_lines()
    figure = prepare_figure(config, nlvls=nlines)

    for field_data in time_series_data.fields:
        props = _get_props(component_props, "time_series", TimeSeriesPlotProps)
        _init_component(figure, TimeSeriesPlotComponent(props), field_data)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def plot_power_spectrum(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["v1", "v2", "v3"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create kinetic energy power spectrum plot from a checkpoint file."""
    if isinstance(files, str):
        files = [files]

    sim_data = load_data(files[0])

    # power spectrum always uses velocity fields
    velocity_fields = fields if len(fields) >= 3 else ["v1", "v2", "v3"]
    spectrum_data = create_power_spectrum_data(
        sim_data, config, velocity_fields
    )

    if not spectrum_data.fields:
        raise ValueError("no power spectrum data generated")

    figure = prepare_figure(config, len(files), projection="cartesian", nlvls=1)

    for field_data in spectrum_data.fields:
        props = _get_props(
            component_props, "power_spectrum", PowerSpectrumProps
        )
        _init_component(figure, PowerSpectrumComponent(props), field_data)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def plot_overlay(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """overlay multiple files on the same axes (line plots only)."""
    if len(files) < 2:
        raise ValueError("overlay requires at least 2 files")

    first_data = load_data(files[0])
    first_plot_data = create_plot_data(first_data, fields, config)

    for f in first_plot_data.fields:
        if f.ndim != 1:
            raise ValueError(
                f"overlay only supports 1D data, got ndim={f.ndim} for '{f.name}'. "
                "use --slice to reduce or try a different plot type."
            )

    nfiles = len(files)
    figure = prepare_figure(
        config,
        nfiles=nfiles,
        projection="cartesian",
        nlvls=nfiles,
        coord_system=CoordSystem(first_data.metadata.coord_system),
        overlay_mode=True,
    )

    for file_path in files:
        sim_data = load_data(file_path)
        plot_data = create_plot_data(sim_data, fields, config)

        for field_data in plot_data.fields:
            if field_data.ndim != 1:
                continue

            base_props = _get_props(component_props, "line", LinePlotProps)
            props = LinePlotProps(
                label=f"{field_data.name}",
                linewidth=base_props.linewidth,
                marker=base_props.marker,
                marker_size=base_props.marker_size,
                alpha=base_props.alpha,
            )
            _init_component(figure, LinePlotComponent(props), field_data)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def plot_coordinate_profile_overlay(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """overlay coordinate profiles from multiple files on the same axes."""
    if len(files) < 2:
        raise ValueError("overlay requires at least 2 files")

    nfiles = len(files)
    figure = prepare_figure(
        config,
        nfiles=nfiles,
        projection="cartesian",
        nlvls=nfiles,
        overlay_mode=True,
    )

    for file_path in files:
        sim_data = load_data(file_path)
        plot_data = create_coordinate_profile_data(sim_data, fields, config)

        if not plot_data.fields:
            continue

        file_label = Path(file_path).stem

        for field_data in plot_data.fields:
            base_props = _get_props(
                component_props, "coordinate_profile", CoordinateProfileProps
            )
            props = CoordinateProfileProps(
                label=f"{field_data.name} ({file_label})",
                color=base_props.color,
                linestyle=base_props.linestyle,
                linewidth=base_props.linewidth,
                normalization=base_props.normalization,
                rbeg=base_props.rbeg,
                rend=base_props.rend,
                show_reference_lines=base_props.show_reference_lines,
                x_scale=base_props.x_scale,
                y_scale=base_props.y_scale,
            )
            _init_component(
                figure, CoordinateProfileComponent(props), field_data
            )

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure
