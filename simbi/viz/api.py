# =============================================================================
# api.py
#
# public api for the visualization system.
# all public functions delegate to _orchestrate(), which handles the
# shared lifecycle: normalize → load → figure → components → render → show.
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
from .figure import Figure, prepare_figure
from .pipeline import create_plot_data, load_data
from .pipeline.coord_binning import create_coordinate_profile_data
from .pipeline.power_spectrum import create_power_spectrum_data
from .pipeline.time_series import create_time_series_data
from .pipeline.transforms import _compose_pcolormesh, compose_fields_for_render
from .registry import (
    select_overlay_component,
    select_scalar_component,
    select_vector_component,
)
from .types import CoordSystem, FieldData

# ---------------------------------------------------------------------------
# shared helpers (unchanged)
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


def _init_component(
    figure: Figure, component: Component, field_data, is_overlay: bool = False
) -> None:
    """initialize a component and attach it to the figure."""
    if figure.fig is None:
        raise RuntimeError("figure not initialized")
    component.initialize(figure.fig, figure.axes["main"])
    figure.add_component(component, field_data, is_overlay=is_overlay)


def _save_and_show(figure: Figure, save_as: Optional[str], show: bool) -> None:
    """save and/or display the figure."""
    if save_as:
        figure.save(save_as)
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# component dispatch helpers
# ---------------------------------------------------------------------------


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
        if props_key in ("polygon", "quad"):
            component = comp_cls(props, bodies)
        else:
            component = comp_cls(props)
        _init_component(figure, component, field_data)


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

    comp_cls, props_cls, props_key = select_vector_component(vector_type)
    props = _get_props(component_props, props_key, props_cls)
    _init_component(figure, comp_cls(props), [v1_field, v2_field])


def _dispatch_overlay_components(
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
        _init_component(figure, comp_cls(props), overlay_field, is_overlay=True)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


def _load_scalar(config, files, fields, component_props, **kwargs):
    """load data and prepare figure for scalar/animation plots."""
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

    vector_fields = kwargs.get("vector_fields")
    if vector_fields:
        _dispatch_vector_components(
            figure,
            sim_data,
            vector_fields,
            config,
            component_props,
            vector_type=kwargs.get("vector_type", "quiver"),
        )

    # merge overlays from config and function argument
    all_overlays = list(config.overlays)
    if kwargs.get("overlays"):
        all_overlays.extend(kwargs["overlays"])
    if all_overlays:
        _dispatch_overlay_components(figure, sim_data, all_overlays, config)

    return figure


def _load_analysis(config, files, fields, component_props, plot_kind, **kwargs):
    """load data and prepare figure for analysis-type plots."""
    sim_data = load_data(files[0])

    if plot_kind == "coordinate_profile":
        plot_data = create_coordinate_profile_data(sim_data, fields, config)
        if not plot_data.fields:
            raise ValueError("no coordinate profiles generated")
        comp_cls, props_cls, props_key = (
            CoordinateProfileComponent,
            CoordinateProfileProps,
            "coordinate_profile",
        )
        figure = prepare_figure(
            config, len(files), projection="cartesian", nlvls=4
        )

    elif plot_kind == "power_spectrum":
        velocity_fields = fields if len(fields) >= 3 else ["v1", "v2", "v3"]
        plot_data = create_power_spectrum_data(
            sim_data, config, velocity_fields
        )
        if not plot_data.fields:
            raise ValueError("no power spectrum data generated")
        comp_cls, props_cls, props_key = (
            PowerSpectrumComponent,
            PowerSpectrumProps,
            "power_spectrum",
        )
        figure = prepare_figure(
            config, len(files), projection="cartesian", nlvls=1
        )

    elif plot_kind == "time_series":
        plot_data = create_time_series_data(files, fields, config)
        if not plot_data.fields:
            raise ValueError("no time series data generated")
        comp_cls, props_cls, props_key = (
            TimeSeriesPlotComponent,
            TimeSeriesPlotProps,
            "time_series",
        )
        nlines = plot_data.count_plot_lines()
        figure = prepare_figure(config, nlvls=nlines)

    else:
        raise ValueError(f"unknown plot_kind: {plot_kind}")

    for field_data in plot_data.fields:
        props = _get_props(component_props, props_key, props_cls)
        _init_component(figure, comp_cls(props), field_data)

    return figure


def _load_overlay(config, files, fields, component_props, plot_kind, **kwargs):
    """load data and prepare figure for multi-file overlay plots."""
    if len(files) < 2:
        raise ValueError("overlay requires at least 2 files")

    nfiles = len(files)

    if plot_kind == "scalar":
        # validate first file is 1d
        first_data = load_data(files[0])
        first_plot_data = create_plot_data(first_data, fields, config)
        for f in first_plot_data.fields:
            if f.ndim != 1:
                raise ValueError(
                    f"overlay only supports 1D data, got ndim={f.ndim} for '{f.name}'. "
                    "use --slice to reduce or try a different plot type."
                )

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

    elif plot_kind == "coordinate_profile":
        figure = prepare_figure(
            config,
            nfiles=nfiles,
            projection="cartesian",
            nlvls=nfiles,
            overlay_mode=True,
        )

        import numpy as np

        per_file_norms = kwargs.get("normalizations")
        per_file_labels = kwargs.get("labels")
        per_file_x_norms = kwargs.get("x_normalizations")

        for ii, file_path in enumerate(files):
            sim_data = load_data(file_path)
            plot_data = create_coordinate_profile_data(sim_data, fields, config)
            if not plot_data.fields:
                continue

            file_label = Path(file_path).stem
            norm = (
                per_file_norms[ii]
                if per_file_norms and ii < len(per_file_norms)
                else None
            )
            label = (
                per_file_labels[ii]
                if per_file_labels and ii < len(per_file_labels)
                else None
            )

            # use explicit x normalization if given, otherwise auto from max extent
            if per_file_x_norms and ii < len(per_file_x_norms):
                x_norm = per_file_x_norms[ii]
            else:
                x_norm = float(np.nanmax(plot_data.fields[0].domain[0]))

            for field_data in plot_data.fields:
                base_props = _get_props(
                    component_props,
                    "coordinate_profile",
                    CoordinateProfileProps,
                )
                props = CoordinateProfileProps(
                    label=label or f"{field_data.name} ({file_label})",
                    color=base_props.color,
                    linestyle=base_props.linestyle,
                    linewidth=base_props.linewidth,
                    normalization=norm or base_props.normalization,
                    x_normalization=x_norm,
                    rbeg=base_props.rbeg,
                    rend=base_props.rend,
                    show_reference_lines=base_props.show_reference_lines,
                    x_scale=base_props.x_scale,
                    y_scale=base_props.y_scale,
                )
                _init_component(
                    figure, CoordinateProfileComponent(props), field_data
                )

    else:
        raise ValueError(f"overlay not supported for plot_kind='{plot_kind}'")

    return figure


def _orchestrate(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str],
    plot_kind: str = "scalar",
    render_mode: str = "static",
    overlay: bool = False,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """
    unified visualization lifecycle.

    plot_kind: "scalar", "coordinate_profile", "time_series", "power_spectrum"
    render_mode: "static" or "animate"
    overlay: if True, overlay multiple files on same axes
    """
    if isinstance(files, str):
        files = [files]

    if render_mode == "animate" and len(files) < 2:
        raise ValueError("animation requires at least 2 files")

    # load data, prepare figure, attach components
    if overlay:
        figure = _load_overlay(
            config, files, fields, component_props, plot_kind, **kwargs
        )
    elif plot_kind == "scalar":
        figure = _load_scalar(config, files, fields, component_props, **kwargs)
    else:
        figure = _load_analysis(
            config, files, fields, component_props, plot_kind, **kwargs
        )

    # render or animate
    fps = kwargs.get("fps") or config.animation.frame_rate
    if render_mode == "animate":
        if plot_kind == "coordinate_profile":
            figure.animate_coordinate_profile(
                files,
                fields,
                config,
                output_path=save_as or "animation.mp4",
                fps=fps,
                save_all_frames=config.animation.save_all_frames,
            )
        else:
            figure.animate(
                files,
                output_path=save_as or "animation.mp4",
                fps=fps,
                save_all_frames=config.animation.save_all_frames,
            )
    else:
        figure.render()

    _save_and_show(figure, save_as, show)
    return figure


# ---------------------------------------------------------------------------
# public api (thin wrappers)
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
    """create a visualization from checkpoint file(s)."""
    return _orchestrate(
        config,
        files,
        fields,
        plot_kind="scalar",
        save_as=save_as,
        show=show,
        component_props=component_props,
        vector_fields=vector_fields,
        overlays=overlays,
        **kwargs,
    )


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
    """create animation from ordered sequence of checkpoint files."""
    return _orchestrate(
        config,
        files,
        fields,
        plot_kind="scalar",
        render_mode="animate",
        save_as=save_as,
        show=show,
        component_props=component_props,
        overlays=overlays,
        **kwargs,
    )


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
    return _orchestrate(
        config,
        files,
        fields,
        plot_kind="coordinate_profile",
        save_as=save_as,
        show=show,
        component_props=component_props,
        **kwargs,
    )


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
    return _orchestrate(
        config,
        files,
        fields,
        plot_kind="coordinate_profile",
        render_mode="animate",
        save_as=save_as,
        show=show,
        component_props=component_props,
        **kwargs,
    )


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
    return _orchestrate(
        config,
        files,
        fields,
        plot_kind="time_series",
        save_as=save_as,
        show=show,
        component_props=component_props,
        **kwargs,
    )


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
    return _orchestrate(
        config,
        files,
        fields,
        plot_kind="power_spectrum",
        save_as=save_as,
        show=show,
        component_props=component_props,
        **kwargs,
    )


def plot_overlay(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ["rho"],
    normalizations: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """overlay multiple files on the same axes (line plots only)."""
    return _orchestrate(
        config,
        files,
        fields,
        plot_kind="scalar",
        overlay=True,
        save_as=save_as,
        show=show,
        component_props=component_props,
        normalizations=normalizations,
        labels=labels,
        **kwargs,
    )


def plot_grid(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ("rho",),
    layout: Optional[tuple[int, int]] = None,
    panel_labels: Optional[Sequence[str]] = None,
    auto_label: bool = False,
    shared_colorbar: bool = True,
    annotate_inside: bool = False,
    wspace: Optional[float] = None,
    hspace: Optional[float] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    panel_overrides: Optional[dict[int, dict]] = None,
    **kwargs,
):
    """create a multi-panel grid figure comparing different checkpoint files."""
    from .grid import plot_grid as _plot_grid

    return _plot_grid(
        config,
        files,
        fields,
        layout=layout,
        panel_labels=panel_labels,
        auto_label=auto_label,
        shared_colorbar=shared_colorbar,
        annotate_inside=annotate_inside,
        wspace=wspace,
        hspace=hspace,
        save_as=save_as,
        show=show,
        component_props=component_props,
        panel_overrides=panel_overrides,
        **kwargs,
    )


def plot_coordinate_profile_overlay(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ["rho"],
    normalizations: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    x_normalizations: Optional[Sequence[float]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """overlay coordinate profiles from multiple files on the same axes."""
    return _orchestrate(
        config,
        files,
        fields,
        plot_kind="coordinate_profile",
        overlay=True,
        save_as=save_as,
        show=show,
        component_props=component_props,
        normalizations=normalizations,
        labels=labels,
        x_normalizations=x_normalizations,
        **kwargs,
    )
