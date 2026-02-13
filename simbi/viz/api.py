# =============================================================================
# api.py
#
# public api for the visualization system.
# each public function is a thin wrapper that wires data to SimFigure or
# directly to Figure for specialized analysis plots.
# shared dispatch logic lives in builder.py.
# =============================================================================
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .builder import (
    SimFigure,
    detect_projection,
    dispatch_overlay_components,
    dispatch_scalar_components,
    dispatch_vector_components,
    get_props,
    init_component,
)
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
from .components.interface import ComponentProps
from .config import OverlayConfig, VisualizationConfig
from .figure import Figure, prepare_figure
from .pipeline import create_plot_data, load_data
from .pipeline.coord_binning import create_coordinate_profile_data
from .pipeline.power_spectrum import create_power_spectrum_data
from .pipeline.temporal_spectrum import create_temporal_spectrum_data
from .pipeline.time_series import create_time_series_data
from .pipeline.transforms import compose_fields_for_render
from .registry import refinement_info
from .types import CoordSystem


# ---------------------------------------------------------------------------
# internal helpers (api-specific)
# ---------------------------------------------------------------------------


def _save_and_show(figure: Figure, save_as: Optional[str], show: bool) -> None:
    """save and/or display the figure."""
    if save_as:
        figure.save(save_as)
    if show:
        plt.show()


def _setup_scalar_figure(config, files, fields, component_props, **kwargs):
    """load data, prepare figure, and attach scalar/vector/overlay components."""
    sim_data = load_data(files[0])
    scalar_plot_data = create_plot_data(sim_data, fields, config)
    final_fields = compose_fields_for_render(scalar_plot_data.fields, config)
    nlvls, use_polygons = refinement_info(scalar_plot_data.fields, config)
    projection = detect_projection(
        final_fields, sim_data.metadata.coord_system
    )

    figure = prepare_figure(
        config,
        len(files),
        projection=projection,
        nlvls=nlvls,
        coord_system=CoordSystem(sim_data.metadata.coord_system),
    )

    dispatch_scalar_components(
        figure,
        final_fields,
        component_props,
        use_polygons,
        bodies=scalar_plot_data.body_collection,
    )

    vector_fields = kwargs.get("vector_fields")
    if vector_fields:
        dispatch_vector_components(
            figure,
            sim_data,
            vector_fields,
            config,
            component_props,
            vector_type=kwargs.get("vector_type", "quiver"),
        )

    all_overlays = list(config.overlays)
    if kwargs.get("overlays"):
        all_overlays.extend(kwargs["overlays"])
    if all_overlays:
        dispatch_overlay_components(figure, sim_data, all_overlays, config)

    return figure


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
    """create a visualization from checkpoint file(s)."""
    if isinstance(files, str):
        files = [files]

    figure = _setup_scalar_figure(
        config,
        files,
        fields,
        component_props,
        vector_fields=vector_fields,
        overlays=overlays,
        **kwargs,
    )
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
    """create animation from ordered sequence of checkpoint files."""
    if isinstance(files, str):
        files = [files]
    if len(files) < 2:
        raise ValueError("animation requires at least 2 files")

    figure = _setup_scalar_figure(
        config,
        files,
        fields,
        component_props,
        overlays=overlays,
        **kwargs,
    )

    fps = kwargs.get("fps") or config.animation.frame_rate
    figure.animate(files, fps=fps)
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
        props = get_props(
            component_props, "coordinate_profile", CoordinateProfileProps
        )
        init_component(figure, CoordinateProfileComponent(props), field_data)

    figure.render()
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
        props = get_props(
            component_props, "coordinate_profile", CoordinateProfileProps
        )
        init_component(figure, CoordinateProfileComponent(props), field_data)

    fps = kwargs.get("fps") or config.animation.frame_rate
    figure.animate_coordinate_profile(files, fields, config, fps=fps)
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
        files = [files]

    plot_data = create_time_series_data(files, fields, config)
    if not plot_data.fields:
        raise ValueError("no time series data generated")

    nlines = plot_data.count_plot_lines()
    figure = prepare_figure(config, nlvls=nlines)

    for field_data in plot_data.fields:
        props = get_props(component_props, "time_series", TimeSeriesPlotProps)
        init_component(figure, TimeSeriesPlotComponent(props), field_data)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def plot_temporal_spectrum(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["mdot"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create temporal power spectrum from a sequence of checkpoint files."""
    if isinstance(files, str):
        files = [files]

    plot_data = create_temporal_spectrum_data(files, fields, config)
    if not plot_data.fields:
        raise ValueError("no temporal spectrum data generated")

    figure = prepare_figure(config, len(files), projection="cartesian", nlvls=1)

    for field_data in plot_data.fields:
        props = get_props(
            component_props, "power_spectrum", PowerSpectrumProps
        )
        init_component(figure, PowerSpectrumComponent(props), field_data)

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

    velocity_fields = fields if len(fields) >= 3 else ["v1", "v2", "v3"]
    sim_data = load_data(files[0])
    plot_data = create_power_spectrum_data(sim_data, config, velocity_fields)
    if not plot_data.fields:
        raise ValueError("no power spectrum data generated")

    figure = prepare_figure(config, len(files), projection="cartesian", nlvls=1)

    for field_data in plot_data.fields:
        props = get_props(
            component_props, "power_spectrum", PowerSpectrumProps
        )
        init_component(figure, PowerSpectrumComponent(props), field_data)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


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
    if len(files) < 2:
        raise ValueError("overlay requires at least 2 files")

    nfiles = len(files)

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
            base_props = get_props(component_props, "line", LinePlotProps)
            props = LinePlotProps(
                label=f"{field_data.name}",
                linewidth=base_props.linewidth,
                marker=base_props.marker,
                marker_size=base_props.marker_size,
                alpha=base_props.alpha,
            )
            init_component(figure, LinePlotComponent(props), field_data)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


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

    for ii, file_path in enumerate(files):
        sim_data = load_data(file_path)
        plot_data = create_coordinate_profile_data(sim_data, fields, config)
        if not plot_data.fields:
            continue

        file_label = Path(file_path).stem
        norm = (
            normalizations[ii]
            if normalizations and ii < len(normalizations)
            else None
        )
        label = labels[ii] if labels and ii < len(labels) else None

        # use explicit x normalization if given, otherwise auto from max extent
        if x_normalizations and ii < len(x_normalizations):
            x_norm = x_normalizations[ii]
        else:
            x_norm = float(np.nanmax(plot_data.fields[0].domain[0]))

        for field_data in plot_data.fields:
            base_props = get_props(
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
            init_component(
                figure, CoordinateProfileComponent(props), field_data
            )

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


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
