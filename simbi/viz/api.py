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
    PolygonPlotComponent,
    PolygonPlotProps,
    QuadPlotComponent,
    QuadPlotProps,
    QuiverPlotComponent,
    QuiverPlotProps,
    StreamPlotComponent,
    StreamPlotProps,
)
from .components.interface import Component, ComponentProps
from .components.time_series import TimeSeriesPlotComponent, TimeSeriesPlotProps
from .config import VisualizationConfig
from .figure import Figure
from .pipeline import create_plot_data, load_data, prepare_figure
from .pipeline.coord_binning import create_coordinate_profile_data
from .pipeline.time_series import create_time_series_data
from .pipeline.transforms import _compose_pcolormesh, _compose_polygons
from .types import CoordSystem, FieldData


def _get_props(
    component_props: Optional[dict[str, ComponentProps]],
    key: str,
    default_factory,
) -> ComponentProps:
    """Get props from dict or create default."""
    if component_props and key in component_props:
        return component_props[key]
    return default_factory()


def plot(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    vector_fields: Optional[Sequence[str]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """
    Create a visualization from checkpoint file(s).

    Args:
        config: visualization configuration (figure, refinement, etc.)
        files: checkpoint file path(s)
        fields: field names to visualize
        vector_fields: optional vector field components for quiver/stream
        save_as: optional output file path
        show: whether to display the plot
        component_props: dict mapping component names to props instances
        **kwargs: additional arguments (vector_type, etc.)
    """
    if isinstance(files, str):
        files = [files]

    sim_data = load_data(files[0])
    scalar_plot_data = create_plot_data(sim_data, fields, config)
    scalar_fields = scalar_plot_data.fields

    from .pipeline.transforms import compose_fields_for_render

    final_fields = compose_fields_for_render(scalar_fields, config)

    nlvls = 1 + sum("_L" in f.name for f in scalar_fields)
    is_refined = nlvls > 1

    # determine rendering mode
    # refined data MUST use polygons (pcolormesh can't handle different grids)
    if is_refined:
        use_polygons = True
    else:
        use_polygons = config.refinement.render_mode == "polygons"

    # determine projection
    projection = "cartesian"
    if final_fields:
        is_2d = final_fields[0].ndim == 2 or final_fields[0].name.endswith(
            "_polygons"
        )
        if is_2d and sim_data.metadata.coord_system == "spherical":
            projection = "polar"

    figure = prepare_figure(
        config,
        len(files),
        projection=projection,
        nlvls=nlvls,
        coord_system=CoordSystem(sim_data.metadata.coord_system),
    )

    # dispatch scalar components
    for field_data in final_fields:
        component: Component

        if field_data.ndim == 1 and field_data.name.endswith("_polygons"):
            # polygon plot (1d array of patches)
            props = _get_props(component_props, "polygon", PolygonPlotProps)
            component = PolygonPlotComponent(props)

        elif field_data.ndim == 1:
            # line plot
            props = _get_props(component_props, "line", LinePlotProps)
            component = LinePlotComponent(props)

        elif field_data.ndim == 2:
            if use_polygons:
                props = _get_props(component_props, "polygon", PolygonPlotProps)
                component = PolygonPlotComponent(props)
            else:
                props = _get_props(component_props, "quad", QuadPlotProps)
                component = QuadPlotComponent(props)

        elif field_data.ndim == 3:
            raise ValueError(
                f"field '{field_data.name}' is 3D. use --slice to reduce."
            )
        else:
            raise ValueError(
                f"field '{field_data.name}' has unsupported ndim={field_data.ndim}"
            )

        if figure.fig is None:
            raise RuntimeError("figure not initialized")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, field_data)

    # dispatch vector components
    if vector_fields:
        vector_plot_data = create_plot_data(sim_data, vector_fields, config)

        if len(vector_fields) != 2:
            raise ValueError(
                f"--vector-fields needs exactly 2 component names (vi, vj); got "
                f"{len(vector_fields)}: {list(vector_fields)}"
            )
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

        vector_type = kwargs.get("vector_type", "quiver")
        if vector_type == "quiver":
            props = _get_props(component_props, "quiver", QuiverPlotProps)
            component = QuiverPlotComponent(props)
        else:
            props = _get_props(component_props, "stream", StreamPlotProps)
            component = StreamPlotComponent(props)

        if figure.fig is None:
            raise RuntimeError("figure not initialized")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, vector_data)

    figure.render()

    if config.figure.draw_bodies:
        _overlay_bodies(figure, files[0], config, sim_data)

    if save_as:
        figure.save(save_as)
    if show:
        plt.show()

    return figure


def _overlay_bodies(figure, checkpoint_path, config, sim_data) -> None:
    """draw each immersed body's silhouette on the rendered field axis, matching the
    field's slice plane. the body signed-distance is cartesian, so a polar/spherical
    field plot (whose axes are not world x/y) is skipped; a doubly-sliced (1-D) field
    has no silhouette and is skipped too."""
    from .bodies import overlay_bodies, slice_to_plane

    if sim_data.metadata.coord_system != "cartesian":
        return
    plane_at = slice_to_plane(config.plot.slice)
    if plane_at is None:
        return
    plane, at = plane_at
    ax = figure.axes.get("main") if figure.axes else None
    if ax is not None:
        overlay_bodies(ax, checkpoint_path, plane=plane, at=at)


def animate(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """
    Create animation from ordered sequence of checkpoint files.

    Args:
        config: visualization configuration
        files: checkpoint file paths (must be > 1)
        fields: field names to visualize
        save_as: output file path
        show: whether to display after rendering
        component_props: dict mapping component names to props instances
        **kwargs: additional arguments (vector_fields, vector_type, fps, etc.)
    """
    if isinstance(files, str):
        files = [files]

    if len(files) < 2:
        raise ValueError("animation requires at least 2 files")

    sim_data = load_data(files[0])
    vector_fields = kwargs.get("vector_fields")

    scalar_plot_data = create_plot_data(sim_data, fields, config)
    scalar_fields = scalar_plot_data.fields

    nlvls = 1 + sum("_L" in f.name for f in scalar_fields)
    is_refined = nlvls > 1

    # determine rendering mode
    # refined data MUST use polygons (pcolormesh can't handle different grids)
    if is_refined:
        use_polygons = True
    else:
        use_polygons = config.refinement.render_mode == "polygons"

    # compose fields for first frame
    final_fields: Sequence[FieldData] = []
    if scalar_fields:
        is_2d = scalar_fields[0].ndim == 2
        if is_2d and use_polygons:
            final_fields = [_compose_polygons(list(scalar_fields))]
        else:
            final_fields = scalar_fields

    # determine projection
    projection = "cartesian"
    if final_fields:
        is_2d = final_fields[0].ndim == 2 or final_fields[0].name.endswith(
            "_polygons"
        )
        if is_2d and sim_data.metadata.coord_system == "spherical":
            projection = "polar"

    figure = prepare_figure(
        config,
        len(files),
        projection=projection,
        nlvls=nlvls,
        coord_system=CoordSystem(sim_data.metadata.coord_system),
    )

    # dispatch scalar components
    for field_data in final_fields:
        component: Component

        if field_data.ndim == 1 and field_data.name.endswith("_polygons"):
            props = _get_props(component_props, "polygon", PolygonPlotProps)
            component = PolygonPlotComponent(props)

        elif field_data.ndim == 1:
            props = _get_props(component_props, "line", LinePlotProps)
            component = LinePlotComponent(props)

        elif field_data.ndim == 2:
            if use_polygons:
                props = _get_props(component_props, "polygon", PolygonPlotProps)
                component = PolygonPlotComponent(props)
            else:
                props = _get_props(component_props, "quad", QuadPlotProps)
                component = QuadPlotComponent(props)

        elif field_data.ndim == 3:
            raise ValueError(
                f"field '{field_data.name}' is 3D. use --slice to reduce."
            )
        else:
            raise ValueError(
                f"field '{field_data.name}' has unsupported ndim={field_data.ndim}"
            )

        if figure.fig is None:
            raise RuntimeError("figure not initialized")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, field_data)

    # dispatch vector components
    if vector_fields:
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

        vector_type = kwargs.get("vector_type", "quiver")
        if vector_type == "quiver":
            props = _get_props(component_props, "quiver", QuiverPlotProps)
            component = QuiverPlotComponent(props)
        else:
            props = _get_props(component_props, "stream", StreamPlotProps)
            component = StreamPlotComponent(props)

        if figure.fig is None:
            raise RuntimeError("figure not initialized")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, vector_data)

    # run animation
    output = save_as or "animation.mp4"
    fps = kwargs.get("fps") or config.animation.frame_rate

    figure.animate(
        files,
        output_path=output,
        fps=fps,
        save_all_frames=config.animation.save_all_frames,
    )

    if save_as:
        figure.save(save_as)

    if show:
        plt.show()

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
    """
    Create animation of coordinate-binned profiles from multiple files.

    Args:
        config: visualization configuration
        files: checkpoint file paths (must be > 1)
        fields: field names to visualize
        save_as: output file path
        show: whether to display after rendering
        component_props: dict mapping component names to props instances
        **kwargs: additional arguments (fps, etc.)
    """
    if isinstance(files, str):
        files = [files]

    if len(files) < 2:
        raise ValueError("animation requires at least 2 files")

    # load first file to set up components
    sim_data = load_data(files[0])
    plot_data = create_coordinate_profile_data(sim_data, fields, config)

    if not plot_data.fields:
        raise ValueError("no coordinate profiles generated")

    # coordinate profiles are always 1D lines
    figure = prepare_figure(config, len(files), projection="cartesian", nlvls=4)

    # initialize components for each field
    for field_data in plot_data.fields:
        props = _get_props(
            component_props, "coordinate_profile", CoordinateProfileProps
        )
        component = CoordinateProfileComponent(props)

        if figure.fig is None:
            raise RuntimeError("figure not initialized")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, field_data)

    # run animation using the coordinate profile data pipeline
    output = save_as or "animation.mp4"
    fps = kwargs.get("fps") or config.animation.frame_rate

    figure.animate_coordinate_profile(
        files,
        fields,
        config,
        output_path=output,
        fps=fps,
        save_all_frames=config.animation.save_all_frames,
    )

    if save_as:
        figure.save(save_as)

    if show:
        plt.show()

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
    """
    Create coordinate-binned profile plot.

    Args:
        config: visualization configuration
        files: checkpoint file path(s)
        fields: field names to visualize
        save_as: optional output file path
        show: whether to display the plot
        component_props: dict mapping component names to props instances
    """
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
        component = CoordinateProfileComponent(props)

        if figure.fig is None:
            raise RuntimeError("figure not initialized")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, field_data)

    figure.render()

    if save_as:
        figure.save(save_as)
    if show:
        plt.show()

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
    """
    Create time series plot from multiple checkpoint files.

    Args:
        config: visualization configuration
        files: checkpoint file paths (must be > 1)
        fields: field names to visualize
        save_as: optional output file path
        show: whether to display the plot
        component_props: dict mapping component names to props instances
    """
    if isinstance(files, str):
        raise ValueError("time series requires multiple files")

    time_series_data = create_time_series_data(files, fields, config)

    if not time_series_data.fields:
        raise ValueError("no time series data generated")

    nlines = time_series_data.count_plot_lines()
    figure = prepare_figure(config, nlvls=nlines)

    for field_data in time_series_data.fields:
        props = _get_props(component_props, "time_series", TimeSeriesPlotProps)
        component = TimeSeriesPlotComponent(props)

        if figure.fig is None:
            raise RuntimeError("figure not initialized")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, field_data)

    figure.render()

    if save_as:
        figure.save(save_as)
    if show:
        plt.show()

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
    """
    Overlay multiple files on the same axes (line plots only).

    Args:
        config: visualization configuration
        files: checkpoint file paths (each gets its own line)
        fields: field names to visualize
        save_as: optional output file path
        show: whether to display the plot
        component_props: dict mapping component names to props instances
    """
    if len(files) < 2:
        raise ValueError("overlay requires at least 2 files")

    # load first file to determine figure setup
    first_data = load_data(files[0])
    first_plot_data = create_plot_data(first_data, fields, config)
    first_fields = first_plot_data.fields

    # overlay only works for 1d data
    for f in first_fields:
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

    if figure.fig is None:
        raise RuntimeError("figure not initialized")

    # iterate through files and add each as a separate line
    for file_path in files:
        sim_data = load_data(file_path)
        plot_data = create_plot_data(sim_data, fields, config)

        file_label = Path(file_path).stem

        for field_data in plot_data.fields:
            if field_data.ndim != 1:
                continue

            # create a new component for each file's data
            base_props = _get_props(component_props, "line", LinePlotProps)
            # override label to include file identifier
            props = LinePlotProps(
                label=f"{field_data.name} ({file_label})",
                linewidth=base_props.linewidth,
                marker=base_props.marker,
                marker_size=base_props.marker_size,
                alpha=base_props.alpha,
            )
            component = LinePlotComponent(props)
            component.initialize(figure.fig, figure.axes["main"])
            figure.add_component(component, field_data)

    figure.render()

    if save_as:
        figure.save(save_as)
    if show:
        plt.show()

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
    """
    Overlay coordinate profiles from multiple files on the same axes.

    Args:
        config: visualization configuration
        files: checkpoint file paths (each gets its own line)
        fields: field names to visualize
        save_as: optional output file path
        show: whether to display the plot
        component_props: dict mapping component names to props instances
    """
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

    if figure.fig is None:
        raise RuntimeError("figure not initialized")

    # iterate through files and add each as a separate profile
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
            # override label to include file identifier
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
            component = CoordinateProfileComponent(props)
            component.initialize(figure.fig, figure.axes["main"])
            figure.add_component(component, field_data)

    figure.render()

    if save_as:
        figure.save(save_as)
    if show:
        plt.show()

    return figure
