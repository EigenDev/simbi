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
import numpy as np

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
from .types import CoordSystem, FieldData, PolygonData
from .utility import get_tracer_field_str


# colormaps handed to successive panels of a shared chart. two quantities that
# share a chart and a colormap read as one field, so each panel takes a
# different one; they are perceptually uniform, so the structure a panel shows
# is the structure its data holds.
PANEL_CMAPS = ("viridis", "magma", "cividis", "plasma")


def _get_props(
    component_props: Optional[dict[str, ComponentProps]],
    key: str,
    default_factory,
) -> ComponentProps:
    """Get props from dict or create default."""
    if component_props and key in component_props:
        return component_props[key]
    return default_factory()


def _panel_cmap(base: str, index: int) -> str:
    """the default colormap for panel `index`.

    the first panel keeps the configured map and the rest take distinct ones."""
    if index == 0:
        return base
    alternatives = [cmap for cmap in PANEL_CMAPS if cmap != base]
    return alternatives[(index - 1) % len(alternatives)]


def _panel_props(
    component_props: Optional[dict[str, ComponentProps]],
    key: str,
    default_factory,
    field_name: str,
    index: int,
    npanels: int,
    color_ranges: Optional[dict] = None,
) -> ComponentProps:
    """props for the panel drawing `field_name`.

    the shared component props carry a distinct colormap per panel, then a
    colour scale swept over the sequence if one was taken, then whatever the
    user asked for that specific quantity: two quantities on one chart rarely
    want the same scaling, and an explicit request outranks anything derived
    from the data."""
    from .pipeline.panels import base_field_name

    quantity = base_field_name(field_name)
    props = _get_props(component_props, key, default_factory)

    if npanels > 1:
        props = props.model_copy(
            update={"cmap": _panel_cmap(props.cmap, index)}
        )

    swept = (color_ranges or {}).get(quantity)
    if swept is not None and not _has_color_range(props):
        props = props.model_copy(update={"color_range": swept})

    override = (component_props or {}).get(f"{key}:{quantity}")
    if override is not None:
        props = props.model_copy(
            update={
                name: getattr(override, name)
                for name in override.model_fields_set
            }
        )

    return props


def _has_color_range(props: ComponentProps) -> bool:
    """whether the caller pinned the colour scale themselves."""
    color_range = getattr(props, "color_range", None)
    return color_range is not None and (
        color_range.min is not None or color_range.max is not None
    )


def _animation_color_ranges(
    files: Sequence[str],
    fields: Sequence[str],
    config: VisualizationConfig,
) -> dict:
    """the colour scale each quantity is drawn on for every frame.

    empty when the scale is left to each frame, which redraws a decaying
    quantity at full brightness throughout."""
    from .pipeline.color_range import sequence_color_range

    choice = getattr(config.plot, "color_scale", "sequence")
    if choice == "frame":
        return {}

    scanned = files[:1] if choice == "first" else files
    return sequence_color_range(scanned, fields, config)


def _panel_layout(
    scalar_fields: Sequence[FieldData], final_fields: Sequence[FieldData]
) -> int:
    """how many quantities share the chart, one sector each.

    composition emits exactly one artist per quantity when the chart is
    divided into sectors; anything else (levels drawn separately, a vector
    overlay) is one quantity seen several ways and keeps a single scale."""
    from .pipeline.panels import group_by_field

    npanels = len(group_by_field(scalar_fields))
    return npanels if npanels == len(final_fields) else 1


def plot_tracers(
    config: VisualizationConfig,
    file: str,
    save_as: Optional[str] = None,
    show: bool = True,
    tracer_render: str = "concentration",
    tracer_smoothing: Optional[float] = None,
    tracer_color_by: str = "flag",
    tracer_cohort: Optional[int] = None,
    **kwargs,
) -> Figure:
    """render an ownership-derived tracer cloud without an eulerian field."""
    from .tracers import (
        cohort_to_gas_ratio,
        load_tracers,
        overlay_tracers,
        projected_gas_concentration,
        tracer_concentration,
        tracer_projection,
    )

    sim_data = load_data(file)
    if sim_data.mesh.ndim < 2:
        raise ValueError("--tracers-only requires a 2d or 3d checkpoint")
    coord_system = sim_data.metadata.coord_system
    collapsed_axis = None
    if config.plot.slice:
        if len(config.plot.slice) != 1:
            raise ValueError("--tracers-only requires a 2d projection, not a 1d slice")
        axis_name = next(iter(config.plot.slice))
        if axis_name not in {"x1", "x2", "x3"}:
            raise ValueError(f"unknown tracer projection axis '{axis_name}'")
        collapsed_axis = int(axis_name[1]) - 1
    chart = tracer_projection(
        coord_system,
        sim_data.mesh.ndim,
        collapsed_axis,
    )
    cloud = load_tracers(file)
    if cloud is None:
        raise ValueError(f"checkpoint '{file}' has no tracer population")
    if len(cloud) == 0:
        raise ValueError(f"checkpoint '{file}' has an empty tracer population")

    figure = prepare_figure(
        config,
        1,
        projection=chart.projection,
        coord_system=CoordSystem(coord_system),
    )
    figure.render()
    ax = figure.axes["main"]
    vertices = tuple(
        np.asarray(getattr(sim_data.mesh, f"x{axis + 1}v"))
        for axis in range(sim_data.mesh.ndim)
    )
    x_edges = vertices[chart.plane[0]]
    y_edges = vertices[chart.plane[1]]
    if tracer_render in {"concentration", "cohort-ratio"}:
        if tracer_render == "cohort-ratio" and tracer_cohort is None:
            raise ValueError("--tracer-render cohort-ratio requires --tracer-cohort")
        concentration = tracer_concentration(
            cloud,
            x_edges,
            y_edges,
            plane=chart.plane,
            smoothing=tracer_smoothing,
            cohort=tracer_cohort,
        )
        display_area = np.outer(np.diff(y_edges), np.diff(x_edges))
        mean = np.sum(concentration * display_area) / np.sum(display_area)
        normalized = concentration / max(mean, np.finfo(float).tiny)
        cmap = "magma"
        label = get_tracer_field_str("tracer_concentration")
        if tracer_render == "cohort-ratio":
            rho = np.squeeze(sim_data.get_field("rho", crop_to_owned=True))
            gas_concentration = projected_gas_concentration(
                rho,
                vertices,
                coord_system,
                chart,
            )
            ratio = cohort_to_gas_ratio(
                concentration,
                gas_concentration,
                display_area,
            )
            normalized = np.log10(np.maximum(ratio, np.finfo(float).tiny))
            cmap = "coolwarm"
            label = get_tracer_field_str(
                "tracer_cohort_ratio",
                tracer_cohort,
            )
        elif tracer_cohort is not None:
            label = get_tracer_field_str(
                "tracer_cohort_concentration",
                tracer_cohort,
            )
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            normalized,
            shading="auto",
            cmap=cmap,
        )
        assert figure.fig is not None
        colorbar = figure.fig.colorbar(mesh, ax=ax)
        colorbar.set_label(label)
    elif tracer_render == "scatter":
        overlay_tracers(
            ax,
            file,
            plane=chart.plane,
            color_by=tracer_color_by,
        )
    else:
        raise ValueError(f"unknown tracer rendering mode '{tracer_render}'")

    if config.figure.xlims is None:
        ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    if config.figure.ylims is None:
        ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    ax.set_xlabel(config.figure.xlabel or chart.labels[0])
    ax.set_ylabel(config.figure.ylabel or chart.labels[1])
    if chart.projection == "cartesian":
        ax.set_aspect("equal")
    if config.figure.title:
        ax.set_title(config.figure.title)

    if save_as:
        figure.save(save_as)
    if show:
        plt.show()
    return figure


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
    # refined data renders as polygons; a pcolormesh quadmesh carries one grid
    if is_refined:
        use_polygons = True
    else:
        use_polygons = config.refinement.render_mode == "polygons"

    # determine projection
    projection = "cartesian"
    if final_fields:
        is_2d = final_fields[0].ndim == 2
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
    npanels = _panel_layout(scalar_fields, final_fields)

    for panel, field_data in enumerate(final_fields):
        component: Component

        if isinstance(field_data, PolygonData):
            component = PolygonPlotComponent(
                _panel_props(
                    component_props,
                    "polygon",
                    PolygonPlotProps,
                    field_data.name,
                    panel,
                    npanels,
                )
            )

        elif field_data.ndim == 1:
            # line plot
            props = _get_props(component_props, "line", LinePlotProps)
            component = LinePlotComponent(props)

        elif field_data.ndim == 2:
            if use_polygons:
                component = PolygonPlotComponent(
                    _panel_props(
                        component_props,
                        "polygon",
                        PolygonPlotProps,
                        field_data.name,
                        panel,
                        npanels,
                    )
                )
            else:
                component = QuadPlotComponent(
                    _panel_props(
                        component_props,
                        "quad",
                        QuadPlotProps,
                        field_data.name,
                        panel,
                        npanels,
                    )
                )

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

    if config.figure.draw_horizon:
        _overlay_horizon(figure, config, sim_data)

    if config.figure.draw_tracers:
        _overlay_tracers(figure, files[0], config)

    if save_as:
        figure.save(save_as)
    if show:
        plt.show()

    return figure


def _overlay_bodies(figure, checkpoint_path, config, sim_data) -> None:
    """draw each immersed body's silhouette on the rendered field axis, matching the
    field's slice plane (the shared gated overlay; cartesian + 2-D only)."""
    from .bodies import overlay_bodies_on_slice

    overlay_bodies_on_slice(
        figure.axes.get("main") if figure.axes else None,
        checkpoint_path,
        config.plot.slice,
        sim_data.metadata.coord_system,
    )


def _overlay_tracers(figure, checkpoint_path, config) -> None:
    """scatter mass-transport tracers on the rendered field axis for the two
    in-plane axes of the field slice."""
    from .bodies import slice_to_plane
    from .tracers import overlay_tracers

    ax = figure.axes.get("main") if figure.axes else None
    if ax is None:
        return
    mapped = slice_to_plane(config.plot.slice)
    if mapped is None:  # the field reduced to a 1-D line: no scatter plane
        return
    plane, at = mapped
    # 2-D runs have no out-of-plane axis (all particles in-plane); a 3-D run projects
    # every particle onto the plane here -- call overlay_tracers with `slab=` directly for
    # a thin sheet matching a 3-D field slab.
    overlay_tracers(ax, checkpoint_path, plane=plane, at=at)


def _overlay_horizon(figure, config, sim_data) -> None:
    """draw the black-hole event horizon on the rendered field axis, read from the
    checkpoint metadata (curved spacetimes only; flat runs draw nothing)."""
    from .horizon import overlay_horizon_on_slice

    overlay_horizon_on_slice(
        figure.axes.get("main") if figure.axes else None,
        sim_data.metadata,
        config.plot.slice,
        sim_data.metadata.coord_system,
    )


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
    # refined data renders as polygons; a pcolormesh quadmesh carries one grid
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
        is_2d = final_fields[0].ndim == 2
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
    npanels = _panel_layout(scalar_fields, final_fields)
    # one scale for the whole movie, so colour means the same in every frame
    color_ranges = _animation_color_ranges(files, fields, config)

    for panel, field_data in enumerate(final_fields):
        component: Component

        if isinstance(field_data, PolygonData):
            component = PolygonPlotComponent(
                _panel_props(
                    component_props,
                    "polygon",
                    PolygonPlotProps,
                    field_data.name,
                    panel,
                    npanels,
                    color_ranges,
                )
            )

        elif field_data.ndim == 1:
            props = _get_props(component_props, "line", LinePlotProps)
            component = LinePlotComponent(props)

        elif field_data.ndim == 2:
            if use_polygons:
                component = PolygonPlotComponent(
                    _panel_props(
                        component_props,
                        "polygon",
                        PolygonPlotProps,
                        field_data.name,
                        panel,
                        npanels,
                        color_ranges,
                    )
                )
            else:
                component = QuadPlotComponent(
                    _panel_props(
                        component_props,
                        "quad",
                        QuadPlotProps,
                        field_data.name,
                        panel,
                        npanels,
                        color_ranges,
                    )
                )

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

    fps = kwargs.get("fps") or config.animation.frame_rate
    figure.animate(files, fps=fps)

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
    fps = kwargs.get("fps") or config.animation.frame_rate
    figure.animate_coordinate_profile(files, fields, config, fps=fps)

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
