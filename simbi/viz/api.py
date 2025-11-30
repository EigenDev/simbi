"""Public API for the visualization system."""

from typing import Optional, Sequence

import matplotlib.pyplot as plt

from simbi.viz.components.time_series import (
    TimeSeriesPlotComponent,
    TimeSeriesPlotProps,
)
from simbi.viz.pipeline.coord_binning import create_coordinate_profile_data
from simbi.viz.pipeline.time_series import create_time_series_data

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
from .components.interface import Component
from .config import VisualizationConfig
from .figure import Figure
from .pipeline import create_plot_data, load_data, prepare_figure
from .pipeline.transforms import _compose_pcolormesh, _compose_polygons
from .types import CoordSystem, FieldData


def plot(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    vector_fields: Optional[Sequence[str]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Figure:
    sim_data = load_data(files[0])

    scalar_plot_data = create_plot_data(sim_data, fields, config)
    scalar_fields = scalar_plot_data.fields
    nlvls = 1
    nlvls += sum("_L" in x.name for x in scalar_fields)
    is_fmr = nlvls > 1

    final_scalar_fields_to_render: Sequence[FieldData] = []
    if scalar_fields:
        is_2d = scalar_fields[0].ndim == 2
        if is_2d and is_fmr:
            final_scalar_fields_to_render = [_compose_polygons(scalar_fields)]
        else:
            final_scalar_fields_to_render = scalar_fields

    projection = "cartesian"
    if final_scalar_fields_to_render:
        is_2d_render = final_scalar_fields_to_render[
            0
        ].ndim == 2 or final_scalar_fields_to_render[0].name.endswith(
            "_polygons"
        )
        if is_2d_render and sim_data.metadata.coord_system == "spherical":
            projection = "polar"

    figure = prepare_figure(
        config,
        len(files),
        projection=projection,
        nlvls=nlvls,
        coord_system=CoordSystem(sim_data.metadata.coord_system),
    )

    # Dispatch Scalar Components
    cmap_cycle = config.style.cmap
    crange_cycle = config.style.color_range
    bodies = scalar_plot_data.bodies
    for i, field_data in enumerate(final_scalar_fields_to_render):
        component: Component
        if field_data.ndim == 1 and field_data.name.endswith("_polygons"):
            props = PolygonPlotProps(
                cmap=next(cmap_cycle),
                color_range=next(crange_cycle),
                log_scale=config.style.log,
                power=kwargs.get("power", 1.0),
                alpha=kwargs.get("alpha", 1.0),
                show_mesh_grid=kwargs.get("show_grid", False),
            )
            component = PolygonPlotComponent(props, bodies)
        elif field_data.ndim == 1:
            props = LinePlotProps(
                label=field_data.name,
                linewidth=kwargs.get("linewidth", 2.0),
                marker=kwargs.get(
                    "markers", [None] * len(scalar_plot_data.fields)
                )[i],
                marker_size=kwargs.get("marker_size", 6.0),
                alpha=kwargs.get("alpha", 1.0),
            )
            component = LinePlotComponent(props)
        elif field_data.ndim == 2:
            props = QuadPlotProps(
                cmap=next(cmap_cycle),
                color_range=next(crange_cycle),
                log_scale=config.style.log,
                power=kwargs.get("power", 1.0),
                shading=kwargs.get("shading", "auto"),
                alpha=kwargs.get("alpha", 1.0),
                plot_type=projection,
            )
            component = QuadPlotComponent(props, bodies)
        elif field_data.ndim == 3:
            raise ValueError(
                f"Field '{field_data.name}' is 3D. "
                "The plot API requires a slice. "
                "Did you mean to use an analysis tool?"
            )
        else:
            raise ValueError(
                f"Field '{field_data.name}' has unsupported ndim={field_data.ndim}."
            )

        if figure.fig is None:
            raise ValueError("Figure not initialized properly")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, field_data)

    if vector_fields:
        vector_plot_data = create_plot_data(sim_data, vector_fields, config)

        # We can't render FMR levels for vectors. We must squash.
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

        # Always squash vectors to a single pcolormesh grid
        # This works for both composite_view and FMR.
        v1_field = _compose_pcolormesh(vi_levels)
        v2_field = _compose_pcolormesh(vj_levels)

        vector_data_payload = [v1_field, v2_field]

        # Dispatch Vector Component
        vector_type = kwargs.get("vector_type", "quiver")
        if vector_type == "quiver":
            props = QuiverPlotProps(**kwargs)
            component = QuiverPlotComponent(props)
        else:
            props = StreamPlotProps(**kwargs)
            component = StreamPlotComponent(props)

        if figure.fig is None:
            raise ValueError("Figure not initialized properly")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, vector_data_payload)

    figure.render()
    if save_as:
        figure.save(save_as)
    if show:
        plt.show()
    return figure


def animate(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Figure:
    """
    Create a movie from an ordered sequence of checkpoint files.

    This function prepares a Figure (initializes components and their
    properties using the first file), then delegates per-frame updates to
    Figure.animate which will drive the animation and assemble frames into
    a movie file (ffmpeg / matplotlib FFMpegWriter).
    """
    if isinstance(files, str):
        files = [files]

    if not files or len(files) < 2:
        raise ValueError("provide at least two files to create a movie")

    # Use the first file to configure the figure and initialize components
    sim_data = load_data(files[0])
    vector_fields = kwargs.get("vector_fields", None)

    scalar_plot_data = create_plot_data(sim_data, fields, config)
    scalar_fields = scalar_plot_data.fields
    nlvls = 1
    nlvls += sum("_L" in x.name for x in scalar_fields)
    is_fmr = nlvls > 1

    final_scalar_fields_to_render: Sequence[FieldData] = []
    if scalar_fields:
        is_2d = scalar_fields[0].ndim == 2
        if is_2d and is_fmr:
            final_scalar_fields_to_render = [_compose_polygons(scalar_fields)]
        else:
            final_scalar_fields_to_render = scalar_fields

    projection = "cartesian"
    if final_scalar_fields_to_render:
        is_2d_render = final_scalar_fields_to_render[
            0
        ].ndim == 2 or final_scalar_fields_to_render[0].name.endswith(
            "_polygons"
        )
        if is_2d_render and sim_data.metadata.coord_system == "spherical":
            projection = "polar"

    figure = prepare_figure(
        config,
        len(files),
        projection=projection,
        nlvls=nlvls,
        coord_system=CoordSystem(sim_data.metadata.coord_system),
    )

    # Dispatch Scalar Components (initialize only, rendering will be done by Figure.animate)
    cmap_cycle = config.style.cmap
    crange_cycle = config.style.color_range
    bodies = scalar_plot_data.bodies
    for i, field_data in enumerate(final_scalar_fields_to_render):
        component: Component
        if field_data.ndim == 1 and field_data.name.endswith("_polygons"):
            props = PolygonPlotProps(
                cmap=next(cmap_cycle),
                color_range=next(crange_cycle),
                log_scale=config.style.log,
                power=kwargs.get("power", 1.0),
                alpha=kwargs.get("alpha", 1.0),
                show_mesh_grid=kwargs.get("show_grid", False),
            )
            component = PolygonPlotComponent(props, bodies)
        elif field_data.ndim == 1:
            props = LinePlotProps(
                label=field_data.name,
                linewidth=kwargs.get("linewidth", 2.0),
                marker=kwargs.get(
                    "markers", [None] * len(scalar_plot_data.fields)
                )[i],
                marker_size=kwargs.get("marker_size", 6.0),
                alpha=kwargs.get("alpha", 1.0),
            )
            component = LinePlotComponent(props)
        elif field_data.ndim == 2:
            props = QuadPlotProps(
                cmap=next(cmap_cycle),
                color_range=next(crange_cycle),
                log_scale=config.style.log,
                power=kwargs.get("power", 1.0),
                shading=kwargs.get("shading", "auto"),
                alpha=kwargs.get("alpha", 1.0),
                plot_type=projection,
            )
            component = QuadPlotComponent(props, bodies)
        elif field_data.ndim == 3:
            raise ValueError(
                f"Field '{field_data.name}' is 3D. "
                "The plot API requires a slice. "
                "Did you mean to use an analysis tool?"
            )
        else:
            raise ValueError(
                f"Field '{field_data.name}' has unsupported ndim={field_data.ndim}."
            )

        if figure.fig is None:
            raise ValueError("Figure not initialized properly")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, field_data)

    # handle vector fields if present (initialize vector component(s))
    if vector_fields:
        vector_plot_data = create_plot_data(sim_data, vector_fields, config)

        # We can't render FMR levels for vectors. We must squash.
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

        # Always squash vectors to a single pcolormesh grid
        # This works for both composite_view and FMR.
        v1_field = _compose_pcolormesh(vi_levels)
        v2_field = _compose_pcolormesh(vj_levels)

        vector_data_payload = [v1_field, v2_field]

        # Dispatch Vector Component
        vector_type = kwargs.get("vector_type", "quiver")
        if vector_type == "quiver":
            props = QuiverPlotProps(**kwargs)
            component = QuiverPlotComponent(props)
        else:
            props = StreamPlotProps(**kwargs)
            component = StreamPlotComponent(props)

        if figure.fig is None:
            raise ValueError("Figure not initialized properly")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, vector_data_payload)

    # kick off animation: let Figure.animate handle per-frame updates and saving
    out = kwargs.get("save_as") or "animation.mp4"
    use_fps = kwargs.get("fps") or config.animation.frame_rate
    figure.animate(
        files,
        output_path=out,
        fps=use_fps,
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
    **kwargs,
) -> Figure:
    """
    Creates a ND dimensionally-averaged profile analysis plot.
    """
    if isinstance(files, str):
        files = [files]

    # Use the first file for snapshot analysis
    sim_data = load_data(files[0])

    # Call the pipeline to do the 3D FMR stitching and binning
    plot_data = create_coordinate_profile_data(sim_data, fields, config)

    if not plot_data.fields:
        raise ValueError("No coordinate profiles were generated.")

    # Prepare the figure
    figure = prepare_figure(config, len(files), projection="cartesian", nlvls=4)
    # Dispatch components
    for field_data in plot_data.fields:
        props = CoordinateProfileProps(
            label=field_data.name,
            x_scale=kwargs["xscale"],
            y_scale=kwargs["yscale"],
            normalization=kwargs["norm"][0],
            rend=kwargs.get("rend", 1),
            rbeg=kwargs.get("rbeg", 0.2),
            **kwargs,  # Pass through CLI args
        )
        component = CoordinateProfileComponent(props)

        if figure.fig is None:
            raise ValueError("Figure not initialized properly")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, field_data)

    # Render
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
    **kwargs,
) -> Figure:
    if isinstance(files, str):
        raise ValueError("Time series should contain at least 2 files")

    time_series_data = create_time_series_data(files, fields, config)
    if not time_series_data.fields:
        raise ValueError("No time series data were generated.")

    figure = prepare_figure(
        config, nlvls=time_series_data.fields[0].values.ndim * 10
    )

    for field_data in time_series_data.fields:
        props = TimeSeriesPlotProps(
            label=field_data.name,
            linewidth=kwargs.get("linewidth", 2.0),
            marker=kwargs.get("marker", "o"),
            marker_size=kwargs.get("marker_size", 6.0),
            alpha=kwargs.get("alpha", 1.0),
            normalization=kwargs["norm"][0],
        )
        component = TimeSeriesPlotComponent(props)

        if figure.fig is None:
            raise ValueError("Figure not initialized properly")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, field_data)

    # Render
    figure.render()

    if save_as:
        figure.save(save_as)
    if show:
        plt.show()

    return figure
