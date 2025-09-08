"""Public API for the visualization system.

This module provides the main entry points for creating different types of
visualizations in simbi.
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt

from .components.histogram import HistogramPlotComponent, HistogramPlotProps
from .components.line import LinePlotComponent, LinePlotProps
from .components.multidim import MultidimPlotComponent
from .components.temporal import TemporalPlotComponent, TemporalPlotProps
from .core.config import (
    VisualizationConfig,
)
from .core.conversion import multidim_props_from_args
from .core.figure import Figure
from .pipeline.transforms import (
    create_plot_data,
    create_time_series_data,
    load_data,
    prepare_figure,
)


def plot_line(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Figure:
    """
    Create a line plot visualization.

    Args:
        files: File path or sequence of file paths to visualize
        fields: Sequence of field names to visualize
        save_as: Optional file path to save the visualization
        show: Whether to display the visualization
        **kwargs: Additional visualization options

    Returns:
        Figure object
    """
    # Handle single file case
    if isinstance(files, str):
        files = [files]

    figure = prepare_figure(config, len(files))
    sim_data = load_data(files[0])
    plot_data = create_plot_data(sim_data, fields, config)

    # Add line component for each field
    for i, field in enumerate(fields):
        props = LinePlotProps(
            field_indices=[i],
            labels=kwargs.get("labels", [field]),
            linewidth=kwargs.get("linewidth", 2.0),
            markers=kwargs.get("markers", []),
            show_legend=kwargs.get("legend", None),
            marker_size=kwargs.get("marker_size", 6.0),
            alpha=kwargs.get("alpha", 1.0),
        )

        component = LinePlotComponent(props)
        if figure.fig is None:
            raise ValueError("Figure not initialized properly")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component)

    figure.load_data(plot_data)
    figure.render()

    if save_as:
        figure.save(save_as)

    if show:
        plt.show()

    return figure


def plot_histogram(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["gamma_beta"],
    save_as: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Figure:
    """
    Create a histogram plot visualization.

    Args:
        files: File path or sequence of file paths to visualize
        fields: Sequence of field names to visualize
        save_as: Optional file path to save the visualization
        show: Whether to display the visualization
        **kwargs: Additional visualization options

    Returns:
        Figure object
    """
    if isinstance(files, str):
        files = [files]

    figure = prepare_figure(config, len(files))
    sim_data = load_data(files[0])
    plot_data = create_plot_data(sim_data, fields, config)

    props = HistogramPlotProps(
        field_index=0,
        nbins=kwargs.get("nbins", 50),
        log_bins=kwargs.get("log_bins", True),
        range=kwargs.get("range"),
        density=kwargs.get("density", False),
        histtype=kwargs.get("histtype", "bar"),
        cumulative=kwargs.get("cumulative", False),
        color=kwargs.get("color"),
        edgecolor=kwargs.get("edgecolor", "black"),
        alpha=kwargs.get("alpha", 0.7),
        linewidth=kwargs.get("linewidth", 1.0),
        label=kwargs.get("label"),
        fit_power_law=kwargs.get("fit_power_law", False),
        power_law_range=kwargs.get("power_law_range"),
    )
    if figure.fig is None:
        raise ValueError("Figure not initialized properly")

    component = HistogramPlotComponent(props)
    component.initialize(figure.fig, figure.axes["main"])
    figure.add_component(component)

    figure.load_data(plot_data)
    figure.render()

    if save_as:
        figure.save(save_as)

    if show:
        plt.show()

    return figure


def plot_multidim(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Figure:
    """
    Create a multidimensional plot visualization.

    Args:
        files: File path or sequence of file paths to visualize
        fields: Sequence of field names to visualize
        save_as: Optional file path to save the visualization
        show: Whether to display the visualization
        **kwargs: Additional visualization options

    Returns:
        Figure object
    """
    if isinstance(files, str):
        files = [files]

    figure = prepare_figure(config, len(files))
    sim_data = load_data(files[0])
    plot_data = create_plot_data(sim_data, fields, config)

    # Add multidim component for each field
    for i, field in enumerate(fields):
        props = multidim_props_from_args(
            kwargs, i, next(config.style.cmap), next(config.style.color_range)
        )
        if figure.fig is None:
            raise ValueError("Figure not initialized properly")

        component = MultidimPlotComponent(props)
        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component)

    figure.load_data(plot_data)
    figure.render()

    if save_as:
        figure.save(save_as)

    if show:
        plt.show()

    return figure


def plot_temporal(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Figure:
    """
    Create a temporal plot visualization.

    Args:
        files: Sequence of file paths to visualize (chronological order)
        fields: Sequence of field names to visualize
        save_as: Optional file path to save the visualization
        show: Whether to display the visualization
        **kwargs: Additional visualization options

    Returns:
        Figure object
    """
    if len(files) < 2:
        raise ValueError("Temporal plots require at least two files")

    figure = prepare_figure(config, len(files))
    time_series = create_time_series_data(files, fields)

    for i, field in enumerate(fields):
        props = TemporalPlotProps(
            field_index=i,
            color=kwargs.get("color"),
            linewidth=kwargs.get("linewidth", 2.0),
            marker=kwargs.get("marker"),
            marker_size=kwargs.get("marker_size", 6.0),
            alpha=kwargs.get("alpha", 1.0),
            label=kwargs.get("label") or field,
            show_moving_average=kwargs.get("show_moving_average", False),
            moving_average_window=kwargs.get("moving_average_window", 5),
            show_trend=kwargs.get("show_trend", False),
            trend_degree=kwargs.get("trend_degree", 1),
        )

        component = TemporalPlotComponent(props=props)
        if figure.fig is None:
            raise ValueError("Figure not initialized properly")

        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component)

    figure.load_data(time_series)
    figure.render()
    figure.tight_layout()
    if save_as:
        figure.save(save_as)

    if show:
        plt.show()

    return figure


def animate(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ["rho"],
    plot_type: str = "multidim",
    save_as: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Figure:
    """
    Create an animation from a sequence of files.

    Args:
        files: Sequence of file paths to animate
        fields: Sequence of field names to visualize
        plot_type: Type of plot to animate ("line", "multidim", "histogram")
        save_as: Optional file path to save the animation
        show: Whether to display the animation
        **kwargs: Additional visualization options

    Returns:
        Figure object with animation
    """
    if plot_type == "line":
        figure = plot_line(config, files[0], fields, None, False, **kwargs)
    elif plot_type == "multidim":
        figure = plot_multidim(config, files[0], fields, None, False, **kwargs)
    elif plot_type == "histogram":
        figure = plot_histogram(config, files[0], fields, None, False, **kwargs)
    elif plot_type == "temporal":
        # Temporal doesn't make sense to animate
        raise ValueError("Temporal plots cannot be animated")
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    frame_rate = kwargs.get("frame_rate", 30)
    figure.animate(files, interval=int(1000 / frame_rate))
    figure.tight_layout()

    if save_as:
        figure.save(save_as)

    if show:
        figure.show()

    return figure
