from typing import Sequence, Optional, Any
import numpy as np

from simbi.tools.visualization.constants.alias import FIELD_ALIASES

from .figure import Figure
from .components.line_plot import LinePlotComponent
from .components.multidim_plot import MultidimPlotComponent
from .components.histogram_plot import HistogramComponent
from .components.temporal_plot import TemporalPlotComponent
from .components.title import TitleComponent
from .config.builder import ConfigBuilder
from .bridge import SimbiDataBridge


def plot_line(
    files: Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    theme: str = "default",
    config: Optional[dict[str, Any]] = None,
    **kwargs,
) -> Figure:
    """Create a line plot visualization with dimension auto-detection"""

    if config is None:
        config = ConfigBuilder.create_config("line", files, fields, **kwargs)
    else:
        # If config directly provided, merge with kwargs
        kwargs_config = ConfigBuilder.from_args(kwargs)
        for section, values in kwargs_config.items():
            if section not in config:
                config[section] = {}
            config[section].update(values)

    # Create figure
    fig = Figure(config, theme=theme)

    # Add title component
    fig.add(TitleComponent, "title", setup=config["plot"]["setup"], ax_title=True)

    # Add line components for each field
    for i, field in enumerate(fields):
        # Get label for this field
        label = (
            kwargs.get("labels", [field])[i]
            if i < len(kwargs.get("labels", []))
            else field
        )

        # Get color for this field
        colors = kwargs.get("colors", [f"C{i}" for i in range(len(fields))])
        color = colors[i % len(colors)]

        fig.add(
            LinePlotComponent,
            f"line_{i}",
            field=field,
            label=label,
            color=color,
            linewidth=kwargs.get("linewidth", 2),
        )

    # Load first file and render
    fig.load_data(files[0])
    fig.render()

    # Save if requested
    if save_as:
        fig.save(save_as)

    # Show if requested
    if show:
        fig.show()

    return fig


def plot_multidim(
    files: Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    theme: str = "default",
    config: Optional[dict[str, Any]] = None,
    **kwargs,
) -> Figure:
    """Create a multidimensional plot visualization with dimension auto-detection"""
    if config is None:
        config = ConfigBuilder.create_config("multidim", files, fields, **kwargs)
    else:
        # If config directly provided, merge with kwargs
        kwargs_config = ConfigBuilder.from_args(kwargs)
        for section, values in kwargs_config.items():
            if section not in config:
                config[section] = {}
            config[section].update(values)

    # Check for polar coordinates
    is_cartesian = kwargs.get("is_cartesian", True)

    # Create figure
    fig = Figure(config, theme=theme)

    # Add title component
    fig.add(
        TitleComponent,
        "title",
        setup=config["plot"]["setup"],
        ax_title=is_cartesian,
        fig_title=not is_cartesian,
    )
    # Add multidim components for each field
    for i, field in enumerate(fields):
        # Get color range for this field (if provided)
        color_range = next(config["style"]["color_range"])

        # Get colormap for this field
        cmaps = config["style"]["cmap"]
        cmap = next(cmaps)

        fig.add(
            MultidimPlotComponent,
            f"multidim_{i}",
            field=field,
            cmap=cmap,
            color_range=color_range,
        )

    # Load first file and render
    fig.load_data(files[0])
    fig.render()

    # Save if requested
    if save_as:
        fig.save(save_as)

    # Show if requested
    if show:
        fig.show()

    return fig


def plot_histogram(
    files: Sequence[str],
    fields: Sequence[str] = ["gamma_beta"],
    save_as: Optional[str] = None,
    show: bool = True,
    theme: str = "default",
    config: Optional[dict[str, Any]] = None,
    **kwargs,
) -> Figure:
    """Create a histogram plot visualization with dimension auto-detection"""
    if config is None:
        config = ConfigBuilder.create_config("histogram", files, fields, **kwargs)
    else:
        # If config directly provided, merge with kwargs
        kwargs_config = ConfigBuilder.from_args(kwargs)
        for section, values in kwargs_config.items():
            if section not in config:
                config[section] = {}
            config[section].update(values)

    # Create figure
    fig = Figure(config, theme=theme)

    # Add title component
    fig.add(TitleComponent, "title", setup=config["plot"]["setup"], ax_title=True)

    # Add histogram component
    fig.add(
        HistogramComponent,
        "histogram",
        field=fields[0],  # Histograms typically only use one field
        label=kwargs.get("label", fields[0]),
        color=kwargs.get("color", "blue"),
    )

    # Load first file and render
    fig.load_data(files[0])
    fig.render()

    # Save if requested
    if save_as:
        fig.save(save_as)

    # Show if requested
    if show:
        fig.show()

    return fig


def plot_temporal(
    files: Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    theme: str = "default",
    config: Optional[dict[str, Any]] = None,
    **kwargs,
) -> Figure:
    """Create a temporal plot visualization"""
    # For temporal plots, we need multiple files
    if len(files) < 2 and not kwargs.get("single_file_mode", False):
        print("Warning: Temporal plots typically need multiple files.")

    if config is None:
        config = ConfigBuilder.create_config("temporal", files, fields, **kwargs)
    else:
        # If config directly provided, merge with kwargs
        kwargs_config = ConfigBuilder.from_args(kwargs)
        for section, values in kwargs_config.items():
            if section not in config:
                config[section] = {}
            config[section].update(values)

    # Create figure
    fig = Figure(config, theme=theme)

    # Add title component
    fig.add(TitleComponent, "title", setup=config["plot"]["setup"], ax_title=True)
    if any(field in FIELD_ALIASES for field in fields):
        # If field is an alias, replace it with the actual field name
        fields = [FIELD_ALIASES.get(field, field) for field in fields]

    # Determine if dealing with accretion data
    is_accretion_data = any(
        field in ["accreted_mass", "accretion_rate"] for field in fields
    )

    # Process all data files to build time series
    times = []
    field_values = {field: [] for field in fields}
    body_data = {}  # For accretion data

    bridge = SimbiDataBridge(fig.state)

    # Collect data from all files
    for file_path in files:
        # Load data
        data = bridge.load_file(file_path)
        time = data.setup.get("time", 0.0)
        times.append(time)

        # Extract field values
        for field in fields:
            if is_accretion_data:
                # Handle accretion data
                if not data.immersed_bodies:
                    continue

                for body_id, body in data.immersed_bodies.items():
                    if field not in body:
                        continue

                    if body_id not in body_data:
                        body_data[body_id] = {field: []}
                    elif field not in body_data[body_id]:
                        body_data[body_id][field] = []

                    body_data[body_id][field].append(body[field])
            else:
                # Handle regular field data
                var = None
                # Calculate field value (may need weighted average)
                if kwargs.get("weight"):
                    # Compute weighted average
                    weight_field = kwargs["weight"]
                    weights = data.fields.get(weight_field)
                    if weights is not None:
                        # Calculate cell volumes
                        from simbi.functional.helpers import calc_cell_volume

                        coords = [
                            data.mesh.get(f"x{i + 1}v")
                            for i in range(data.setup.get("dimensions", 1))
                        ]
                        dV = calc_cell_volume(
                            coords,
                            data.setup.get("coord_system", "cartesian"),
                            vertices=True,
                        )

                        # Compute weighted mean
                        if field in data.fields:
                            var_data = data.fields[field]
                            var = np.sum(weights * var_data * dV) / np.sum(weights * dV)

                # If no weighted calculation, use max value
                if var is None and field in data.fields:
                    var = np.max(data.fields[field])

                if var is not None:
                    field_values[field].append(var)

    # Load first file just to set up the state
    fig.load_data(files[0])

    # Add component for each field or body
    if is_accretion_data:
        # For accretion data, create a temporal plot for each body
        field = fields[0]
        body_id = kwargs.get("body_id")

        if body_id and body_id in body_data:
            # Plot specific body
            fig.add(
                TemporalPlotComponent,
                "temporal_0",
                field=field,
                label=kwargs.get("label", f"Body {body_id}"),
                color=kwargs.get("color", "blue"),
                body_id=body_id,
                times=times,
                values=body_data[body_id][field],
                auto_scale=True,
            )
        else:
            # Plot all bodies
            for i, (bid, data) in enumerate(body_data.items()):
                if field not in data:
                    continue

                fig.add(
                    TemporalPlotComponent,
                    f"temporal_{i}",
                    field=field,
                    label=rf"$ M_{i + 1} $",
                    color=f"C{i}",
                    body_id=bid,
                    times=times,
                    values=data[field],
                    auto_scale=True,
                )
    else:
        # For regular fields, create a temporal plot for each field
        for i, field in enumerate(fields):
            if not field_values[field]:
                continue

            label = (
                kwargs.get("labels", [field])[i]
                if i < len(kwargs.get("labels", []))
                else field
            )

            fig.add(
                TemporalPlotComponent,
                f"temporal_{i}",
                field=field,
                label=label,
                color=f"C{i}",
                times=times,
                values=field_values[field],
                auto_scale=True,
            )

    # Render the figure
    fig.render()

    # Save if requested
    if save_as:
        fig.save(save_as)

    # Show if requested
    if show:
        fig.show()

    return fig


def animate(
    files: Sequence[str],
    plot_type: str = "line",
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    frame_rate: int = 30,
    theme: str = "default",
    **kwargs,
) -> Figure:
    """
    Unified animation interface for all plot types

    Args:
        files: Sequence of data files to animate
        plot_type: Type of plot ("line", "multidim", "histogram", "temporal")
        fields: Sequence of fields to visualize
        save_as: Output file path
        show: Whether to display the animation
        frame_rate: Animation frame rate
        **kwargs: Additional arguments specific to the plot type

    Returns:
        Figure object
    """
    # Create appropriate plot based on type
    if plot_type == "line":
        fig = plot_line(files[:1], fields, None, False, theme=theme, **kwargs)
    elif plot_type == "multidim":
        fig = plot_multidim(files[:1], fields, None, False, theme=theme, **kwargs)
    elif plot_type == "histogram":
        fig = plot_histogram(files[:1], fields, None, False, theme=theme, **kwargs)
    elif plot_type == "temporal":
        fig = plot_temporal(files[:1], fields, None, False, theme=theme, **kwargs)
    else:
        raise ValueError(
            f"Unknown plot type: {plot_type}. Must be one of: line, multidim, histogram, temporal"
        )

    # Create animation
    fig.animate(files, interval=int(1000 / frame_rate))

    fig.tight_layout()
    # Save if requested
    if save_as:
        fig.save(save_as)

    # Show if requested
    if show:
        fig.show()

    return fig
