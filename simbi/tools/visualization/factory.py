from typing import Any, Sequence
import matplotlib.pyplot as plt
from .figure import Figure
from .components.line_plot import LinePlotComponent
from .components.multidim_plot import MultidimPlotComponent
from .components.title import TitleComponent


def create_line_plot(config: dict[str, Any], files: Sequence[str]) -> Figure:
    """Create a line plot visualization"""
    fig = Figure(config)

    # Add title component
    fig.add(
        TitleComponent,
        "title",
        setup=config.get("plot", {}).get("setup", "Simulation"),
        ax_title=True,
    )

    # Add line components for each field
    for i, field in enumerate(config.get("plot", {}).get("fields", ["rho"])):
        fig.add(
            LinePlotComponent,
            f"line_{i}",
            field=field,
            label=field,
            color=f"C{i}",
            linewidth=2,
        )

    # Load first file for initial render
    if files:
        fig.load_data(files[0])

    return fig


def create_multidim_plot(config: dict[str, Any], files: Sequence[str]) -> Figure:
    """Create a multidimensional plot visualization"""
    fig = Figure(config)

    # Add title component
    fig.add(
        TitleComponent,
        "title",
        setup=config.get("plot", {}).get("setup", "Simulation"),
        ax_title=config.get("plot", {}).get("is_cartesian", True),
        fig_title=not config.get("plot", {}).get("is_cartesian", True),
    )

    # Add multidim components for each field
    fields = config.get("plot", {}).get("fields", ["rho"])
    for i, field in enumerate(fields):
        fig.add(
            MultidimPlotComponent,
            f"multidim_{i}",
            field=field,
            cmap=config.get("style", {}).get("color_maps", ["viridis"])[
                i % len(config.get("style", {}).get("color_maps", ["viridis"]))
            ],
            log=config.get("style", {}).get("log", False),
            power=config.get("style", {}).get("power", 1.0),
            show_colorbar=config.get("style", {}).get("show_colorbar", True),
            draw_immersed_bodies=config.get("style", {}).get(
                "draw_immersed_bodies", False
            ),
        )

    # Load first file for initial render
    if files:
        fig.load_data(files[0])

    return fig


def create_animation(fig: Figure, files: Sequence[str], frame_rate: int = 30) -> None:
    """Create animation from a figure and files"""
    # Setup animation
    fig.animate(files, interval=1000 / frame_rate)
