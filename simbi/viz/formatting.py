from typing import Any, Optional

from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from simbi.viz.utility import get_field_str

from .config import StyleConfig


def set_title(
    ax: Axes, fig: Figure, config: StyleConfig, time: Optional[float] = None
) -> None:
    """Sets the title on the appropriate object (fig or ax)."""
    title = config.setup
    time_units = config.time_units
    title_time = time
    time_scale = config.time_scale
    if time_scale and title_time is not None:
        title_time /= time_scale

    title_str = (
        f"{title}, t={title_time:.2f} {time_units}"
        if title_time is not None
        else f"{title}"
    )

    if "polar" in ax.name:
        fig.suptitle(title_str)
    else:
        ax.set_title(title_str)


def apply_scaling(ax: Axes, config: StyleConfig) -> None:
    """Applies log or semilog scaling."""
    ax.set_xscale(config.xscale)
    ax.set_yscale(config.yscale)
    # Note: 'log' is handled by the component's norm


def apply_axis_labels(
    ax: Axes,
    config: StyleConfig,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Applies axis labels from config or derived values."""
    if ax.name == "polar":
        return

    # Use explicit config label if provided
    ax.set_xlabel(config.x_label or xlabel or "$x$")
    ax.set_ylabel(config.y_label or ylabel or "$y$")


def apply_axis_limits(ax: Axes, config: StyleConfig) -> None:
    """Sets axis limits if provided in config."""
    if config.xlims:
        ax.set_xlim(config.xlims.min, config.xlims.max)
    if config.ylims:
        ax.set_ylim(config.ylims.min, config.ylims.max)


def apply_legend(ax: Axes, config: StyleConfig) -> None:
    """Adds a legend if configured."""
    if config.legend:
        ax.legend(loc=config.legend_loc)


def add_colorbar(
    fig: Figure,
    artist: Any,
    cax: Axes,  # The colorbar axes MUST be provided
    label: Optional[str] = None,
    orientation: str = "vertical",
) -> Colorbar:
    """
    Adds a colorbar to the *provided* cax.
    This is now a "simple" formatter.
    """
    cbar = fig.colorbar(artist, cax=cax, orientation=orientation)
    if label:
        cbar.set_label(get_field_str(label))
    return cbar


def remove_spines(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
