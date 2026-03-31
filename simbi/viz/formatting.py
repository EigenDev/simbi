# =============================================================================
# formatting.py
#
# figure-level formatting: titles, labels, limits, colorbars, legends, spines.
# delegates to small pure functions; the FigureFormatter class orchestrates.
# =============================================================================
from typing import Any, Optional

from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from simbi.viz.utility import get_field_str

from .config import FigureConfig

LABEL_MAP = {
    "x1": "$x$",
    "x2": "$y$",
    "x3": "$z$",
    "r": "$r$",
    "theta": r"$\theta$",
    "phi": r"$\phi$",
    "radius": "$r$",
    "radius_cylindrical": "$r$",
    "angle": r"$\theta$",
}


def set_title(
    ax: Axes, fig: Figure, config: FigureConfig, time: Optional[float] = None
) -> None:
    """sets the title on the appropriate object (fig or ax)."""
    title = config.title or "Simulation"
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
    # title_str = title

    if "polar" in ax.name:
        fig.suptitle(title_str)
    else:
        ax.set_title(title_str)


def apply_scaling(ax: Axes, config: FigureConfig) -> None:
    """applies log or semilog scaling."""
    ax.set_xscale(config.xscale)
    ax.set_yscale(config.yscale)


def apply_axis_labels(
    ax: Axes,
    config: FigureConfig,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """applies axis labels from config or derived values."""
    if ax.name == "polar":
        return

    if not ax.get_xlabel():
        ax.set_xlabel(config.xlabel or xlabel or "")

    if not ax.get_ylabel():
        ax.set_ylabel(config.ylabel or ylabel or "")


def apply_axis_limits(ax: Axes, config: FigureConfig) -> None:
    """sets axis limits if provided in config."""
    if config.xlims:
        ax.set_xlim(config.xlims.min, config.xlims.max)
    if config.ylims:
        ax.set_ylim(config.ylims.min, config.ylims.max)


def apply_legend(ax: Axes) -> None:
    """adds a legend to the axes."""
    ax.legend(loc="best")


def add_colorbar(
    fig: Figure,
    artist: Any,
    cax: Axes,
    label: Optional[str] = None,
    orientation: str = "vertical",
) -> Colorbar:
    """adds a colorbar to the provided cax."""
    cbar = fig.colorbar(artist, cax=cax, orientation=orientation)
    if label:
        cbar.set_label(get_field_str(label))
    return cbar


def _normalize_entry(entry: Any) -> tuple[dict, dict | None]:
    """normalize a single rendered_artists entry to (artists_dict, metadata)."""
    if entry is None:
        return {}, None

    if isinstance(entry, (list, tuple)) and len(entry) >= 1:
        artists = entry[0] if isinstance(entry[0], dict) else {}
        metadata = entry[1] if len(entry) > 1 else None
        return artists, metadata

    if isinstance(entry, dict):
        return entry, None

    artists = getattr(entry, "artists", None)
    metadata = getattr(entry, "metadata", None)
    if isinstance(artists, dict):
        return artists, metadata

    return {}, None


def _extract_labels(
    normalized: list[tuple[dict, dict | None]],
) -> tuple[Optional[str], list[str], bool]:
    """extract label info from component metadata.

    returns (single_label, all_labels, use_legend).
    """
    single_label: Optional[str] = None
    all_labels: list[str] = []

    for _, metadata in normalized:
        if not metadata or not isinstance(metadata, dict):
            continue
        if "label" in metadata and metadata["label"]:
            single_label = metadata["label"]
            all_labels.append(single_label)
        if "labels" in metadata and metadata["labels"]:
            all_labels += list(metadata["labels"])

    unique = list(set(all_labels))
    use_legend = len(unique) > 1
    return single_label, unique, use_legend


def _find_mappable(normalized: list[tuple[dict, dict | None]]) -> Any:
    """find the first scalar-mappable artist from normalized entries."""
    for artists, _ in normalized:
        if not isinstance(artists, dict):
            continue
        for key in ("mesh", "collection"):
            if key in artists and artists[key] is not None:
                return artists[key]
        for a in artists.values():
            if hasattr(a, "get_array") or hasattr(a, "get_cmap"):
                return a
    return None


def _has_line_artists(normalized: list[tuple[dict, dict | None]]) -> bool:
    """check if any rendered output contains line-like artists."""
    for artists, metadata in normalized:
        if metadata and isinstance(metadata, dict):
            if (
                metadata.get("labels")
                or metadata.get("label")
                or metadata.get("is_line")
            ):
                return True

        if isinstance(artists, dict):
            if "line" in artists and artists["line"] is not None:
                return True
            for a in artists.values():
                if a is not None and "Line2D" in type(a).__name__:
                    return True
    return False


class FigureFormatter:
    """figure-level formatting and layout."""

    def __init__(self, style_config):
        self.style = style_config

    def apply_figure_formatting(
        self,
        fig: Figure,
        main_ax: Axes,
        rendered_artists: list,
        first_data: Any,
        coord_system: Optional[Any] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        show_legend: bool = True,
    ) -> None:
        """apply complete figure-level formatting."""
        time = getattr(first_data, "time", None)
        set_title(main_ax, fig, self.style, time)
        ndim = first_data.values.ndim if first_data else 0

        normalized = [_normalize_entry(e) for e in rendered_artists]

        single_label, label_set, use_legend_from_metadata = _extract_labels(
            normalized
        )

        # derive ylabel from metadata
        derived_ylabel = None
        if len(label_set) == 1:
            derived_ylabel = get_field_str(label_set[0])
            show_legend = False
        elif single_label and not use_legend_from_metadata:
            derived_ylabel = get_field_str(single_label)

        # derive axis labels from data
        if xlabel is None or ylabel is None:
            axis_names = getattr(first_data, "axis_names", None)
            if axis_names and isinstance(axis_names, (list, tuple)):
                if xlabel is None and len(axis_names) >= 1:
                    xlabel = LABEL_MAP.get(axis_names[0], axis_names[0])
                if ylabel is None and len(axis_names) >= 2:
                    ylabel = LABEL_MAP.get(axis_names[1], axis_names[1])

            if ylabel is None and derived_ylabel:
                ylabel = derived_ylabel

            if (
                ylabel is None
                and hasattr(first_data, "name")
                and len(label_set) == 1
            ):
                ylabel = get_field_str(first_data.name)

        apply_axis_labels(main_ax, self.style, xlabel, ylabel)
        apply_axis_limits(main_ax, self.style)

        # colorbar
        mappable = _find_mappable(normalized)
        if mappable is not None:
            self._format_colorbar(fig, main_ax, mappable, first_data)

        # legend
        has_lines = _has_line_artists(normalized)
        should_show_legend = use_legend_from_metadata or (
            show_legend and has_lines
        )
        if should_show_legend and len(label_set) > 1:
            apply_legend(main_ax)

        # spines for 1d plots only
        if main_ax.name != "polar" and ndim == 1:
            remove_spines(main_ax)

        # polar formatting
        if main_ax.name == "polar":
            main_ax.grid(False)
            main_ax.set_theta_zero_location("N")
            main_ax.set_theta_direction(-1)
            main_ax.set_xticklabels([])
            main_ax.set_yticklabels([])

    def _format_colorbar(
        self, fig: Figure, ax: Axes, artist: Any, field_data: Any
    ) -> None:
        """place or update a colorbar for the axes."""
        from matplotlib.colorbar import Colorbar as MplColorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        label = getattr(field_data, "name", None)
        if label and "_polygons" in label:
            label = label.split("_polygons")[0]
        label = get_field_str(label) if label else None

        # update existing colorbar in-place if possible
        existing_cbar = getattr(ax, "_simbi_colorbar", None)
        if existing_cbar is not None and isinstance(existing_cbar, MplColorbar):
            existing_cbar.mappable = artist
            existing_cbar.update_normal(artist)
            if label:
                existing_cbar.set_label(label)
            return

        # create new colorbar
        cax = None
        orientation = "vertical"

        if hasattr(ax, "name") and "polar" in getattr(ax, "name", ""):
            polar_pos = ax.get_position()
            theta = getattr(field_data, "domain", [None, None])
            half_sphere = False
            if len(theta) > 1 and theta[1] is not None:
                import math

                half_sphere = theta[1][-1] == 0.5 * math.pi

            if half_sphere:
                width = min(0.6, 0.78)
                x = polar_pos.x0 + (polar_pos.width - width) / 2 - 0.01
                cax = fig.add_axes((x, 0.2, width, 0.03))
                orientation = "horizontal"
            else:
                height = 0.8
                x = polar_pos.x0 + polar_pos.width + 0.05
                y = polar_pos.y0 + (polar_pos.height - height) / 2
                cax = fig.add_axes((x, y, 0.03, height))
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig.colorbar(artist, cax=cax, orientation=orientation)
        if label:
            cbar.set_label(label)
        setattr(ax, "_simbi_colorbar", cbar)


def remove_spines(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
