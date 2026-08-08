import math
from typing import Any, Optional

import numpy as np
from matplotlib.axes import Axes
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
    """Sets the title on the appropriate object (fig or ax)."""
    title = config.title or "Simulation"
    time_units = config.time_units
    title_time = time if config.show_time else None
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


def apply_scaling(ax: Axes, config: FigureConfig) -> None:
    """Applies log or semilog scaling."""
    ax.set_xscale(config.xscale)
    ax.set_yscale(config.yscale)
    # note: 'log' is handled by the component's norm


def apply_axis_labels(
    ax: Axes,
    config: FigureConfig,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Applies axis labels from config or derived values."""
    if ax.name == "polar":
        return

    # use explicit config label if provided
    if not ax.get_xlabel():
        ax.set_xlabel(config.xlabel or xlabel or "")

    if not ax.get_ylabel():
        ax.set_ylabel(config.ylabel or ylabel or "")


def apply_axis_limits(ax: Axes, config: FigureConfig) -> None:
    """Sets axis limits if provided in config."""

    if config.xlims:
        ax.set_xlim(config.xlims.min, config.xlims.max)
    if config.ylims:
        ax.set_ylim(config.ylims.min, config.ylims.max)


def apply_legend(ax: Axes) -> None:
    """Adds a legend to the axes."""
    ax.legend(loc="best")


# a formatting or frame failure is otherwise swallowed silently, and an
# unlabeled plot or a frozen movie exits 0 looking finished. each distinct
# failure warns once, carrying the real exception, so the defect is visible
# without spamming once per frame.
_WARNED: set[str] = set()


def warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        import warnings

        warnings.warn(msg, stacklevel=3)


def find_mappables(normalized: Any) -> list[tuple[Any, Optional[str]]]:
    """the color-mapped artists among normalized (artists, metadata) pairs,
    each with the field name it draws, in the order they were rendered.

    one entry is one quantity wanting a scale of its own. a field render
    publishes its artist under 'mesh' or 'collection'; a vector overlay is
    colormapped too but reads off the field beneath it, so it is passed over
    unless nothing else is drawn."""
    entries: list[tuple[Any, Optional[str]]] = []

    for artists, metadata in normalized:
        if not isinstance(artists, dict):
            continue

        label = None
        if isinstance(metadata, dict):
            label = metadata.get("colorbar_label")

        for key in ("mesh", "collection"):
            if artists.get(key) is not None:
                entries.append((artists[key], label))
                break

    if entries:
        return entries

    for artists, _ in normalized:
        if not isinstance(artists, dict):
            continue
        for artist in artists.values():
            try:
                if hasattr(artist, "get_array") or hasattr(artist, "get_cmap"):
                    return [(artist, None)]
            except Exception:
                continue

    return []


class FigureFormatter:
    """
    encapsulates all figure-level formatting and layout responsibilities.

    responsibilities:
      - set titles and time labels
      - apply axis labels and limits from style or data
      - create and place colorbars (cartesian / polar heuristics)
      - apply legend and spine policies

    the Figure object should construct one formatter (or use this default)
    and delegate all layout work to it. this removes formatting logic from
    the Figure orchestration code and restores single-responsibility.
    """

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
        """
        Apply complete figure-level formatting.

        This function is flexible in the form of `rendered_artists` it accepts:
          - list of dicts (legacy)
          - list of (artists_dict, metadata_dict) tuples
          - list of RenderResult-like objects (having .artists and .metadata)
        The formatter will try to extract both the artists mapping and any
        metadata provided by components. Metadata is used to decide whether
        to show a legend (only line-like artists) and for other display hints.

        Label derivation from metadata:
          - "label" (str): single label -> use as ylabel
          - "labels" (list): if len == 1 -> ylabel, no legend
                             if len > 1 -> legend, no ylabel

        Args:
            fig: matplotlib Figure
            main_ax: the main Axes to format
            rendered_artists: list of artist outputs returned by components.
                              entries may be dict, tuple(artists, metadata), or objects.
            first_data: the primary FieldData-like object for context
            coord_system: optional coord system meta (not required)
            xlabel, ylabel: optional explicit axis labels to prefer
            show_legend: whether to allow showing a legend (subject to presence
                         of line-like artists)
        """
        # title and time
        time = getattr(first_data, "time", None)
        set_title(main_ax, fig, self.style, time)
        ndim = 0
        if first_data:
            ndim = first_data.values.ndim

        # normalize rendered_artists entries first (needed for label extraction)
        normalized: list[tuple[dict, dict | None]] = []
        for entry in rendered_artists:
            if entry is None:
                normalized.append(({}, None))
                continue
            # tuple/list of (artists, metadata)
            if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                artists = entry[0] if len(entry) > 0 else {}
                metadata = entry[1] if len(entry) > 1 else None
                normalized.append(
                    (artists if isinstance(artists, dict) else {}, metadata)
                )
                continue
            # dict -> legacy artists-only return
            if isinstance(entry, dict):
                normalized.append((entry, None))
                continue
            # object with .artists/.metadata attributes (RenderResult-like)
            artists = getattr(entry, "artists", None)
            metadata = getattr(entry, "metadata", None)
            if isinstance(artists, dict):
                normalized.append((artists, metadata))
            else:
                try:
                    cast_map = dict(entry)
                    normalized.append((cast_map, None))
                except Exception:
                    normalized.append(({}, None))

        # extract labels from component metadata for smart ylabel/legend handling
        # "label" (str) -> single ylabel
        # "labels" (list) -> if 1 item: ylabel, no legend; if >1: legend, no ylabel
        metadata_label: Optional[str] = None
        metadata_labels: list = []
        use_legend_from_metadata = False

        for _, metadata in normalized:
            if metadata and isinstance(metadata, dict):
                if "label" in metadata and metadata["label"]:
                    metadata_label = metadata["label"]
                    metadata_labels.append(metadata_label)
                if "labels" in metadata and metadata["labels"]:
                    metadata_labels += list(metadata["labels"])
                    if len(metadata_labels) > 1:
                        use_legend_from_metadata = True

        # determine ylabel from metadata if not explicitly provided
        derived_ylabel = None
        label_set = list(set(metadata_labels))
        if len(label_set) == 1:
            # single label in list -> use as ylabel, suppress legend
            derived_ylabel = get_field_str(label_set[0])
            show_legend = False
        elif len(label_set) > 1:
            # multiple labels -> show legend, no ylabel from metadata
            use_legend_from_metadata = True
            derived_ylabel = None
        elif metadata_label:
            # single "label" key -> use as ylabel
            derived_ylabel = get_field_str(metadata_label)

        # determine axis labels
        if xlabel is None or ylabel is None:
            try:
                # with no derived label a 1D plot shows a legend and leaves the
                # y-axis unlabeled
                # prefer explicit axis names from data when present
                axis_names = getattr(first_data, "axis_names", None)
                if axis_names and isinstance(axis_names, (list, tuple)):
                    if xlabel is None and len(axis_names) >= 1:
                        xlabel = LABEL_MAP.get(axis_names[0], axis_names[0])
                    if ylabel is None and len(axis_names) >= 2:
                        ylabel = LABEL_MAP.get(axis_names[1], axis_names[1])

                # use derived ylabel from metadata
                if ylabel is None and derived_ylabel:
                    ylabel = derived_ylabel

                # fallback to field name for y-axis for 1D/line-like data
                if (
                    ylabel is None
                    and hasattr(first_data, "name")
                    and len(label_set) == 1
                ):
                    ylabel = get_field_str(first_data.name)

            except Exception:
                pass

        # axis labels & limits
        try:
            apply_axis_labels(main_ax, self.style, xlabel, ylabel)
        except Exception:
            pass

        try:
            apply_axis_limits(main_ax, self.style)
        except Exception:
            pass

        # the chart is oriented before anything is placed around it: where a
        # wedge sits in the axes box, and so where there is room for a colorbar,
        # follows from the zero direction and the sense of increasing angle
        if main_ax.name == "polar":
            main_ax.grid(False)
            main_ax.set_theta_zero_location("N")
            main_ax.set_theta_direction(-1)

            main_ax.set_xticklabels([])
            main_ax.set_yticklabels([])

        try:
            self._format_colorbars(
                fig, main_ax, find_mappables(normalized), first_data
            )
        except Exception as exc:
            # a plot missing its colorbar is still a plot, so this does not halt
            # the render -- but it is a defect, and swallowing it silently is how
            # a chart ships with no scale at all
            warn_once(
                f"colorbar:{type(exc).__name__}", f"colorbar failed: {exc}"
            )

        # legend: only show if there are line-like artists (or metadata indicates labels)
        has_line_like = False
        for artists, metadata in normalized:
            # check explicit metadata hint first
            if metadata and isinstance(metadata, dict):
                if (
                    metadata.get("labels")
                    or metadata.get("label")
                    or metadata.get("is_line")
                    or metadata.get("is_vector") is False
                ):
                    has_line_like = True
                    break

            # check artist keys and types
            if isinstance(artists, dict):
                if "line" in artists and artists["line"] is not None:
                    has_line_like = True
                    break
                for a in artists.values():
                    try:
                        cls_name = a.__class__.__name__
                        if (
                            cls_name.endswith("Line2D")
                            or "Line2D" in cls_name
                            or "Line" in cls_name
                        ):
                            has_line_like = True
                            break
                    except Exception:
                        continue
            if has_line_like:
                break

        try:
            # apply legend when:
            # - explicitly requested via metadata (multiple labels)
            # - or has line-like artists and show_legend is True
            should_show_legend = use_legend_from_metadata or (
                show_legend and has_line_like
            )

            # but not when the single label was used as the ylabel
            if should_show_legend and len(label_set) > 1:
                apply_legend(main_ax)
        except Exception:
            pass

        # remove top/right spines only for 1D plots
        if main_ax.name != "polar" and ndim == 1:
            try:
                remove_spines(main_ax)
            except Exception:
                pass




    def refresh_colorbars(
        self, fig: Figure, ax: Axes, normalized: Any, field_data: Any
    ) -> None:
        """re-point the colorbars at the artists of a freshly drawn frame.

        a quadmesh whose vertices moved between checkpoints is rebuilt rather
        than refilled, so a colorbar is left addressing a discarded artist and
        stops tracking the data range."""
        self._format_colorbars(fig, ax, find_mappables(normalized), field_data)

    def _format_colorbars(
        self,
        fig: Figure,
        ax: Axes,
        entries: list[tuple[Any, Optional[str]]],
        field_data: Any,
    ) -> None:
        """Place or update one colorbar per drawn quantity.

        each bar is kept against its slot on the axes and updated in place, so
        an animation neither accumulates colorbar axes nor shifts its layout
        between frames.
        """
        bars = getattr(ax, "_simbi_colorbars", None)
        if not isinstance(bars, dict):
            bars = {}
            setattr(ax, "_simbi_colorbars", bars)

        for slot, (artist, name) in enumerate(entries):
            label = _colorbar_label(name, field_data)
            existing = bars.get(slot)

            if existing is not None:
                existing.mappable = artist
                existing.update_normal(artist)
                if label:
                    existing.set_label(label)
                continue

            box = under_wedge_box(ax, slot, len(entries))
            if box is not None:
                # an inset rides with its parent, so a bar laid in the chart's
                # own empty strip keeps its place when the layout reflows
                bar = fig.colorbar(
                    artist, cax=ax.inset_axes(box), orientation="horizontal"
                )
            else:
                # letting the colorbar take its space from the parent axes keeps
                # the layout engine in charge of the margins, so a left-hand
                # bar's ticks and label land inside the canvas
                bar = fig.colorbar(
                    artist,
                    ax=ax,
                    location=colorbar_side(slot, len(entries)),
                    fraction=0.046,
                    pad=0.04,
                )

            if label:
                bar.set_label(label)
            bars[slot] = bar


def _colorbar_label(name: Optional[str], field_data: Any) -> Optional[str]:
    """the axis label for the bar describing `name`, in display notation."""
    label = name or getattr(field_data, "name", None)
    if label and "_polygons" in label:
        label = label.split("_polygons")[0]
    return get_field_str(label) if label else None


def colorbar_side(slot: int, total: int) -> str:
    """which side of the chart the bar for panel `slot` belongs on.

    panels alternate sides, so a mirrored pair carries each half's scale beside
    that half; a single quantity keeps the conventional right-hand bar."""
    if total < 2:
        return "right"
    return "right" if slot % 2 == 0 else "left"


# geometry of a colorbar laid in the strip a hemispherical chart leaves empty,
# in fractions of the axes it is drawn inside
BAR_THICKNESS = 0.035
BAR_GAP = 0.03
BAR_WIDTH_IN_CELL = 0.8


def is_hemispherical(ax: Axes) -> bool:
    """whether the chart draws a half-plane.

    a polar axes centres its sector in the axes box, so a wedge that spans pi
    fills the box edge to edge and only half its height, leaving a strip below
    the flat edge that a narrower wedge or a full circle does not."""
    if "polar" not in getattr(ax, "name", ""):
        return False
    lo, hi = ax.get_xlim()
    return abs(abs(hi - lo) - math.pi) < 1.0e-6


def wedge_extent(ax: Axes) -> tuple[float, float, float, float]:
    """the drawn sector's bounding box, in fractions of the axes box.

    the data and axes transforms share the axes position, so the fractions
    survive a layout reflow that moves or resizes the chart."""
    angles = np.linspace(*ax.get_xlim(), 129)
    radii = ax.get_ylim()
    corners = [(angle, radius) for radius in radii for angle in angles]
    fractions = ax.transAxes.inverted().transform(
        ax.transData.transform(corners)
    )
    return (
        float(fractions[:, 0].min()),
        float(fractions[:, 0].max()),
        float(fractions[:, 1].min()),
        float(fractions[:, 1].max()),
    )


def bar_cell(slot: int, total: int) -> int:
    """where panel `slot` sits in a left-to-right row of bars.

    the bars read in the order the panels do, so the leftmost bar describes the
    leftmost panel."""
    order = sorted(
        range(total),
        key=lambda index: (colorbar_side(index, total) != "left", index),
    )
    return order.index(slot)


def under_wedge_box(
    ax: Axes, slot: int, total: int
) -> Optional[list[float]]:
    """[x, y, width, height] for a bar beneath a hemispherical chart, or None
    when the chart leaves no strip to put one in."""
    if not is_hemispherical(ax):
        return None

    left, right, bottom, _ = wedge_extent(ax)

    # the bar and the tick labels below it have to clear the axes floor
    top = bottom - BAR_GAP
    if top - BAR_THICKNESS < 0.0:
        return None

    cell_width = (right - left) / total
    width = cell_width * BAR_WIDTH_IN_CELL
    x = left + cell_width * (bar_cell(slot, total) + 0.5) - width / 2

    return [x, top - BAR_THICKNESS, width, BAR_THICKNESS]


def remove_spines(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
