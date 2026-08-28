import math
from typing import Any, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from simbi.viz.utility import get_field_str

from .config import FigureConfig
from .types import RenderResult

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


def find_mappables(
    results: "Sequence[RenderResult]",
) -> list[tuple[Any, Optional[str]]]:
    """the colour-mapped artists among the frame's results, each with the name
    of the quantity it draws, in the order they were rendered.

    one entry is one quantity wanting a scale of its own. a vector overlay is
    colour-mapped too, but it reads off the field beneath it and declares no
    mappable, so it takes no bar."""
    return [
        (result.mappable, result.colorbar_label)
        for result in results
        if result.mappable is not None
    ]


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

        results = [entry for entry in rendered_artists if entry is not None]

        # one drawn series names the y axis; several name a legend
        labels: list[str] = []
        for result in results:
            labels += [name for name in result.labels if name]

        derived_ylabel = None
        label_set = list(set(labels))
        use_legend_from_metadata = len(label_set) > 1
        if len(label_set) == 1:
            derived_ylabel = get_field_str(label_set[0])
            show_legend = False

        # determine axis labels. with no derived label a 1D plot shows a legend
        # and leaves the y axis unlabeled
        if xlabel is None or ylabel is None:
            # the data's own axis names come first when it carries them
            axis_names = getattr(first_data, "axis_names", None)
            if axis_names and isinstance(axis_names, (list, tuple)):
                if xlabel is None and len(axis_names) >= 1:
                    xlabel = LABEL_MAP.get(axis_names[0], axis_names[0])
                if ylabel is None and len(axis_names) >= 2:
                    ylabel = LABEL_MAP.get(axis_names[1], axis_names[1])

            if ylabel is None and derived_ylabel:
                ylabel = derived_ylabel

            # a single drawn series names the y axis after the quantity
            if (
                ylabel is None
                and hasattr(first_data, "name")
                and len(label_set) == 1
            ):
                ylabel = get_field_str(first_data.name)

        # a plot that loses its labels or limits is still a plot, so these do
        # not halt the render -- but an unlabeled axis shipped as a success is
        # a defect, and it is the silence that lets one ship
        try:
            apply_axis_labels(main_ax, self.style, xlabel, ylabel)
        except Exception as exc:
            warn_once(
                f"axis-labels:{type(exc).__name__}",
                f"axis labels failed: {exc}",
            )

        try:
            apply_axis_limits(main_ax, self.style)
        except Exception as exc:
            warn_once(
                f"axis-limits:{type(exc).__name__}",
                f"axis limits failed: {exc}",
            )

        # the chart is oriented before anything is placed around it: where a
        # wedge sits in the axes box, and so where there is room for a colorbar,
        # follows from the zero direction and the sense of increasing angle
        if main_ax.name == "polar":
            main_ax.grid(False)
            main_ax.set_theta_zero_location("N")
            main_ax.set_theta_direction(-1)

            # the scale option keeps matplotlib's automatic tick locators, so
            # the labeled radii follow the view limits each frame sets and
            # stay current as a moving mesh expands. blanking installs a fixed
            # empty formatter instead, which holds every later frame blank.
            if getattr(self.style, "show_scales", False):
                style_radial_scale(main_ax, self.style)
            else:
                main_ax.set_xticklabels([])
                main_ax.set_yticklabels([])

        try:
            self._format_colorbars(
                fig, main_ax, find_mappables(results), first_data
            )
        except Exception as exc:
            # a plot missing its colorbar is still a plot, so the render
            # continues -- but it is a defect: a chart needs its scale to be
            # readable, so the failure is announced once
            warn_once(
                f"colorbar:{type(exc).__name__}", f"colorbar failed: {exc}"
            )

        # a legend describes line-like artists; a field render is described by
        # its colorbar instead
        has_line_like = any(
            result.labels
            or "line" in result.artists
            or "lines" in result.artists
            for result in results
        )

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
        except Exception as exc:
            warn_once(f"legend:{type(exc).__name__}", f"legend failed: {exc}")

        # remove top/right spines only for 1D plots
        if main_ax.name != "polar" and ndim == 1:
            try:
                remove_spines(main_ax)
            except Exception as exc:
                warn_once(
                    f"spines:{type(exc).__name__}", f"spine removal failed: {exc}"
                )




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

        # the radial scale labels occupy a band below a hemisphere's flat
        # edge, so the bars drop beneath it rather than print through it
        clearance = (
            RADIAL_LABEL_CLEARANCE
            if getattr(self.style, "show_scales", False)
            else 0.0
        )

        for slot, (artist, name) in enumerate(entries):
            label = _colorbar_label(name, field_data)
            existing = bars.get(slot)

            if existing is not None:
                existing.mappable = artist
                existing.update_normal(artist)
                if label:
                    existing.set_label(label)
                continue

            box = under_wedge_box(ax, slot, len(entries), clearance=clearance)
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


# the labeled radial ticks a chart carries at most. each label is a number
# plus a unit string laid along half the flat edge, so a handful is what fits
RADIAL_TICK_BINS = 4


class ScaledRadialLocator(MaxNLocator):
    """tick radii that land on round values of r / scale.

    picking nice values in data radii and dividing afterwards prints every
    label at full float width (0.45 / 0.8779 = 0.512587...); picking them in
    the scaled units and mapping back keeps the labels short."""

    def __init__(self, scale: float):
        super().__init__(nbins=RADIAL_TICK_BINS, steps=[1, 2, 2.5, 5, 10])
        self._scale = scale

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        scaled = super().tick_values(vmin / self._scale, vmax / self._scale)
        return [value * self._scale for value in scaled]


def style_radial_scale(ax: Axes, style: Any) -> None:
    """label the radial axis in the declared physical units.

    the ticks land on round values of r / length_scale with the unit string
    attached, so a run in code units reads in physical ones. locator and
    formatter both run at draw time, which keeps the labels current as a
    moving mesh carries the radial limits outward."""
    from matplotlib.projections.polar import RadialLocator
    from matplotlib.ticker import FuncFormatter

    scale = getattr(style, "length_scale", None) or 1.0
    units = getattr(style, "length_units", "")
    suffix = f" {units}" if units else ""

    def format_radius(value: float, _position: int) -> str:
        return f"{value / scale:g}{suffix}"

    # the radial wrapper keeps polar-specific behavior (ticks clipped to the
    # visible annulus) around the scaled tick choice
    ax.yaxis.set_major_locator(RadialLocator(ScaledRadialLocator(scale), ax))
    ax.yaxis.set_major_formatter(FuncFormatter(format_radius))


def _colorbar_label(name: Optional[str], field_data: Any) -> Optional[str]:
    """the axis label for the bar describing `name`, in display notation."""
    label = name or getattr(field_data, "name", None)
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
# the band the radial scale labels occupy along the flat edge, kept clear of
# the bars when the scale is drawn
RADIAL_LABEL_CLEARANCE = 0.05


def is_hemispherical(ax: Axes) -> bool:
    """whether the chart draws a half-plane.

    a polar axes centers its sector in the axes box, so a wedge that spans pi
    fills the box edge to edge and half its height, leaving a blank strip below
    the flat edge; a narrower wedge or a full circle fills the box evenly."""
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
    ax: Axes, slot: int, total: int, clearance: float = 0.0
) -> Optional[list[float]]:
    """[x, y, width, height] for a bar beneath a hemispherical chart, or None
    when the chart leaves no strip to put one in. `clearance` widens the gap
    below the flat edge, in axes fractions, leaving room for scale labels."""
    if not is_hemispherical(ax):
        return None

    left, right, bottom, _ = wedge_extent(ax)

    # the bar and the tick labels below it have to clear the axes floor
    top = bottom - BAR_GAP - clearance
    if top - BAR_THICKNESS < 0.0:
        return None

    cell_width = (right - left) / total
    width = cell_width * BAR_WIDTH_IN_CELL
    x = left + cell_width * (bar_cell(slot, total) + 0.5) - width / 2

    return [x, top - BAR_THICKNESS, width, BAR_THICKNESS]


def remove_spines(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
