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
    """Sets the title on the appropriate object (fig or ax)."""
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

    if "polar" in ax.name:
        fig.suptitle(title_str)
    else:
        ax.set_title(title_str)


def apply_scaling(ax: Axes, config: FigureConfig) -> None:
    """Applies log or semilog scaling."""
    ax.set_xscale(config.xscale)
    ax.set_yscale(config.yscale)
    # Note: 'log' is handled by the component's norm


def apply_axis_labels(
    ax: Axes,
    config: FigureConfig,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Applies axis labels from config or derived values."""
    if ax.name == "polar":
        return

    # Use explicit config label if provided
    ax.set_xlabel(config.xlabel or xlabel or "$x$")
    ax.set_ylabel(config.ylabel or ylabel or "$y$")


def apply_axis_limits(ax: Axes, config: FigureConfig) -> None:
    """Sets axis limits if provided in config."""
    if config.xlims:
        ax.set_xlim(config.xlims.min, config.xlims.max)
    if config.ylims:
        ax.set_ylim(config.ylims.min, config.ylims.max)


def apply_legend(ax: Axes) -> None:
    """Adds a legend to the axes."""
    ax.legend(loc="best")


def add_colorbar(
    fig: Figure,
    artist: Any,
    cax: Axes,  # The colorbar axes MUST be provided
    label: Optional[str] = None,
    orientation: str = "vertical",
) -> Colorbar:
    """
    Adds a colorbar to the *provided* cax.
    This is now a simple formatter.
    """
    cbar = fig.colorbar(artist, cax=cax, orientation=orientation)
    if label:
        cbar.set_label(get_field_str(label))
    return cbar


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
          - "label" (str): single label → use as ylabel
          - "labels" (list): if len == 1 → ylabel, no legend
                             if len > 1 → legend, no ylabel

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
        # Title and time
        time = getattr(first_data, "time", None)
        set_title(main_ax, fig, self.style, time)

        # Normalize rendered_artists entries first (needed for label extraction)
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
        # "label" (str) → single ylabel
        # "labels" (list) → if 1 item: ylabel, no legend; if >1: legend, no ylabel
        metadata_label: Optional[str] = None
        metadata_labels: list = []
        use_legend_from_metadata = False

        for _, metadata in normalized:
            if metadata and isinstance(metadata, dict):
                if "label" in metadata and metadata["label"]:
                    metadata_label = metadata["label"]
                if "labels" in metadata and metadata["labels"]:
                    metadata_labels = list(metadata["labels"])
                    if len(metadata_labels) > 1:
                        use_legend_from_metadata = True

        # determine ylabel from metadata if not explicitly provided
        derived_ylabel = None
        if len(metadata_labels) == 1:
            # single label in list → use as ylabel, suppress legend
            derived_ylabel = get_field_str(metadata_labels[0])
            show_legend = False
        elif len(metadata_labels) > 1:
            # multiple labels → show legend, no ylabel from metadata
            use_legend_from_metadata = True
            derived_ylabel = None
        elif metadata_label:
            # single "label" key → use as ylabel
            derived_ylabel = get_field_str(metadata_label)

        # determine axis labels
        if xlabel is None or ylabel is None:
            try:
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
                if ylabel is None and hasattr(first_data, "name"):
                    ylabel = get_field_str(first_data.name)
            except Exception:
                pass

        if ylabel and not ylabel.startswith("$"):
            ylabel = get_field_str(ylabel)
        # axis labels & limits
        try:
            apply_axis_labels(main_ax, self.style, xlabel, ylabel)
        except Exception:
            pass

        try:
            apply_axis_limits(main_ax, self.style)
        except Exception:
            pass

        # continue processing normalized list (already built above)
        # Colorbar: find the first mappable among normalized artists
        mappable = None
        for artists, metadata in normalized:
            if not isinstance(artists, dict):
                continue
            # common keys for mappables
            if "mesh" in artists and artists["mesh"] is not None:
                mappable = artists["mesh"]
                break
            if "collection" in artists and artists["collection"] is not None:
                mappable = artists["collection"]
                break
            # fallback: look for any artist that looks like a ScalarMappable
            for a in artists.values():
                try:
                    if hasattr(a, "get_array") or hasattr(a, "get_cmap"):
                        mappable = a
                        break
                except Exception:
                    continue
            if mappable is not None:
                break

        if mappable is not None:
            try:
                self._format_colorbar(fig, main_ax, mappable, first_data)
            except Exception:
                # ensure formatting step never halts the render pipeline
                pass

        # Legend: only show if there are line-like artists (or metadata indicates labels)
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

            # some rendered artists may have multiple
            # labels and lines, so we should search through
            # the list and verify if multiple labels exist
            multiple_labels = False
            for artists, metadata in normalized:
                if metadata and isinstance(metadata, dict):
                    labels = metadata.get("labels", [])
                    if labels and len(labels) > 1:
                        multiple_labels = True
                        break

            # but not if we used the single label as ylabel
            if should_show_legend and multiple_labels:
                apply_legend(main_ax)
        except Exception:
            pass

        # remove top/right spines for cleaner look
        # but not if plotting multidim plots (e.g, polygons, quadplot)
        # if any("lines" in x.keys() for x in rendered_artists) and not any(
        #     isinstance(x["collection"], (PolyCollection, QuadMesh))
        #     for x in rendered_artists
        # ):
        try:
            remove_spines(main_ax)
        except Exception:
            pass

    def _format_colorbar(
        self, fig: Figure, ax: Axes, artist: Any, field_data: Any
    ):
        """
        Place or update a colorbar appropriate for the axes projection.

        if a colorbar was previously created for this axes, update it in-place
        (mappable, norm, label) instead of creating a new axes. this prevents
        accumulation of colorbar axes during animations and keeps layout stable.
        """
        from matplotlib.colorbar import Colorbar as MplColorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        label = getattr(field_data, "name", None)
        if label and "_polygons" in label:
            label = label.split("_polygons")[0]

        label = get_field_str(label) if label else None

        # try to pick up an explicit color range from the field_data if present
        color_range = None
        try:
            color_range = getattr(field_data, "color_range", None)
        except Exception:
            color_range = None

        # Helper: attempt to update an existing colorbar in-place
        existing_cbar = getattr(ax, "_simbi_colorbar", None)
        if existing_cbar is not None and isinstance(existing_cbar, MplColorbar):
            try:
                # update the underlying mappable reference
                existing_cbar.mappable = artist
                # apply explicit color_range if provided
                if color_range and isinstance(color_range, dict):
                    vmin = color_range.get("min")
                    vmax = color_range.get("max")
                    try:
                        if vmin is not None and vmax is not None:
                            artist.set_clim(vmin, vmax)
                    except Exception:
                        # some artist types may not support set_clim; ignore
                        pass
                # ensure the colorbar reflects the new mappable / normalization
                try:
                    existing_cbar.update_normal(artist)
                except Exception:
                    # older mpl versions / some mappables may require updating via mappable.set_norm
                    pass

                if label:
                    try:
                        existing_cbar.set_label(label)
                    except Exception:
                        pass

                # successful update — no need to recreate
                return
            except Exception:
                # if updating fails, remove and proceed to recreate
                try:
                    existing_cbar.remove()
                except Exception:
                    pass
                try:
                    delattr(ax, "_simbi_colorbar")
                except Exception:
                    pass

        # if we reach here, create a new colorbar and attach it to the axes
        cax = None
        orientation = "vertical"

        if hasattr(ax, "name") and "polar" in getattr(ax, "name"):
            # Polar-specific placements (horizontal for half-sphere, vertical otherwise)
            try:
                theta = field_data.domain[1]
                max_angle = theta[-1]
                half_sphere = max_angle == 0.5 * 3.141592653589793
            except Exception:
                half_sphere = False

            if half_sphere:
                # place a small horizontal colorbar below the plot
                polar_pos = ax.get_position()
                width = min(0.6, 0.78)
                x = polar_pos.x0 + (polar_pos.width - width) / 2 - 0.01
                cax = fig.add_axes((x, 0.2, width, 0.03))
                orientation = "horizontal"
            else:
                # vertical alongside polar plot
                polar_pos = ax.get_position()
                height = 0.8
                x = polar_pos.x0 + polar_pos.width + 0.05
                y = polar_pos.y0 + (polar_pos.height - height) / 2
                cax = fig.add_axes((x, y, 0.03, height))
                orientation = "vertical"
        else:
            # Cartesian default: use an axes divider for a vertical colorbar
            try:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                orientation = "vertical"
            except Exception:
                # fallback: allocate a tiny axes to the right of the main axes
                pos = ax.get_position()
                try:
                    cax = fig.add_axes(
                        (pos.x1 + 0.02, pos.y0, 0.03, pos.height)
                    )
                    orientation = "vertical"
                except Exception:
                    # last resort: give up creating a colorbar
                    return

        # create the colorbar and store a reference on the axes for future updates
        try:
            cbar = fig.colorbar(artist, cax=cax, orientation=orientation)
            if label:
                try:
                    cbar.set_label(label)
                except Exception:
                    pass
            # store for later in-place updates during animations
            try:
                setattr(ax, "_simbi_colorbar", cbar)
            except Exception:
                # ignore attribute set failures
                pass
        except Exception:
            # creating a colorbar failed; ensure no left-over attribute
            try:
                if hasattr(ax, "_simbi_colorbar"):
                    delattr(ax, "_simbi_colorbar")
            except Exception:
                pass


def remove_spines(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
