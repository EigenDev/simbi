# =============================================================================
# grid.py
#
# multi-panel grid plotting for comparing simulations side-by-side.
# drives components directly against raw matplotlib axes — bypasses the
# Figure class entirely since components are already axes-agnostic.
#
# usage:
#   from simbi.viz.grid import plot_grid
#   plot_grid(config, files, fields=["rho"], layout=(2, 2))
#
#   # or via cli:
#   simbi plot *.h5 --fields rho --subplot
#   simbi plot *.h5 --fields rho --layout 2 3 --auto-label
# =============================================================================
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure as MplFigure

from .builder import create_scalar_component, get_props
from .components.interface import ComponentProps
from .components.shared import ColormappedProps
from .config import VisualizationConfig
from .formatting import apply_scaling, remove_spines
from .pipeline import create_plot_data, load_data
from .pipeline.transforms import _compose_pcolormesh, compose_fields_for_render
from .registry import refinement_info, select_vector_component
from .types import ColorRange, FieldData
from .utility import get_field_str


def _compute_grid_shape(n: int) -> tuple[int, int]:
    """compute (nrows, ncols) for n panels. prefers wide grids."""
    if n <= 0:
        raise ValueError("need at least 1 panel")
    if n == 1:
        return (1, 1)
    if n == 2:
        return (1, 2)
    if n == 3:
        return (1, 3)

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return (nrows, ncols)


def _extract_panel_label(
    file_path: str, sim_data: Any, auto_label: bool
) -> str:
    """derive panel label from file or metadata."""
    if not auto_label:
        return Path(str(file_path)).stem

    try:
        meta = sim_data.metadata
    except AttributeError:
        return Path(str(file_path)).stem

    parts = []
    if getattr(meta, "time", None) is not None:
        parts.append(f"t={meta.time:.2g}")
    if getattr(meta, "gamma", None) is not None:
        parts.append(f"$\\gamma$={meta.gamma:.2g}")
    if getattr(meta, "coord_system", None):
        parts.append(meta.coord_system)

    return ", ".join(parts) if parts else Path(str(file_path)).stem


def _resolve_panel_props(
    base_props: Optional[dict[str, ComponentProps]],
    panel_overrides: Optional[dict[int, dict]],
    panel_idx: int,
) -> dict[str, ComponentProps]:
    """merge base props with per-panel overrides."""
    result = dict(base_props) if base_props else {}

    if not panel_overrides or panel_idx not in panel_overrides:
        return result

    overrides = panel_overrides[panel_idx]
    for comp_name, comp_overrides in overrides.items():
        if comp_name == "label":
            continue
        if comp_name in result:
            existing = result[comp_name]
            merged = {**existing.model_dump(), **comp_overrides}
            result[comp_name] = type(existing)(**merged)
        else:
            from .registry import get_props_class

            try:
                props_cls = get_props_class(comp_name)
                result[comp_name] = props_cls(**comp_overrides)
            except KeyError:
                pass

    return result


def _compute_global_range(
    all_fields: list[list[FieldData]],
) -> tuple[float, float]:
    """compute global vmin/vmax across all panels and fields."""
    global_min = float("inf")
    global_max = float("-inf")

    for panel_fields in all_fields:
        for field_data in panel_fields:
            vals = field_data.values
            fmin = float(np.nanmin(vals))
            fmax = float(np.nanmax(vals))
            global_min = min(global_min, fmin)
            global_max = max(global_max, fmax)

    if global_min == float("inf"):
        global_min, global_max = 0.0, 1.0

    return global_min, global_max


def _override_color_range(
    props: ComponentProps, vmin: float, vmax: float
) -> ComponentProps:
    """return a copy of props with color_range set to (vmin, vmax)."""
    if not isinstance(props, ColormappedProps):
        return props

    # only override if user didn't explicitly set a range
    cr = props.color_range
    if cr.min is not None or cr.max is not None:
        return props

    return props.model_copy(
        update={"color_range": ColorRange(min=vmin, max=vmax)}
    )


def _annotate_panel(ax, label: str, inside: bool, fontsize: int = 10) -> None:
    """place label as title or interior annotation."""
    if inside:
        ax.text(
            0.03,
            0.97,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
            ),
        )
    else:
        ax.set_title(label, fontsize=fontsize)


def plot_grid(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ("rho",),
    layout: Optional[tuple[int, int]] = None,
    panel_labels: Optional[Sequence[str]] = None,
    auto_label: bool = False,
    shared_colorbar: bool = True,
    annotate_inside: bool = False,
    wspace: Optional[float] = None,
    hspace: Optional[float] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    panel_overrides: Optional[dict[int, dict]] = None,
    **kwargs,
) -> MplFigure:
    """
    create a multi-panel grid figure comparing different checkpoint files.

    each file gets its own panel with the same field(s) rendered.
    """
    if not files:
        raise ValueError("no files provided")

    nfiles = len(files)
    nrows, ncols = layout if layout else _compute_grid_shape(nfiles)

    if nrows * ncols < nfiles:
        raise ValueError(f"layout {nrows}x{ncols} too small for {nfiles} files")

    # apply theme
    config.theme.apply(nfiles=nfiles, nfields=len(fields), overlay_mode=False)

    # load all data upfront (needed for shared colorbar range)
    panel_data = []
    panel_sim_data = []
    for file_path in files:
        sim_data = load_data(file_path)
        plot_data = create_plot_data(sim_data, list(fields), config)
        final_fields = compose_fields_for_render(plot_data.fields, config)
        panel_data.append((plot_data, final_fields))
        panel_sim_data.append(sim_data)

    # compute global color range for shared colorbar
    global_range = None
    if shared_colorbar:
        all_2d_fields = [
            [f for f in final_fields if f.ndim == 2]
            for _, final_fields in panel_data
        ]
        if any(all_2d_fields):
            gmin, gmax = _compute_global_range(all_2d_fields)
            global_range = (gmin, gmax)

    # detect projection from first panel
    first_fields = panel_data[0][1] if panel_data else []
    first_coord = (
        panel_sim_data[0].metadata.coord_system
        if panel_sim_data
        else "cartesian"
    )
    is_polar = (
        first_fields
        and first_fields[0].ndim == 2
        and first_coord == "spherical"
    )

    # create figure
    subplot_kw = {"projection": "polar"} if is_polar else {}
    base_w, base_h = config.figure.fig_size
    fig_w = base_w * min(ncols, 3) / 1.5
    fig_h = base_h * min(nrows, 3) / 1.5

    # default spacing: tight for interior annotations, moderate otherwise
    ws = wspace if wspace is not None else (0.05 if annotate_inside else 0.15)
    hs = hspace if hspace is not None else (0.05 if annotate_inside else 0.2)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        subplot_kw=subplot_kw,
        gridspec_kw={"wspace": ws, "hspace": hs},
    )

    # flatten axes to 1d array for uniform indexing
    if nrows == 1 and ncols == 1:
        axes_flat = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes_flat = np.atleast_1d(axes)
    else:
        axes_flat = axes.flatten()

    # render each panel
    last_mappable = None
    field_label = None
    has_colormapped = False

    for ii, (file_path, (plot_data, final_fields)) in enumerate(
        zip(files, panel_data)
    ):
        ax = axes_flat[ii]
        sim_data = panel_sim_data[ii]
        nlvls, use_polygons = refinement_info(plot_data.fields, config)

        # resolve per-panel props
        panel_props = _resolve_panel_props(component_props, panel_overrides, ii)

        # check for explicit label in panel_overrides
        override_label = None
        if panel_overrides and ii in panel_overrides:
            override_label = panel_overrides[ii].get("label")

        apply_scaling(ax, config.figure)

        for field_data in final_fields:
            component, props_key = create_scalar_component(
                field_data, panel_props, use_polygons,
                bodies=plot_data.body_collection,
            )

            # apply shared color range
            if global_range and shared_colorbar:
                component.props = _override_color_range(
                    component.props, *global_range
                )

            component.initialize(fig, ax)
            result = component.render(field_data, config.figure)

            # track mappable for colorbar
            if hasattr(result, "metadata") and result.metadata:
                mappable = result.metadata.get("mappable")
                if mappable is not None:
                    last_mappable = mappable

            if props_key in ("polygon", "quad"):
                has_colormapped = True

            if field_label is None:
                name = field_data.name
                if "_polygons" in name:
                    name = name.split("_polygons")[0]
                if name.endswith("_L"):
                    name = name.rsplit("_L", 1)[0]
                field_label = name

        # vector fields (quiver/streamplot) on this panel
        vector_fields = kwargs.get("vector_fields")
        if vector_fields and len(vector_fields) >= 2:
            vector_type = kwargs.get("vector_type", "quiver")
            vec_plot_data = create_plot_data(
                sim_data, list(vector_fields), config
            )

            vi_levels = [
                f
                for f in vec_plot_data.fields
                if f.name.startswith(vector_fields[0])
            ]
            vj_levels = [
                f
                for f in vec_plot_data.fields
                if f.name.startswith(vector_fields[1])
            ]

            if vi_levels and vj_levels:
                v1_field = _compose_pcolormesh(vi_levels)
                v2_field = _compose_pcolormesh(vj_levels)
                comp_cls, props_cls, props_key = select_vector_component(
                    vector_type
                )
                vec_props = get_props(panel_props, props_key, props_cls)
                vec_comp = comp_cls(vec_props)
                vec_comp.initialize(fig, ax)
                vec_comp.render([v1_field, v2_field], config.figure)

        # panel label
        if panel_labels and ii < len(panel_labels):
            label = panel_labels[ii]
        elif override_label:
            label = override_label
        else:
            label = _extract_panel_label(file_path, sim_data, auto_label)
        _annotate_panel(ax, label, inside=annotate_inside)

    # hide unused axes
    for jj in range(nfiles, nrows * ncols):
        axes_flat[jj].set_visible(False)

    # colorbar
    if last_mappable is not None and has_colormapped:
        cbar_label = get_field_str(field_label) if field_label else ""
        if shared_colorbar:
            # draw once so axes positions are finalized
            fig.canvas.draw()

            # find top of first row, bottom of last row (midpoints)
            top_ax = axes_flat[0]
            bot_ax = axes_flat[min(nfiles - 1, (nrows - 1) * ncols)]
            rightmost_ax = axes_flat[min(ncols - 1, nfiles - 1)]

            top_pos = top_ax.get_position()
            bot_pos = bot_ax.get_position()
            right_pos = rightmost_ax.get_position()

            top_mid = top_pos.y0 + top_pos.height * 0.5
            bot_mid = bot_pos.y0 + bot_pos.height * 0.5
            cbar_height = top_mid - bot_mid

            # single row: span full panel height
            if nrows == 1:
                top_mid = top_pos.y0 + top_pos.height
                bot_mid = top_pos.y0
                cbar_height = top_pos.height

            cbar_width = 0.02
            cbar_pad = 0.015
            cbar_x = right_pos.x0 + right_pos.width + cbar_pad

            cax = fig.add_axes([cbar_x, bot_mid, cbar_width, cbar_height])
            fig.colorbar(last_mappable, cax=cax, label=cbar_label)
        else:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            for ii in range(nfiles):
                divider = make_axes_locatable(axes_flat[ii])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(last_mappable, cax=cax, label=cbar_label)

    # per-panel formatting
    max_xticks = kwargs.get("max_xticks")
    max_yticks = kwargs.get("max_yticks")

    for ii in range(nfiles):
        row, col = divmod(ii, ncols)
        ax = axes_flat[ii]

        if ax.name == "polar":
            continue

        is_bottom = row == nrows - 1 or ii + ncols >= nfiles
        is_left = col == 0

        if not has_colormapped:
            remove_spines(ax)

        # limit tick density to avoid collisions at tight spacing
        if max_xticks is not None:
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=max_xticks))
        if max_yticks is not None:
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=max_yticks))

        # axis labels on edges only, tick numbers on every panel
        if is_bottom and config.figure.xlabel:
            ax.set_xlabel(config.figure.xlabel)
        elif not is_bottom:
            ax.set_xlabel("")

        if is_left and config.figure.ylabel:
            ax.set_ylabel(config.figure.ylabel)
        elif not is_left:
            ax.set_ylabel("")

    # save and show
    if save_as:
        fig.savefig(
            save_as,
            dpi=config.figure.dpi,
            bbox_inches="tight",
            transparent=config.figure.transparent,
        )
    if show:
        plt.show()

    return fig
