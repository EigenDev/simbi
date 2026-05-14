# =============================================================================
# grid.py
#
# multi-panel grid plotting and animation.
# drives components directly against raw matplotlib axes — bypasses the
# Figure class entirely since components are already axes-agnostic.
#
# features:
# - per-panel slice specs for showing different views of the same data
# - per-panel axis limits for zoom control
# - grid animation across checkpoint sequences
#
# usage:
#   from simbi.viz.grid import plot_grid, animate_grid
#   plot_grid(config, files, fields=["rho"], layout=(2, 2))
#
#   # or via cli:
#   simbi plot *.h5 --fields rho --subplot
#   simbi plot *.h5 --fields rho --subplot --animate --config views.yaml
# =============================================================================
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure

from .builder import create_scalar_component, get_props
from .components.interface import ComponentProps
from .components.shared import ColormappedProps
from .config import FigureConfig, VisualizationConfig
from .config_loader import resolve_per_file_props
from .formatting import apply_scaling, remove_spines
from .pipeline import create_plot_data, load_data
from .pipeline.plot_data import apply_slicing, prepare_fields
from .pipeline.transforms import _compose_pcolormesh, compose_fields_for_render
from .registry import refinement_info, select_vector_component
from .types import ColorRange, CoordSystem, FieldData, PlotData
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


def _build_panel_specs(
    files: Sequence[str],
    panel_overrides: Optional[dict[int, dict]],
    npanels: int,
) -> list[dict]:
    """build per-panel spec dicts from files and overrides."""
    specs = []
    for ii in range(npanels):
        overrides = (panel_overrides or {}).get(ii, {})
        file_ref = overrides.get("file")
        if file_ref is None:
            src = str(files[ii]) if ii < len(files) else str(files[-1])
        elif isinstance(file_ref, int):
            src = str(files[file_ref])
        else:
            src = str(file_ref)
        specs.append({
            "file": src,
            "slice": overrides.get("slice"),
            "xlims": overrides.get("xlims"),
            "ylims": overrides.get("ylims"),
        })
    return specs


def _prepare_panel_fields(
    sim_data: Any,
    field_names: Sequence[str],
    config: VisualizationConfig,
    panel_slice: Optional[dict[str, float]] = None,
) -> tuple[PlotData, list[FieldData]]:
    """prepare sliced and composed fields for a single panel."""
    full_fields = prepare_fields(sim_data, field_names, config)
    slice_spec = panel_slice if panel_slice is not None else config.plot.slice
    sliced_fields = apply_slicing(full_fields, slice_spec)
    plot_data = PlotData(
        fields=sliced_fields,
        body_collection=sim_data.body_collection,
        time=sim_data.metadata.time,
        dimensions=sliced_fields[0].ndim if sliced_fields else 0,
        coord_system=CoordSystem(sim_data.metadata.coord_system),
        hierarchy=sim_data.hierarchy() if sim_data.has_refinement() else None,
    )
    final_fields = compose_fields_for_render(plot_data.fields, config)
    return plot_data, final_fields


def _create_grid_figure(
    config: VisualizationConfig,
    nrows: int,
    ncols: int,
    is_polar: bool,
    annotate_inside: bool,
    wspace: Optional[float],
    hspace: Optional[float],
) -> tuple[MplFigure, np.ndarray]:
    """create matplotlib figure and flattened axes array for grid layout."""
    subplot_kw = {"projection": "polar"} if is_polar else {}
    base_w, base_h = config.figure.fig_size
    fig_w = base_w * min(ncols, 3) / 1.5
    fig_h = base_h * min(nrows, 3) / 1.5
    ws = wspace if wspace is not None else (0.05 if annotate_inside else 0.15)
    hs = hspace if hspace is not None else (0.05 if annotate_inside else 0.2)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        subplot_kw=subplot_kw,
        gridspec_kw={"wspace": ws, "hspace": hs},
    )
    if nrows == 1 and ncols == 1:
        axes_flat = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes_flat = np.atleast_1d(axes)
    else:
        axes_flat = axes.flatten()
    return fig, axes_flat


def _apply_panel_limits(
    ax: Axes, spec: dict, figure_config: FigureConfig
) -> None:
    """apply per-panel axis limits, falling back to global config."""
    panel_xlims = spec.get("xlims")
    panel_ylims = spec.get("ylims")
    if panel_xlims is not None:
        ax.set_xlim(*panel_xlims)
    elif figure_config.xlims is not None:
        if figure_config.xlims.min is not None or figure_config.xlims.max is not None:
            ax.set_xlim(figure_config.xlims.min, figure_config.xlims.max)
    if panel_ylims is not None:
        ax.set_ylim(*panel_ylims)
    elif figure_config.ylims is not None:
        if figure_config.ylims.min is not None or figure_config.ylims.max is not None:
            ax.set_ylim(figure_config.ylims.min, figure_config.ylims.max)


def _add_colorbar(
    fig: MplFigure,
    axes_flat: np.ndarray,
    npanels: int,
    nrows: int,
    ncols: int,
    last_mappable: Any,
    field_label: Optional[str],
    shared_colorbar: bool,
) -> None:
    """add shared or per-panel colorbar to grid figure."""
    if last_mappable is None:
        return
    cbar_label = get_field_str(field_label) if field_label else ""
    if shared_colorbar:
        fig.canvas.draw()
        top_ax = axes_flat[0]
        bot_ax = axes_flat[min(npanels - 1, (nrows - 1) * ncols)]
        rightmost_ax = axes_flat[min(ncols - 1, npanels - 1)]
        top_pos = top_ax.get_position()
        bot_pos = bot_ax.get_position()
        right_pos = rightmost_ax.get_position()
        top_mid = top_pos.y0 + top_pos.height * 0.5
        bot_mid = bot_pos.y0 + bot_pos.height * 0.5
        cbar_height = top_mid - bot_mid
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

        for ii in range(npanels):
            divider = make_axes_locatable(axes_flat[ii])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(last_mappable, cax=cax, label=cbar_label)


def _add_per_column_colorbars(
    fig: MplFigure,
    axes_flat: np.ndarray,
    nrows: int,
    ncols: int,
    column_mappables: list[Any],
    column_labels: list[str],
) -> None:
    """add one colorbar per column, spanning the full column height."""
    fig.canvas.draw()
    cbar_width = 0.02
    cbar_pad = 0.015
    for col in range(ncols):
        mappable = column_mappables[col]
        if mappable is None:
            continue
        top_ax = axes_flat[col]
        bot_ax = axes_flat[(nrows - 1) * ncols + col]
        top_pos = top_ax.get_position()
        bot_pos = bot_ax.get_position()
        col_ax = axes_flat[col]
        col_pos = col_ax.get_position()
        cbar_x = col_pos.x0 + col_pos.width + cbar_pad
        cbar_bottom = bot_pos.y0
        cbar_height = top_pos.y0 + top_pos.height - bot_pos.y0
        cax = fig.add_axes([cbar_x, cbar_bottom, cbar_width, cbar_height])
        label = get_field_str(column_labels[col]) if column_labels[col] else ""
        fig.colorbar(mappable, cax=cax, label=label)


def _format_grid_panels(
    axes_flat: np.ndarray,
    npanels: int,
    nrows: int,
    ncols: int,
    config: VisualizationConfig,
    has_colormapped: bool,
    **kwargs,
) -> None:
    """apply per-panel formatting (edge labels, tick density, spines)."""
    max_xticks = kwargs.get("max_xticks")
    max_yticks = kwargs.get("max_yticks")
    for ii in range(npanels):
        row, col = divmod(ii, ncols)
        ax = axes_flat[ii]
        if ax.name == "polar":
            continue
        is_bottom = row == nrows - 1 or ii + ncols >= npanels
        is_left = col == 0
        if not has_colormapped:
            remove_spines(ax)
        if max_xticks is not None:
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=max_xticks))
        if max_yticks is not None:
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=max_yticks))
        if is_bottom and config.figure.xlabel:
            ax.set_xlabel(config.figure.xlabel)
        elif not is_bottom:
            ax.set_xlabel("")
        if is_left and config.figure.ylabel:
            ax.set_ylabel(config.figure.ylabel)
        elif not is_left:
            ax.set_ylabel("")


def _plot_file_field_grid(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str],
    nrows: int,
    ncols: int,
    npanels: int,
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
    file-field grid: rows = files, cols = fields.

    panel index = row * ncols + col, where row indexes files and col indexes
    fields. colorbars are per-column (one per field) when shared_colorbar is
    true, since each column shows the same physical quantity across files.
    """
    nfiles = len(files)
    nfields = len(fields)

    # load each file once; prepare fields for all requested field names
    file_sim_data = []
    file_panel_data: list[list[tuple[PlotData, list[FieldData]]]] = []
    for file_path in files:
        sim_data = load_data(str(file_path))
        file_sim_data.append(sim_data)
        per_field = []
        for field_name in fields:
            plot_data, final_fields = _prepare_panel_fields(
                sim_data, [field_name], config,
            )
            per_field.append((plot_data, final_fields))
        file_panel_data.append(per_field)

    # per-column global color range (same field across files)
    column_ranges: list[Optional[tuple[float, float]]] = [None] * nfields
    if shared_colorbar:
        for col in range(nfields):
            col_2d = []
            for row in range(nfiles):
                _, flds = file_panel_data[row][col]
                col_2d.append([f for f in flds if f.ndim == 2])
            if any(col_2d):
                gmin, gmax = _compute_global_range(col_2d)
                column_ranges[col] = (gmin, gmax)

    # detect projection from first panel
    first_fields = file_panel_data[0][0][1] if file_panel_data else []
    first_coord = (
        file_sim_data[0].metadata.coord_system
        if file_sim_data
        else "cartesian"
    )
    is_polar = (
        first_fields
        and first_fields[0].ndim == 2
        and first_coord == "spherical"
    )

    fig, axes_flat = _create_grid_figure(
        config, nrows, ncols, is_polar, annotate_inside, wspace, hspace
    )

    column_mappables: list[Any] = [None] * nfields
    column_labels: list[str] = [""] * nfields
    has_colormapped = False

    for row in range(nfiles):
        sim_data = file_sim_data[row]
        for col in range(nfields):
            panel_idx = row * ncols + col
            ax = axes_flat[panel_idx]
            plot_data, final_fields = file_panel_data[row][col]
            nlvls, use_polygons = refinement_info(plot_data.fields, config)
            panel_props = resolve_per_file_props(
                component_props, panel_overrides, panel_idx
            )

            apply_scaling(ax, config.figure)

            for field_data in final_fields:
                component, props_key = create_scalar_component(
                    field_data, panel_props, use_polygons,
                    bodies=plot_data.body_collection,
                )

                col_range = column_ranges[col]
                if col_range and shared_colorbar:
                    component.props = _override_color_range(
                        component.props, *col_range
                    )

                component.initialize(fig, ax)
                result = component.render(field_data, config.figure)

                if hasattr(result, "metadata") and result.metadata:
                    mappable = result.metadata.get("mappable")
                    if mappable is not None:
                        column_mappables[col] = mappable

                if props_key in ("polygon", "quad"):
                    has_colormapped = True

                if not column_labels[col]:
                    name = field_data.name
                    if "_polygons" in name:
                        name = name.split("_polygons")[0]
                    if name.endswith("_L"):
                        name = name.rsplit("_L", 1)[0]
                    column_labels[col] = name

            # apply axis limits from config or panel overrides
            panel_spec = (
                (panel_overrides or {}).get(panel_idx, {})
            )
            _apply_panel_limits(ax, panel_spec, config.figure)

            # panel label: panel_labels are per-row (per-file) and
            # broadcast across all columns in that row
            if panel_labels and row < len(panel_labels):
                label = panel_labels[row]
            elif panel_overrides and panel_idx in panel_overrides:
                label = panel_overrides[panel_idx].get("label", "")
            else:
                label = _extract_panel_label(
                    str(files[row]), sim_data, auto_label
                )
            _annotate_panel(ax, label, inside=annotate_inside)

    # hide unused axes
    for jj in range(npanels, nrows * ncols):
        axes_flat[jj].set_visible(False)

    if has_colormapped:
        if shared_colorbar:
            _add_per_column_colorbars(
                fig, axes_flat, nrows, ncols,
                column_mappables, column_labels,
            )
        else:
            _add_colorbar(
                fig, axes_flat, npanels, nrows, ncols,
                column_mappables[-1], column_labels[-1], False,
            )

    _format_grid_panels(
        axes_flat, npanels, nrows, ncols, config, has_colormapped, **kwargs
    )

    eff_ws = wspace if wspace is not None else (0.05 if annotate_inside else 0.15)
    eff_hs = hspace if hspace is not None else (0.05 if annotate_inside else 0.2)
    fig.subplots_adjust(wspace=eff_ws, hspace=eff_hs)

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
    create a multi-panel grid figure.

    supports per-panel slice specs and axis limits via panel_overrides:
        panel_overrides = {
            0: {"slice": {"x3": 0.0}, "label": "x-y plane"},
            1: {"slice": {"x2": 0.0}, "xlims": [-1, 1], "ylims": [-1, 1]},
        }
    """
    if not files:
        raise ValueError("no files provided")

    nfiles = len(files)
    nfields = len(fields)
    multi_field = nfields > 1

    if multi_field:
        # file-field grid: rows = files, cols = fields
        npanels = nfiles * nfields
        nrows = nfiles
        ncols = nfields
        if layout:
            nrows, ncols = layout
            if nrows * ncols < npanels:
                raise ValueError(
                    f"layout {nrows}x{ncols} too small for "
                    f"{nfiles} files x {nfields} fields = {npanels} panels"
                )
    else:
        npanels = nfiles
        nrows, ncols = layout if layout else _compute_grid_shape(npanels)
        if nrows * ncols < npanels:
            raise ValueError(
                f"layout {nrows}x{ncols} too small for {npanels} files"
            )

    config.theme.apply(nfiles=npanels, nfields=nfields, overlay_mode=False)

    if multi_field:
        return _plot_file_field_grid(
            config=config,
            files=files,
            fields=fields,
            nrows=nrows,
            ncols=ncols,
            npanels=npanels,
            panel_labels=panel_labels,
            auto_label=auto_label,
            shared_colorbar=shared_colorbar,
            annotate_inside=annotate_inside,
            wspace=wspace,
            hspace=hspace,
            save_as=save_as,
            show=show,
            component_props=component_props,
            panel_overrides=panel_overrides,
            **kwargs,
        )

    # --- single-field grid (original behavior) ---

    # build per-panel specs and load data with per-panel slicing
    panel_specs = _build_panel_specs(files, panel_overrides, npanels)
    panel_data = []
    panel_sim_data = []
    for spec in panel_specs:
        sim_data = load_data(spec["file"])
        plot_data, final_fields = _prepare_panel_fields(
            sim_data, list(fields), config, spec["slice"]
        )
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

    fig, axes_flat = _create_grid_figure(
        config, nrows, ncols, is_polar, annotate_inside, wspace, hspace
    )

    # render each panel
    last_mappable = None
    field_label = None
    has_colormapped = False

    for ii, spec in enumerate(panel_specs):
        ax = axes_flat[ii]
        plot_data, final_fields = panel_data[ii]
        sim_data = panel_sim_data[ii]
        nlvls, use_polygons = refinement_info(plot_data.fields, config)

        panel_props = resolve_per_file_props(component_props, panel_overrides, ii)

        override_label = None
        if panel_overrides and ii in panel_overrides:
            override_label = panel_overrides[ii].get("label")

        apply_scaling(ax, config.figure)

        for field_data in final_fields:
            component, props_key = create_scalar_component(
                field_data, panel_props, use_polygons,
                bodies=plot_data.body_collection,
            )

            if global_range and shared_colorbar:
                component.props = _override_color_range(
                    component.props, *global_range
                )

            component.initialize(fig, ax)
            result = component.render(field_data, config.figure)

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

        # vector fields on this panel
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

        _apply_panel_limits(ax, spec, config.figure)

        # panel label
        if panel_labels and ii < len(panel_labels):
            label = panel_labels[ii]
        elif override_label:
            label = override_label
        else:
            label = _extract_panel_label(spec["file"], sim_data, auto_label)
        _annotate_panel(ax, label, inside=annotate_inside)

    # hide unused axes
    for jj in range(npanels, nrows * ncols):
        axes_flat[jj].set_visible(False)

    if has_colormapped:
        _add_colorbar(
            fig, axes_flat, npanels, nrows, ncols,
            last_mappable, field_label, shared_colorbar,
        )

    _format_grid_panels(
        axes_flat, npanels, nrows, ncols, config, has_colormapped, **kwargs
    )

    # reassert spacing so bbox_inches="tight" doesn't override it
    eff_ws = wspace if wspace is not None else (0.05 if annotate_inside else 0.15)
    eff_hs = hspace if hspace is not None else (0.05 if annotate_inside else 0.2)
    fig.subplots_adjust(wspace=eff_ws, hspace=eff_hs)

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


def animate_grid(
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
    animate a multi-panel grid across a sequence of checkpoint files.

    panel layout is fixed; files provide the time dimension.
    each panel can have its own slice and axis limits via panel_overrides.
    all panels render the current frame file unless a panel specifies a
    fixed file source via the "file" key in its override.
    """
    if not files:
        raise ValueError("no files provided")
    if len(files) < 2:
        raise ValueError("animation requires at least 2 files")
    if not panel_overrides:
        raise ValueError(
            "animate_grid requires panel_overrides to define panel views"
        )

    npanels = max(panel_overrides.keys()) + 1
    nrows, ncols = layout if layout else _compute_grid_shape(npanels)
    if nrows * ncols < npanels:
        raise ValueError(f"layout {nrows}x{ncols} too small for {npanels} panels")

    # build panel specs — file=None means use the frame file
    panel_specs = []
    for ii in range(npanels):
        overrides = panel_overrides.get(ii, {})
        panel_specs.append({
            "file": overrides.get("file"),
            "slice": overrides.get("slice"),
            "xlims": overrides.get("xlims"),
            "ylims": overrides.get("ylims"),
        })

    fps = kwargs.pop("fps", None) or config.animation.frame_rate
    nframes = len(files)
    field_names = list(fields)
    config.theme.apply(nfiles=npanels, nfields=len(fields), overlay_mode=False)

    # load all panels for a single frame, caching shared file loads
    def _load_frame(frame_file: str):
        sim_cache: dict[str, Any] = {}
        results = []
        for spec in panel_specs:
            src = spec["file"] or frame_file
            if src not in sim_cache:
                sim_cache[src] = load_data(src)
            sim_data = sim_cache[src]
            plot_data, final_fields = _prepare_panel_fields(
                sim_data, field_names, config, spec["slice"]
            )
            results.append((sim_data, plot_data, final_fields))
        return results

    # frame 0: full setup
    frame0_data = _load_frame(files[0])

    global_range = None
    if shared_colorbar:
        all_2d = [
            [f for f in flds if f.ndim == 2]
            for _, _, flds in frame0_data
        ]
        if any(all_2d):
            gmin, gmax = _compute_global_range(all_2d)
            global_range = (gmin, gmax)

    first_fields = frame0_data[0][2] if frame0_data else []
    first_coord = (
        frame0_data[0][0].metadata.coord_system
        if frame0_data
        else "cartesian"
    )
    is_polar = (
        first_fields
        and first_fields[0].ndim == 2
        and first_coord == "spherical"
    )

    fig, axes_flat = _create_grid_figure(
        config, nrows, ncols, is_polar, annotate_inside, wspace, hspace
    )

    # render frame 0 and store components for reuse
    panel_components: list[list[tuple[Any, str]]] = []
    last_mappable = None
    field_label = None
    has_colormapped = False

    for ii, (sim_data, plot_data, final_fields) in enumerate(frame0_data):
        ax = axes_flat[ii]
        spec = panel_specs[ii]
        panel_props = resolve_per_file_props(component_props, panel_overrides, ii)
        nlvls, use_polygons = refinement_info(plot_data.fields, config)

        apply_scaling(ax, config.figure)

        components_for_panel = []
        for field_data in final_fields:
            component, props_key = create_scalar_component(
                field_data, panel_props, use_polygons,
                bodies=plot_data.body_collection,
            )
            if global_range and shared_colorbar:
                component.props = _override_color_range(
                    component.props, *global_range
                )
            component.initialize(fig, ax)
            component.render(field_data, config.figure)
            components_for_panel.append((component, field_data.name))

            if hasattr(component, "props") and props_key in ("polygon", "quad"):
                has_colormapped = True
                # grab mappable from the component's internal state
                mesh = getattr(component, "_main_mesh", None)
                if mesh is not None:
                    last_mappable = mesh

            if field_label is None:
                name = field_data.name
                if "_polygons" in name:
                    name = name.split("_polygons")[0]
                if name.endswith("_L"):
                    name = name.rsplit("_L", 1)[0]
                field_label = name

        panel_components.append(components_for_panel)
        _apply_panel_limits(ax, spec, config.figure)

        # panel label
        override_label = panel_overrides.get(ii, {}).get("label")
        if panel_labels and ii < len(panel_labels):
            label = panel_labels[ii]
        elif override_label:
            label = override_label
        else:
            src = spec["file"] or files[0]
            label = _extract_panel_label(src, sim_data, auto_label)
        _annotate_panel(ax, label, inside=annotate_inside)

    for jj in range(npanels, nrows * ncols):
        axes_flat[jj].set_visible(False)

    if has_colormapped:
        _add_colorbar(
            fig, axes_flat, npanels, nrows, ncols,
            last_mappable, field_label, shared_colorbar,
        )

    _format_grid_panels(
        axes_flat, npanels, nrows, ncols, config, has_colormapped, **kwargs
    )

    # time annotation
    time_text = None
    t0 = frame0_data[0][1].time if frame0_data else None
    if t0 is not None:
        time_scale = config.figure.time_scale
        time_units = config.figure.time_units
        t_display = t0 / time_scale if time_scale and time_scale > 0 else t0
        time_text = fig.suptitle(
            f"t = {t_display:.2f} {time_units}".strip(), fontsize=10
        )

    fig.canvas.draw()

    def _init():
        return []

    def _update(frame_idx: int):
        frame_data = _load_frame(files[frame_idx])
        for ii, (_, plot_data, final_fields) in enumerate(frame_data):
            for (component, _), field_data in zip(
                panel_components[ii], final_fields
            ):
                component.render(field_data, config.figure)
            _apply_panel_limits(axes_flat[ii], panel_specs[ii], config.figure)

        if time_text is not None and frame_data:
            t = frame_data[0][1].time
            if t is not None:
                time_scale = config.figure.time_scale
                time_units = config.figure.time_units
                t_display = (
                    t / time_scale if time_scale and time_scale > 0 else t
                )
                time_text.set_text(
                    f"t = {t_display:.2f} {time_units}".strip()
                )

        fig.canvas.draw_idle()
        return []

    anim = FuncAnimation(
        fig,
        _update,
        frames=nframes,
        init_func=_init,
        blit=False,
        interval=int(1000 / fps),
    )

    if save_as:
        import os

        from simbi.reader import logger as reader_logger
        from simbi.reader.progress import create_progress_bar

        base, ext = os.path.splitext(save_as)
        base = base.strip().replace(" ", "_")
        if ext not in (".mp4", ".avi", ".mov", ".gif"):
            ext = ".mp4"
        save_path = base + ext

        prog_bar = create_progress_bar()
        with prog_bar:
            task = prog_bar.add_task(
                f"[green]saving animation to {save_path}...",
                total=nframes,
            )

            def prog_callback(current: int, total: int) -> None:
                prog_bar.update(task, advance=1)

            anim.save(
                save_path,
                dpi=config.figure.dpi,
                progress_callback=prog_callback,
            )
        reader_logger.info(f"animation saved: {save_path}")

    if show:
        plt.show()

    return fig
