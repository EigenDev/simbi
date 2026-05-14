# =============================================================================
# api.py
#
# public api for the visualization system.
# each public function is a thin wrapper that wires data to SimFigure or
# directly to Figure for specialized analysis plots.
# shared dispatch logic lives in builder.py.
# =============================================================================
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .builder import (
    detect_projection,
    dispatch_overlay_components,
    dispatch_scalar_components,
    dispatch_vector_components,
    get_props,
    init_component,
)
from .components import (
    CoordinateProfileComponent,
    CoordinateProfileProps,
    LinePlotComponent,
    LinePlotProps,
    PowerSpectrumComponent,
    PowerSpectrumProps,
    TimeSeriesPlotComponent,
    TimeSeriesPlotProps,
)
from .components.interface import ComponentProps
from .config import OverlayConfig, VisualizationConfig
from .config_loader import ConfigDict, resolve_per_file_props
from .figure import Figure, prepare_figure
from .pipeline import create_plot_data, load_data
from .pipeline.coord_binning import create_coordinate_profile_data
from .pipeline.power_spectrum import (
    create_angular_spectrum_data,
    create_power_spectrum_data,
)
from .pipeline.phase_fold import create_phase_fold_data
from .pipeline.temporal_spectrum import create_temporal_spectrum_data
from .pipeline.time_series import create_time_series_data
from .pipeline.transforms import compose_fields_for_render
from .registry import refinement_info
from .types import CoordSystem

# ---------------------------------------------------------------------------
# internal helpers (api-specific)
# ---------------------------------------------------------------------------


def _save_and_show(figure: Figure, save_as: Optional[str], show: bool) -> None:
    """save and/or display the figure."""
    if save_as:
        figure.save(save_as)
    if show:
        plt.show()


def _apply_broken_axis(
    figure: Figure,
    config,
    gap_threshold: float = 1e3,
    height_ratio: tuple[float, float] = (3, 1),
    blackboard: bool = True,
) -> bool:
    """split axes into broken y-axis if data spans a huge dynamic range.

    detects clusters of y-values separated by more than gap_threshold.
    if found, replaces the single axes with two stacked panels and
    adds diagonal break marks. returns True if the break was applied.
    """
    ax = figure.axes["main"]
    lines = ax.get_lines()
    if len(lines) < 2:
        return False

    # collect all y-values from plotted lines
    all_y = []
    for line in lines:
        yd = np.asarray(line.get_ydata(), dtype=float)
        valid = yd[yd > 0]
        if len(valid) > 0:
            all_y.append(valid)

    if len(all_y) < 2:
        return False

    # find per-curve peak values and check for a large gap
    peaks = sorted([chunk.max() for chunk in all_y])
    max_gap_ratio = 0
    gap_idx = 0
    for ii in range(1, len(peaks)):
        ratio = peaks[ii] / peaks[ii - 1] if peaks[ii - 1] > 0 else 0
        if ratio > max_gap_ratio:
            max_gap_ratio = ratio
            gap_idx = ii

    if max_gap_ratio < gap_threshold:
        return False

    # split point: geometric mean of the two clusters
    lo_max = peaks[gap_idx - 1]
    hi_min = peaks[gap_idx]
    split = np.sqrt(lo_max * hi_min)

    # collect line data before clearing
    line_data = []
    for line in lines:
        line_data.append(
            {
                "x": line.get_xdata().copy(),
                "y": line.get_ydata().copy(),
                "color": line.get_color(),
                "lw": line.get_linewidth(),
                "ls": line.get_linestyle(),
                "label": line.get_label(),
                "alpha": line.get_alpha() or 1.0,
                "marker": line.get_marker(),
                "markevery": line.get_markevery(),
                "markersize": line.get_markersize(),
            }
        )

    # grab axis labels and formatting before replacing
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    title = ax.get_title()

    # determine y-limits for each panel
    lo_vals = np.concatenate(
        [chunk[chunk <= split] for chunk in all_y if np.any(chunk <= split)]
    )
    hi_vals = np.concatenate(
        [chunk[chunk > split] for chunk in all_y if np.any(chunk > split)]
    )

    lo_ymin = lo_vals.min() * 0.3 if len(lo_vals) > 0 else split * 0.01
    lo_ymax = lo_vals.max() * 3.0 if len(lo_vals) > 0 else split
    hi_ymin = hi_vals.min() * 0.3 if len(hi_vals) > 0 else split
    hi_ymax = hi_vals.max() * 10.0 if len(hi_vals) > 0 else split * 100

    # replace figure with two subplots
    fig = figure.fig
    fig.clear()
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=height_ratio,
        hspace=0.08,
    )
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    figure.axes["main"] = ax_top
    figure.axes["broken_bottom"] = ax_bot

    # plot data on both panels
    for ld in line_data:
        if ld["label"].startswith("_"):
            label_top = ld["label"]
            label_bot = ld["label"]
        else:
            label_top = ld["label"]
            label_bot = "_" + ld["label"]
        shared = dict(
            color=ld["color"],
            lw=ld["lw"],
            ls=ld["ls"],
            alpha=ld["alpha"],
            marker=ld["marker"],
            markevery=ld["markevery"],
            markersize=ld["markersize"],
        )
        ax_top.loglog(ld["x"], ld["y"], label=label_top, **shared)
        ax_bot.loglog(ld["x"], ld["y"], label=label_bot, **shared)

    # set independent y-limits
    ax_top.set_ylim(hi_ymin, hi_ymax)
    ax_bot.set_ylim(lo_ymin, lo_ymax)

    # x-limits: respect user overrides, fall back to data range
    if hasattr(config, "figure") and config.figure.xlims is not None:
        ax_top.set_xlim(config.figure.xlims.min, config.figure.xlims.max)
    else:
        all_x = np.concatenate([ld["x"] for ld in line_data])
        all_x = all_x[all_x > 0]
        ax_top.set_xlim(all_x.min(), all_x.max())

    # shared x-axis: hide top panel x ticks, keep bottom panel x ticks
    ax_top.tick_params(axis="x", which="both", labelbottom=False, bottom=False)

    if blackboard:
        # blackboard style: strip all tick labels and marks
        ax_top.tick_params(axis="y", which="both", labelleft=False, length=0)
        ax_bot.tick_params(axis="x", which="both", labelbottom=False, length=0)
        ax_bot.tick_params(axis="y", which="both", labelleft=False, length=0)

    # labels
    ax_bot.set_xlabel(xlabel)
    ax_top.set_ylabel(ylabel)
    if blackboard:
        ax_bot.set_ylabel("noise floor", fontsize=8, fontstyle="italic")
    ax_top.set_title(title)

    # legend on top panel only
    handles, lbls = ax_top.get_legend_handles_labels()
    if lbls:
        ax_top.legend(loc="best")

    # spines and break marks
    ax_top.spines["top"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)

    if blackboard:
        ax_top.spines["bottom"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)

        # diagonal break marks (left side only — right spines are hidden)
        size = 6
        for a, yc in [(ax_top, 0.0), (ax_bot, 1.0)]:
            trans = a.transAxes
            a.plot(
                [0],
                [yc],
                transform=trans,
                marker=[(-1, -1), (1, 1)],
                markersize=size,
                markeredgewidth=0.8,
                markeredgecolor="k",
                markerfacecolor="none",
                clip_on=False,
                linestyle="none",
            )

        # arrow-tipped axes
        color = ax_top.spines["left"].get_edgecolor()
        lw = ax_top.spines["left"].get_linewidth()
        ax_top.annotate(
            "",
            xy=(0, 1.02),
            xycoords="axes fraction",
            xytext=(0, 0.97),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
            annotation_clip=False,
        )
        color = ax_bot.spines["bottom"].get_edgecolor()
        lw = ax_bot.spines["bottom"].get_linewidth()
        ax_bot.annotate(
            "",
            xy=(1.02, 0),
            xycoords="axes fraction",
            xytext=(0.97, 0),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
            annotation_clip=False,
        )
    else:
        # standard broken-axis: hide shared boundary and right spines
        ax_top.spines["bottom"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)

        # diagonal break marks (left side only — right spines are hidden)
        size = 6
        for a, yc in [(ax_top, 0.0), (ax_bot, 1.0)]:
            trans = a.transAxes
            a.plot(
                [0],
                [yc],
                transform=trans,
                marker=[(-1, -1), (1, 1)],
                markersize=size,
                markeredgewidth=0.8,
                markeredgecolor="k",
                markerfacecolor="none",
                clip_on=False,
                linestyle="none",
            )

    return True


def _tighten_spectrum_axes(figure: Figure, config) -> None:
    """clamp axes tightly to the data range for power spectrum plots."""
    if config.figure.xlims is not None or config.figure.ylims is not None:
        return

    ax = figure.axes["main"]
    all_x = []
    all_y = []
    for comp, data, _ in figure._components:
        if isinstance(comp, PowerSpectrumComponent) and data is not None:
            all_x.append(data.domain[0])
            vals = data.values
            valid = vals[vals > 0]
            if len(valid) > 0:
                all_y.append(valid)

    if not all_x or not all_y:
        return

    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)
    ax.set_xlim(x_all.min(), x_all.max())
    ax.set_ylim(y_all.min() * 0.5, y_all.max() * 3.0)


def _setup_scalar_figure(config, files, fields, component_props, **kwargs):
    """load data, prepare figure, and attach scalar/vector/overlay components."""
    sim_data = load_data(files[0])
    scalar_plot_data = create_plot_data(sim_data, fields, config)
    final_fields = compose_fields_for_render(scalar_plot_data.fields, config)
    nlvls, use_polygons = refinement_info(scalar_plot_data.fields, config)
    projection = detect_projection(final_fields, sim_data.metadata.coord_system)

    figure = prepare_figure(
        config,
        len(files),
        projection=projection,
        nlvls=nlvls,
        coord_system=CoordSystem(sim_data.metadata.coord_system),
    )

    dispatch_scalar_components(
        figure,
        final_fields,
        component_props,
        use_polygons,
        bodies=scalar_plot_data.body_collection,
    )

    vector_fields = kwargs.get("vector_fields")
    if vector_fields:
        dispatch_vector_components(
            figure,
            sim_data,
            vector_fields,
            config,
            component_props,
            vector_type=kwargs.get("vector_type", "quiver"),
        )

    all_overlays = list(config.overlays)
    if kwargs.get("overlays"):
        all_overlays.extend(kwargs["overlays"])
    if all_overlays:
        dispatch_overlay_components(figure, sim_data, all_overlays, config)

    return figure


# ---------------------------------------------------------------------------
# public api
# ---------------------------------------------------------------------------


def plot(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    vector_fields: Optional[Sequence[str]] = None,
    overlays: Optional[Sequence[OverlayConfig]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create a visualization from checkpoint file(s)."""
    if isinstance(files, str):
        files = [files]

    figure = _setup_scalar_figure(
        config,
        files,
        fields,
        component_props,
        vector_fields=vector_fields,
        overlays=overlays,
        **kwargs,
    )
    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def animate(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    overlays: Optional[Sequence[OverlayConfig]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create animation from ordered sequence of checkpoint files."""
    if isinstance(files, str):
        files = [files]
    if len(files) < 2:
        raise ValueError("animation requires at least 2 files")

    figure = _setup_scalar_figure(
        config,
        files,
        fields,
        component_props,
        overlays=overlays,
        **kwargs,
    )

    fps = kwargs.get("fps") or config.animation.frame_rate
    figure.animate(files, fps=fps)
    _save_and_show(figure, save_as, show)
    return figure


def plot_coordinate_profile(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create coordinate-binned profile plot."""
    if isinstance(files, str):
        files = [files]

    sim_data = load_data(files[0])
    plot_data = create_coordinate_profile_data(sim_data, fields, config)
    if not plot_data.fields:
        raise ValueError("no coordinate profiles generated")

    figure = prepare_figure(config, len(files), projection="cartesian", nlvls=4)

    for field_data in plot_data.fields:
        props = get_props(
            component_props, "coordinate_profile", CoordinateProfileProps
        )
        init_component(figure, CoordinateProfileComponent(props), field_data)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def animate_coordinate_profile(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create animation of coordinate-binned profiles from multiple files."""
    if isinstance(files, str):
        files = [files]
    if len(files) < 2:
        raise ValueError("animation requires at least 2 files")

    sim_data = load_data(files[0])
    plot_data = create_coordinate_profile_data(sim_data, fields, config)
    if not plot_data.fields:
        raise ValueError("no coordinate profiles generated")

    figure = prepare_figure(config, len(files), projection="cartesian", nlvls=4)

    for field_data in plot_data.fields:
        props = get_props(
            component_props, "coordinate_profile", CoordinateProfileProps
        )
        init_component(figure, CoordinateProfileComponent(props), field_data)

    fps = kwargs.get("fps") or config.animation.frame_rate
    figure.animate_coordinate_profile(files, fields, config, fps=fps)
    _save_and_show(figure, save_as, show)
    return figure


def plot_time_series(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["rho"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create time series plot from multiple checkpoint files."""
    if isinstance(files, str):
        files = [files]

    plot_data = create_time_series_data(files, fields, config)
    if not plot_data.fields:
        raise ValueError("no time series data generated")

    nlines = plot_data.count_plot_lines()
    figure = prepare_figure(config, nlvls=nlines)

    for field_data in plot_data.fields:
        props = get_props(component_props, "time_series", TimeSeriesPlotProps)
        init_component(figure, TimeSeriesPlotComponent(props), field_data)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def plot_temporal_spectrum(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["mdot"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create temporal power spectrum from a sequence of checkpoint files."""
    if isinstance(files, str):
        files = [files]

    plot_data = create_temporal_spectrum_data(files, fields, config)
    if not plot_data.fields:
        raise ValueError("no temporal spectrum data generated")

    # extract pipeline metadata for smart props
    meta = plot_data.extra or {}
    binary_params = meta.get("binary_params")
    orbital_period = meta.get("orbital_period")
    n_samples = meta.get("n_samples", 0)
    n_freqs = meta.get("n_freqs", 1024)
    omega_nyquist = meta.get("omega_nyquist")

    nlines = len(plot_data.fields)
    figure = prepare_figure(
        config, len(files), projection="cartesian", nlvls=nlines
    )

    _linestyles = ("-", "--", "-.", ":")
    for ii, field_data in enumerate(plot_data.fields):
        base_props = get_props(
            component_props, "power_spectrum", PowerSpectrumProps
        )

        # auto-populate reference frequencies for binary systems
        # only when frequency axis is normalized (orbital_period is set)
        # filter out any harmonics above the nyquist limit
        ref_freqs = base_props.reference_frequencies
        ref_labels = base_props.reference_frequency_labels
        if (
            not ref_freqs
            and binary_params is not None
            and orbital_period is not None
            and orbital_period > 0
        ):
            all_freqs = (1.0, 2.0, 4.0)
            all_labels = (r"$\Omega$", r"$2\Omega$", r"$4\Omega$")
            if omega_nyquist is not None:
                pairs = [
                    (f, l)
                    for f, l in zip(all_freqs, all_labels)
                    if f < omega_nyquist
                ]
                ref_freqs = tuple(f for f, _ in pairs)
                ref_labels = tuple(l for _, l in pairs)
            else:
                ref_freqs = all_freqs
                ref_labels = all_labels

        # auto-populate FAP params from pipeline metadata
        fap_n = base_props.fap_n_samples or n_samples
        fap_norm = 2.0 / n_samples if n_samples > 0 else 1.0

        # per-body lines (no body_names) get reduced alpha so the total stands out
        is_total = bool(field_data.body_names)
        line_alpha = base_props.alpha if is_total else base_props.alpha * 0.25

        props = PowerSpectrumProps(
            show_reference_slopes=base_props.show_reference_slopes,
            reference_slopes=base_props.reference_slopes,
            compensated=base_props.compensated,
            arbitrary_units=base_props.arbitrary_units,
            linewidth=base_props.linewidth,
            linestyle="-" if is_total else _linestyles[ii % len(_linestyles)],
            color=base_props.color,
            label="_nolegend_" if not is_total else base_props.label,
            alpha=line_alpha,
            reference_frequencies=ref_freqs,
            reference_frequency_labels=ref_labels,
            show_smoothed=base_props.show_smoothed,
            smooth_window=base_props.smooth_window,
            smooth_polyorder=base_props.smooth_polyorder,
            show_fap_levels=base_props.show_fap_levels,
            fap_levels=base_props.fap_levels,
            fap_n_samples=fap_n,
            fap_psd_normalization=fap_norm,
            show_xlabel=base_props.show_xlabel,
        )
        init_component(figure, PowerSpectrumComponent(props), field_data)

    figure.render()

    # clamp x-axis to nyquist limit (after render so it's the final word)
    if omega_nyquist is not None and config.figure.xlims is None:
        figure.axes["main"].set_xlim(right=omega_nyquist)

    _save_and_show(figure, save_as, show)
    return figure


def plot_phase_fold(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["mdot"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create phase-folded time series from a sequence of checkpoint files."""
    if isinstance(files, str):
        files = [files]

    plot_data = create_phase_fold_data(files, fields, config)
    if not plot_data.fields:
        raise ValueError("no phase-fold data generated")

    nlines = len(plot_data.fields)
    figure = prepare_figure(config, nlvls=nlines)

    for field_data in plot_data.fields:
        base_props = get_props(
            component_props, "time_series", TimeSeriesPlotProps
        )

        # orbit traces get reduced alpha
        is_trace = field_data.name == "_orbit_traces"
        if is_trace:
            props = TimeSeriesPlotProps(
                alpha=0.15,
                linewidth=0.5,
                label="_nolegend_",
            )
        else:
            props = base_props

        init_component(figure, TimeSeriesPlotComponent(props), field_data)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def plot_power_spectrum(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["v1", "v2", "v3"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create power spectrum plot from a checkpoint file."""
    if isinstance(files, str):
        files = [files]

    sim_data = load_data(files[0])
    base_props = get_props(
        component_props, "power_spectrum", PowerSpectrumProps
    )
    plot_data = create_power_spectrum_data(
        sim_data, config, fields,
        subtract_radial_mean=base_props.subtract_radial_mean,
        use_composite=base_props.use_composite,
    )
    if not plot_data.fields:
        raise ValueError("no power spectrum data generated")

    figure = prepare_figure(config, len(files), projection="cartesian", nlvls=1)

    for field_data in plot_data.fields:
        props = PowerSpectrumProps(
            show_reference_slopes=base_props.show_reference_slopes,
            reference_slopes=base_props.reference_slopes,
            compensated=base_props.compensated,
            arbitrary_units=base_props.arbitrary_units,
            linewidth=base_props.linewidth,
            linestyle=base_props.linestyle,
            color=base_props.color,
            label=base_props.label,
            # marker=base_props.marker,
            # mark_every=base_props.mark_every,
        )
        init_component(figure, PowerSpectrumComponent(props), field_data)

    figure.render()
    _tighten_spectrum_axes(figure, config)
    _save_and_show(figure, save_as, show)
    return figure


def plot_power_spectrum_overlay(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ["v1", "v2", "v3"],
    labels: Optional[Sequence[str]] = None,
    linestyles: Optional[Sequence[str]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    per_file_overrides: Optional[dict[int, ConfigDict]] = None,
    **kwargs,
) -> Figure:
    """overlay power spectra from multiple checkpoint files on the same axes."""
    if len(files) < 2:
        raise ValueError("overlay requires at least 2 files")

    nfiles = len(files)

    figure = prepare_figure(
        config,
        nfiles=nfiles,
        projection="cartesian",
        nlvls=nfiles,
        overlay_mode=True,
    )

    global_props = get_props(
        component_props, "power_spectrum", PowerSpectrumProps
    )

    for ii, file_path in enumerate(files):
        sim_data = load_data(file_path)
        plot_data = create_power_spectrum_data(
            sim_data, config, fields,
            subtract_radial_mean=global_props.subtract_radial_mean,
            use_composite=global_props.use_composite,
        )
        if not plot_data.fields:
            continue

        file_props = resolve_per_file_props(
            component_props, per_file_overrides, ii
        )
        label = labels[ii] if labels and ii < len(labels) else None
        ls = linestyles[ii] if linestyles and ii < len(linestyles) else None

        for field_data in plot_data.fields:
            base_props = get_props(
                file_props, "power_spectrum", PowerSpectrumProps
            )
            props = PowerSpectrumProps(
                label=label or base_props.label,
                linewidth=base_props.linewidth,
                linestyle=ls or base_props.linestyle,
                color=base_props.color,
                # marker=base_props.marker,
                # mark_every=base_props.mark_every,
                compensated=base_props.compensated,
                arbitrary_units=base_props.arbitrary_units,
                # defer slopes to post-render so they see all data
                show_reference_slopes=False,
                reference_slopes=base_props.reference_slopes,
                show_smoothed=base_props.show_smoothed,
                smooth_window=base_props.smooth_window,
                smooth_polyorder=base_props.smooth_polyorder,
            )
            init_component(figure, PowerSpectrumComponent(props), field_data)

    figure.render()
    _tighten_spectrum_axes(figure, config)

    # auto-detect large dynamic range and apply broken y-axis
    use_blackboard = get_props(
        component_props, "power_spectrum", PowerSpectrumProps
    ).arbitrary_units
    broken = _apply_broken_axis(figure, config, blackboard=use_blackboard)

    # draw reference slopes on the top panel after all data is visible
    slope_ax = figure.axes["main"]
    first_comp = None
    first_data = None
    for comp, data, _ in figure._components:
        if isinstance(comp, PowerSpectrumComponent):
            base = get_props(
                component_props, "power_spectrum", PowerSpectrumProps
            )
            if base.show_reference_slopes:
                first_comp = comp
                first_data = data
                break
    if first_comp is not None and first_data is not None:
        first_comp.ax = slope_ax
        first_comp._draw_reference_slopes(
            first_data.domain[0], first_data.values
        )

    # draw reference frequency lines on both panels (post broken-axis)
    ref_props = get_props(
        component_props, "power_spectrum", PowerSpectrumProps
    )
    if ref_props.reference_frequencies:
        bot_ax = figure.axes.get("broken_bottom")
        for ii, freq in enumerate(ref_props.reference_frequencies):
            slope_ax.axvline(
                freq, color="grey", linestyle=":", linewidth=0.8, alpha=0.6,
            )
            if bot_ax is not None:
                bot_ax.axvline(
                    freq, color="grey", linestyle=":", linewidth=0.8, alpha=0.6,
                )
            if ii < len(ref_props.reference_frequency_labels):
                slope_ax.annotate(
                    ref_props.reference_frequency_labels[ii],
                    xy=(freq, 1.0),
                    xycoords=("data", "axes fraction"),
                    color="grey", alpha=0.8,
                    ha="center", va="bottom",
                    xytext=(0, 2), textcoords="offset points",
                )

    _save_and_show(figure, save_as, show)
    return figure


def plot_angular_spectrum(
    config: VisualizationConfig,
    files: str | Sequence[str],
    fields: Sequence[str] = ["entropy-measure"],
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    **kwargs,
) -> Figure:
    """create angular power spectrum C_l from a checkpoint file."""
    if isinstance(files, str):
        files = [files]

    sim_data = load_data(files[0])
    base_props = get_props(
        component_props, "power_spectrum", PowerSpectrumProps
    )

    radii = list(base_props.angular_radii) if base_props.angular_radii else None
    is_vector = len(fields) >= 3
    plot_data = create_angular_spectrum_data(
        sim_data, config,
        field=fields[0],
        fields=fields if is_vector else None,
        radii=radii,
        n_shells=base_props.angular_n_shells,
        n_theta=base_props.angular_n_theta,
        n_phi=base_props.angular_n_phi,
        subtract_mean=base_props.subtract_radial_mean,
    )
    if not plot_data.fields:
        raise ValueError("no angular spectrum data generated")

    figure = prepare_figure(config, len(files), projection="cartesian", nlvls=1)

    for field_data in plot_data.fields:
        props = PowerSpectrumProps(
            show_reference_slopes=False,
            linewidth=base_props.linewidth,
            linestyle=base_props.linestyle,
            color=base_props.color,
            label=base_props.label,
        )
        init_component(figure, PowerSpectrumComponent(props), field_data)

    figure.render()
    _tighten_spectrum_axes(figure, config)
    _save_and_show(figure, save_as, show)
    return figure


def plot_angular_spectrum_overlay(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ["entropy-measure"],
    labels: Optional[Sequence[str]] = None,
    linestyles: Optional[Sequence[str]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    per_file_overrides: Optional[dict[int, ConfigDict]] = None,
    **kwargs,
) -> Figure:
    """overlay angular power spectra from multiple checkpoint files."""
    if len(files) < 2:
        raise ValueError("overlay requires at least 2 files")

    nfiles = len(files)
    figure = prepare_figure(
        config, nfiles=nfiles, projection="cartesian",
        nlvls=nfiles, overlay_mode=True,
    )

    global_props = get_props(
        component_props, "power_spectrum", PowerSpectrumProps
    )

    is_vector = len(fields) >= 3

    for ii, file_path in enumerate(files):
        sim_data = load_data(file_path)
        radii = list(global_props.angular_radii) if global_props.angular_radii else None
        plot_data = create_angular_spectrum_data(
            sim_data, config,
            field=fields[0],
            fields=fields if is_vector else None,
            radii=radii,
            n_shells=global_props.angular_n_shells,
            n_theta=global_props.angular_n_theta,
            n_phi=global_props.angular_n_phi,
            subtract_mean=global_props.subtract_radial_mean,
        )
        if not plot_data.fields:
            continue

        file_props = resolve_per_file_props(
            component_props, per_file_overrides, ii
        )
        label = labels[ii] if labels and ii < len(labels) else None
        ls = linestyles[ii] if linestyles and ii < len(linestyles) else None

        for field_data in plot_data.fields:
            base_props = get_props(
                file_props, "power_spectrum", PowerSpectrumProps
            )
            props = PowerSpectrumProps(
                label=label or base_props.label,
                linewidth=base_props.linewidth,
                linestyle=ls or base_props.linestyle,
                color=base_props.color,
                show_reference_slopes=False,
                reference_slopes=base_props.reference_slopes,
            )
            init_component(figure, PowerSpectrumComponent(props), field_data)

    figure.render()
    _tighten_spectrum_axes(figure, config)

    # auto-detect large dynamic range and apply broken y-axis
    use_blackboard = global_props.arbitrary_units
    broken = _apply_broken_axis(figure, config, blackboard=use_blackboard)

    # draw reference slopes on the top panel if user explicitly requested them
    if global_props.show_reference_slopes:
        slope_ax = figure.axes["main"]
        first_comp = None
        first_data = None
        for comp, data_item, _ in figure._components:
            if isinstance(comp, PowerSpectrumComponent):
                first_comp = comp
                first_data = data_item
                break
        if first_comp is not None and first_data is not None:
            first_comp.ax = slope_ax
            first_comp.props = PowerSpectrumProps(
                **{
                    **first_comp.props.model_dump(),
                    "show_reference_slopes": True,
                    "reference_slopes": global_props.reference_slopes,
                    "reference_frequencies": global_props.reference_frequencies,
                }
            )
            first_comp._draw_reference_slopes(
                first_data.domain[0], first_data.values
            )

    # draw reference frequency lines on both panels (post broken-axis)
    slope_ax = figure.axes["main"]
    ref_props = global_props
    if ref_props.reference_frequencies:
        bot_ax = figure.axes.get("broken_bottom")
        for ii, freq in enumerate(ref_props.reference_frequencies):
            slope_ax.axvline(
                freq, color="grey", linestyle=":", linewidth=0.8, alpha=0.6,
            )
            if bot_ax is not None:
                bot_ax.axvline(
                    freq, color="grey", linestyle=":", linewidth=0.8, alpha=0.6,
                )
            if ii < len(ref_props.reference_frequency_labels):
                slope_ax.annotate(
                    ref_props.reference_frequency_labels[ii],
                    xy=(freq, 1.0),
                    xycoords=("data", "axes fraction"),
                    color="grey", alpha=0.8,
                    ha="center", va="bottom",
                    xytext=(0, 2), textcoords="offset points",
                )

    _save_and_show(figure, save_as, show)
    return figure


def plot_overlay(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ["rho"],
    normalizations: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    per_file_overrides: Optional[dict[int, ConfigDict]] = None,
    **kwargs,
) -> Figure:
    """overlay multiple files on the same axes (line plots only)."""
    if len(files) < 2:
        raise ValueError("overlay requires at least 2 files")

    nfiles = len(files)

    # validate first file is 1d
    first_data = load_data(files[0])
    first_plot_data = create_plot_data(first_data, fields, config)
    for f in first_plot_data.fields:
        if f.ndim != 1:
            raise ValueError(
                f"overlay only supports 1D data, got ndim={f.ndim} for '{f.name}'. "
                "use --slice to reduce or try a different plot type."
            )

    figure = prepare_figure(
        config,
        nfiles=nfiles,
        projection="cartesian",
        nlvls=nfiles,
        coord_system=CoordSystem(first_data.metadata.coord_system),
        overlay_mode=True,
    )

    for ii, file_path in enumerate(files):
        sim_data = load_data(file_path)
        plot_data = create_plot_data(sim_data, fields, config)

        file_props = resolve_per_file_props(
            component_props, per_file_overrides, ii
        )
        label = labels[ii] if labels and ii < len(labels) else None

        for field_data in plot_data.fields:
            if field_data.ndim != 1:
                continue
            base_props = get_props(file_props, "line", LinePlotProps)
            props = LinePlotProps(
                label=label or f"{field_data.name}",
                linewidth=base_props.linewidth,
                marker=base_props.marker,
                marker_size=base_props.marker_size,
                alpha=base_props.alpha,
            )
            init_component(figure, LinePlotComponent(props), field_data)

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def _build_profile_props(
    base_props: CoordinateProfileProps,
    sim_data,
    field_data,
    label: Optional[str],
    file_label: str,
    norm: Optional[float],
    x_norm: float,
    x_normalizations: Optional[Sequence[float]],
) -> CoordinateProfileProps:
    """assemble per-file per-field coordinate profile props."""
    bondi_gamma = base_props.bondi_gamma or sim_data.metadata.gamma
    bondi_mass = base_props.bondi_total_mass
    if (
        bondi_mass == 1.0
        and sim_data.body_collection
        and sim_data.body_collection.binary_params
    ):
        bondi_mass = sim_data.body_collection.binary_params.get(
            "total_mass", 1.0
        )

    return CoordinateProfileProps(
        label=label
        or base_props.label
        or f"{field_data.name} ({file_label})",
        color=base_props.color,
        linestyle=base_props.linestyle,
        linewidth=base_props.linewidth,
        normalization=norm or base_props.normalization,
        x_normalization=x_norm
        if x_normalizations
        else base_props.x_normalization or x_norm,
        rend=base_props.rend,
        show_reference_lines=base_props.show_reference_lines,
        reference_fields=base_props.reference_fields,
        show_bondi=base_props.show_bondi,
        bondi_gamma=bondi_gamma,
        bondi_rho_inf=base_props.bondi_rho_inf,
        bondi_cs_inf=base_props.bondi_cs_inf,
        bondi_total_mass=bondi_mass,
    )


def plot_coordinate_profile_overlay(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str] = ["rho"],
    normalizations: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    x_normalizations: Optional[Sequence[float]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    per_file_overrides: Optional[dict[int, ConfigDict]] = None,
    **kwargs,
) -> Figure:
    """overlay coordinate profiles from multiple files on the same axes.

    when layout is provided and there are multiple fields, creates a
    multi-panel figure with one field per panel, all files overlaid on each.
    """
    if len(files) < 2:
        raise ValueError("overlay requires at least 2 files")

    field_grid = kwargs.get("field_grid")

    # multi-panel mode: one panel per field, files overlaid on each
    if field_grid and len(fields) > 1:
        return _plot_coordinate_profile_grid(
            config, files, fields, field_grid,
            normalizations=normalizations,
            labels=labels,
            x_normalizations=x_normalizations,
            save_as=save_as,
            show=show,
            component_props=component_props,
            per_file_overrides=per_file_overrides,
            **{k: v for k, v in kwargs.items() if k != "field_grid"},
        )

    # single-axes mode (original behavior)
    nfiles = len(files)
    figure = prepare_figure(
        config,
        nfiles=nfiles,
        projection="cartesian",
        nlvls=nfiles,
        overlay_mode=True,
    )

    for ii, file_path in enumerate(files):
        sim_data = load_data(file_path)
        plot_data = create_coordinate_profile_data(sim_data, fields, config)
        if not plot_data.fields:
            continue

        file_props = resolve_per_file_props(
            component_props, per_file_overrides, ii
        )
        file_label = Path(file_path).stem
        norm = (
            normalizations[ii]
            if normalizations and ii < len(normalizations)
            else None
        )
        label = labels[ii] if labels and ii < len(labels) else None
        if x_normalizations and ii < len(x_normalizations):
            x_norm = x_normalizations[ii]
        else:
            x_norm = float(np.nanmax(plot_data.fields[0].domain[0]))

        for field_data in plot_data.fields:
            base_props = get_props(
                file_props, "coordinate_profile", CoordinateProfileProps,
            )
            props = _build_profile_props(
                base_props, sim_data, field_data,
                label, file_label, norm, x_norm, x_normalizations,
            )
            init_component(
                figure, CoordinateProfileComponent(props), field_data
            )

    figure.render()
    _save_and_show(figure, save_as, show)
    return figure


def _plot_coordinate_profile_grid(
    config: VisualizationConfig,
    files: Sequence[str],
    fields: Sequence[str],
    layout: tuple[int, int],
    normalizations: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    x_normalizations: Optional[Sequence[float]] = None,
    save_as: Optional[str] = None,
    show: bool = True,
    component_props: Optional[dict[str, ComponentProps]] = None,
    per_file_overrides: Optional[dict[int, ConfigDict]] = None,
    **kwargs,
) -> Figure:
    """multi-panel profile grid: one panel per field, files overlaid on each."""
    from .components.coord_binning import stripped_field_name

    nrows, ncols = layout
    nfields = len(fields)
    nfiles = len(files)
    wspace = kwargs.get("wspace", 0.3)
    hspace = kwargs.get("hspace", 0.3)

    config.theme.apply(nfiles=nfiles, nfields=nfields, overlay_mode=True)

    base_w, base_h = config.figure.fig_size
    fig_w = base_w * min(ncols, 3) / 1.5
    fig_h = base_h * min(nrows, 3) / 1.5
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": wspace, "hspace": hspace},
    )
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        axes_flat = list(np.atleast_1d(axes))
    else:
        axes_flat = list(axes.flatten())

    # hide unused panels
    for jj in range(nfields, nrows * ncols):
        axes_flat[jj].set_visible(False)

    # preload all file data
    file_data = []
    for ii, file_path in enumerate(files):
        sim_data = load_data(file_path)
        plot_data = create_coordinate_profile_data(sim_data, fields, config)
        file_data.append((sim_data, plot_data))

    # get the color cycle so each file has a consistent color across panels
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ff, field_name in enumerate(fields):
        ax = axes_flat[ff]
        _, field_str = stripped_field_name(f"{field_name}_vs_r")

        for ii, (sim_data, plot_data) in enumerate(file_data):
            # find the field_data matching this field
            target_name = f"{field_name}_vs_r"
            field_data = None
            for fd in plot_data.fields:
                if fd.name == target_name:
                    field_data = fd
                    break
            if field_data is None:
                continue

            file_props = resolve_per_file_props(
                component_props, per_file_overrides, ii
            )
            file_label = Path(files[ii]).stem
            norm = (
                normalizations[ii]
                if normalizations and ii < len(normalizations)
                else None
            )
            label = labels[ii] if labels and ii < len(labels) else None
            if x_normalizations and ii < len(x_normalizations):
                x_norm = x_normalizations[ii]
            else:
                x_norm = float(np.nanmax(plot_data.fields[0].domain[0]))

            base_props = get_props(
                file_props, "coordinate_profile", CoordinateProfileProps,
            )
            props = _build_profile_props(
                base_props, sim_data, field_data,
                label, file_label, norm, x_norm, x_normalizations,
            )

            # render directly onto this panel's axes
            if not props.color:
                from matplotlib.colors import to_hex

                props = CoordinateProfileProps(
                    **{
                        **props.model_dump(),
                        "color": to_hex(color_cycle[ii % len(color_cycle)]),
                    }
                )
            comp = CoordinateProfileComponent(props)
            comp.initialize(fig, ax)
            comp.render(field_data, config.figure)

        ax.set_ylabel(field_str)
        if ff < nfields - ncols:
            ax.set_xlabel("")

        # legend only on first panel
        if ff == 0:
            handles, leg_labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best")

    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    if save_as:
        fig.savefig(
            save_as,
            dpi=config.figure.dpi,
            bbox_inches="tight",
            transparent=config.figure.transparent,
        )
    if show:
        plt.show()

    # return a dummy Figure for api consistency
    figure = prepare_figure(config, nfiles=nfiles, projection="cartesian")
    figure.fig = fig
    return figure


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
):
    """create a multi-panel grid figure with optional per-panel views."""
    from .grid import plot_grid as _plot_grid

    return _plot_grid(
        config,
        files,
        fields,
        layout=layout,
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
):
    """animate a multi-panel grid across a checkpoint sequence."""
    from .grid import animate_grid as _animate_grid

    return _animate_grid(
        config,
        files,
        fields,
        layout=layout,
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
