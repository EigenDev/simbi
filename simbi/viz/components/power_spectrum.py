# =============================================================================
# power_spectrum.py
#
# component for rendering power spectra as log-log line plots.
# handles both spatial E(k) and temporal PSD with:
# - optional reference slope overlays (spatial spectra)
# - optional reference frequency annotations (temporal PSD)
# - optional savitzky-golay smoothed envelope
# - optional false-alarm probability threshold lines
#
# usage:
#   component = PowerSpectrumComponent(PowerSpectrumProps())
#   component.initialize(fig, ax)
#   result = component.render(field_data, style)
# =============================================================================
from typing import List, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..config import FigureConfig
from ..types import FieldData, RenderResult
from .interface import Component, ComponentProps


class PowerSpectrumProps(ComponentProps):
    """properties for power spectrum component."""

    show_reference_slopes: bool = True
    reference_slopes: tuple[float, ...] = (-2.0, -5.0 / 3.0)
    compensated: bool = False
    linewidth: float = 2.0
    color: Optional[str] = None
    label: Optional[str] = None

    # reference frequency annotations (vertical lines at known frequencies)
    reference_frequencies: tuple[float, ...] = ()
    reference_frequency_labels: tuple[str, ...] = ()

    # smoothed envelope overlay
    show_smoothed: bool = False
    smooth_window: int = 51
    smooth_polyorder: int = 3

    # blackboard-style axes: arrows, no tick labels
    arbitrary_units: bool = False

    # false-alarm probability levels
    show_fap_levels: bool = False
    fap_levels: tuple[float, ...] = (0.01, 0.001)
    fap_n_samples: int = 0
    fap_psd_normalization: float = 1.0


_SLOPE_LABELS = {
    -5.0 / 3.0: r"$k^{-5/3}$",
    -2.0: r"$k^{-2}$",
    -3.0: r"$k^{-3}$",
    -4.0: r"$k^{-4}$",
}


class PowerSpectrumComponent(Component[PowerSpectrumProps, FieldData]):
    """renders power spectrum as a log-log line plot."""

    def __init__(self, props: PowerSpectrumProps):
        self.props = props
        self._main_line: Optional[Line2D] = None
        self._smooth_line: Optional[Line2D] = None
        self._ref_lines: List[Line2D] = []
        self._ref_annotations: List = []
        self._freq_lines: List = []
        self._fap_lines: List[Line2D] = []
        self._initialized: bool = False

    def initialize(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax = ax
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, props: PowerSpectrumProps) -> None:
        self.props = props

    def render(self, data: FieldData, style: FigureConfig) -> RenderResult:
        if not self._initialized:
            raise RuntimeError("Component not initialized.")

        if data.ndim != 1:
            raise ValueError("PowerSpectrumComponent expects 1D FieldData.")

        k = data.domain[0]
        e_k = data.values

        # detect temporal PSD by frequency-like axis names (omega, Omega)
        has_axis_names = data.axis_names and len(data.axis_names) >= 1
        is_temporal = has_axis_names and "omega" in data.axis_names[0].lower()
        default_xlabel = data.axis_names[0] if has_axis_names else r"$k$"
        default_ylabel = data.name if has_axis_names else r"$E(k)$"

        if self.props.compensated and not is_temporal:
            y = k ** (5.0 / 3.0) * e_k
            ylabel = r"$k^{5/3}\,E(k)$"
        else:
            y = e_k
            ylabel = default_ylabel

        # prefer body name for legend (multi-body PSD), then props, then ylabel
        body_label = data.body_names[0] if data.body_names else None
        label = self.props.label or body_label or ylabel

        # when smoothing is on, raw data gets reduced alpha
        raw_alpha = 0.3 if self.props.show_smoothed else 1.0

        line_kwargs = {
            "linewidth": self.props.linewidth,
            "label": label,
            "alpha": raw_alpha,
        }
        if self.props.color:
            line_kwargs["color"] = self.props.color

        # temporal PSD: linear x, log y. spatial E(k): log-log.
        self._is_temporal = is_temporal
        plot_fn = self.ax.semilogy if is_temporal else self.ax.loglog

        if self._main_line is None:
            self._main_line = plot_fn(k, y, **line_kwargs)[0]
        else:
            self._main_line.set_data(k, y)
            self._main_line.set_alpha(raw_alpha)

        # smoothed envelope
        if self.props.show_smoothed:
            self._draw_smoothed(k, y, label)

        # reference slopes only for spatial spectra (not temporal PSD)
        if (
            self.props.show_reference_slopes
            and not self.props.compensated
            and not is_temporal
        ):
            self._draw_reference_slopes(k, e_k)

        # reference frequency vertical lines (temporal PSD)
        if self.props.reference_frequencies:
            self._draw_reference_frequencies()

        # FAP threshold lines
        if self.props.show_fap_levels and self.props.fap_n_samples > 0:
            self._draw_fap_levels(k)

        if self.ax.get_legend_handles_labels()[1]:
            self.ax.legend(loc="best")

        self.ax.set_xlabel(default_xlabel)
        self.ax.set_ylabel(ylabel)

        if self.props.arbitrary_units:
            self._apply_blackboard_axes()

        return RenderResult(
            artists={"line": self._main_line, "refs": self._ref_lines},
            metadata={"label": label, "is_line": True},
        )

    def _draw_smoothed(self, k: np.ndarray, y: np.ndarray, label: str) -> None:
        """overlay savitzky-golay smoothed curve in log-space."""
        from scipy.signal import savgol_filter

        # clean data for log-space filtering
        valid = y > 0
        if np.sum(valid) < self.props.smooth_window:
            return

        log_y = np.full_like(y, dtype=float, fill_value=np.nan)
        log_y[valid] = np.log10(y[valid])

        # interpolate gaps for smooth filtering
        if np.any(~valid):
            nans = np.isnan(log_y)
            log_y[nans] = np.interp(
                np.flatnonzero(nans), np.flatnonzero(~nans), log_y[~nans]
            )

        window = min(self.props.smooth_window, len(log_y))
        if window % 2 == 0:
            window -= 1
        if window < self.props.smooth_polyorder + 2:
            return

        smoothed = 10.0 ** savgol_filter(
            log_y, window, self.props.smooth_polyorder
        )

        color = self._main_line.get_color() if self._main_line else None

        plot_fn = self.ax.semilogy if self._is_temporal else self.ax.loglog

        if self._smooth_line is None:
            self._smooth_line = plot_fn(
                k,
                smoothed,
                color=color,
                linewidth=self.props.linewidth * 1.5,
                alpha=0.9,
                label=f"{label} (smoothed)",
            )[0]
        else:
            self._smooth_line.set_data(k, smoothed)

    def _draw_reference_frequencies(self) -> None:
        """draw vertical lines at known frequencies (e.g., orbital harmonics)."""
        for item in self._freq_lines:
            item.remove()
        self._freq_lines = []

        colors = ["grey", "grey", "grey", "grey"]
        for ii, freq in enumerate(self.props.reference_frequencies):
            label = (
                self.props.reference_frequency_labels[ii]
                if ii < len(self.props.reference_frequency_labels)
                else None
            )
            vline = self.ax.axvline(
                freq,
                color=colors[ii % len(colors)],
                linestyle=":",
                linewidth=0.8,
                alpha=0.6,
                label=label,
            )
            self._freq_lines.append(vline)

    def _draw_fap_levels(self, k: np.ndarray) -> None:
        """draw horizontal lines at false-alarm probability thresholds."""
        from simbi.analysis import lomb_scargle_fap_levels

        for line in self._fap_lines:
            if line in self.ax.lines:
                line.remove()
        self._fap_lines = []

        thresholds = lomb_scargle_fap_levels(
            self.props.fap_n_samples,
            len(k),
            levels=self.props.fap_levels,
            psd_normalization=self.props.fap_psd_normalization,
        )

        for fap, threshold in thresholds.items():
            pct = fap * 100
            if pct >= 1:
                label = f"{pct:.0f}% FAP"
            else:
                label = f"{pct:.1f}% FAP"
            fap_line = self.ax.axhline(
                threshold,
                color="red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                label=label,
            )
            self._fap_lines.append(fap_line)

    def _apply_blackboard_axes(self) -> None:
        """strip tick labels, add arrow tips to axes."""
        from matplotlib.ticker import NullFormatter

        # NullFormatter persists through draw cycles (set_xticklabels does not)
        self.ax.xaxis.set_major_formatter(NullFormatter())
        self.ax.xaxis.set_minor_formatter(NullFormatter())
        self.ax.yaxis.set_major_formatter(NullFormatter())
        self.ax.yaxis.set_minor_formatter(NullFormatter())
        self.ax.tick_params(axis="both", which="both", length=0)

        # hide the default box spines, keep only bottom and left
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        # arrow tips at the end of each visible spine
        for spine, direction in [("bottom", "right"), ("left", "up")]:
            self.ax.spines[spine].set_visible(True)
            color = self.ax.spines[spine].get_edgecolor()
            lw = self.ax.spines[spine].get_linewidth()
            if direction == "right":
                xy, xytext = (1.02, 0), (0.97, 0)
            else:
                xy, xytext = (0, 1.02), (0, 0.97)
            self.ax.annotate(
                "",
                xy=xy,
                xycoords="axes fraction",
                xytext=xytext,
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                annotation_clip=False,
            )

    def _draw_reference_slopes(self, k: np.ndarray, e_k: np.ndarray) -> None:
        """overlay short reference power-law segments in the inertial range."""
        for line in self._ref_lines:
            if line in self.ax.lines:
                line.remove()
        self._ref_lines = []
        for ann in self._ref_annotations:
            ann.remove()
        self._ref_annotations = []

        n = len(k)
        if n < 10:
            return

        valid = e_k > 0
        if np.sum(valid) < 4:
            return

        # center the segment in log-space, spanning ~1 decade
        k_pos = k[valid]
        log_k = np.log10(k_pos)
        log_center = 0.5 * (log_k[0] + log_k[-1])
        log_half_span = 0.5  # half a decade each side
        k_lo = 10.0 ** (log_center - log_half_span)
        k_hi = 10.0 ** (log_center + log_half_span)
        mask = (k >= k_lo) & (k <= k_hi)
        k_seg = k[mask]
        if len(k_seg) < 4:
            return

        # anchor at the geometric center, placed above the data
        i_mid = np.searchsorted(k, 10.0**log_center)
        i_mid = min(i_mid, n - 1)
        k_anchor = k[i_mid]
        e_anchor = e_k[i_mid]
        if e_anchor <= 0:
            return

        # place slopes above the data, stagger so they don't overlap
        offsets = [3.0, 10.0]

        for ii, slope in enumerate(self.props.reference_slopes):
            slope_label = _SLOPE_LABELS.get(slope, rf"$k^{{{slope:.1f}}}$")
            offset = (
                offsets[ii]
                if ii < len(offsets)
                else offsets[-1] * (2.0 ** (ii - len(offsets) + 1))
            )
            ref_y = offset * e_anchor * (k_seg / k_anchor) ** slope

            ref_line = self.ax.loglog(
                k_seg,
                ref_y,
                linestyle="--",
                color="black",
                linewidth=1.0,
                alpha=0.5,
            )[0]
            self._ref_lines.append(ref_line)

            # annotate just above the midpoint of the segment
            mid = len(k_seg) // 2
            ann = self.ax.annotate(
                slope_label,
                xy=(k_seg[mid], ref_y[mid]),
                fontsize=10,
                color="black",
                alpha=0.7,
                ha="center",
                va="bottom",
                xytext=(0, 2),
                textcoords="offset points",
            )
            self._ref_annotations.append(ann)

    def cleanup(self) -> None:
        if hasattr(self, "ax"):
            if self._main_line and self._main_line in self.ax.lines:
                self._main_line.remove()
            if self._smooth_line and self._smooth_line in self.ax.lines:
                self._smooth_line.remove()
            for line in self._ref_lines:
                if line in self.ax.lines:
                    line.remove()
            for ann in self._ref_annotations:
                ann.remove()
            for item in self._freq_lines:
                item.remove()
            for line in self._fap_lines:
                if line in self.ax.lines:
                    line.remove()
        self._main_line = None
        self._smooth_line = None
        self._ref_lines = []
        self._ref_annotations = []
        self._freq_lines = []
        self._fap_lines = []
