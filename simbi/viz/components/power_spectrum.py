# =============================================================================
# power_spectrum.py
#
# component for rendering kinetic energy power spectrum E(k).
# log-log line plot with optional reference slope overlays.
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
    reference_slopes: tuple[float, ...] = (-5.0 / 3.0, -2.0)
    compensated: bool = False
    linewidth: float = 2.0
    color: Optional[str] = None
    label: Optional[str] = None


_SLOPE_LABELS = {
    -5.0 / 3.0: r"$k^{-5/3}$",
    -2.0: r"$k^{-2}$",
    -3.0: r"$k^{-3}$",
    -4.0: r"$k^{-4}$",
}


class PowerSpectrumComponent(Component[PowerSpectrumProps, FieldData]):
    """renders kinetic energy power spectrum as a log-log line plot."""

    def __init__(self, props: PowerSpectrumProps):
        self.props = props
        self._main_line: Optional[Line2D] = None
        self._ref_lines: List[Line2D] = []
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

        # axis labels from data when available, fall back to E(k) vs k
        has_custom_axes = data.axis_names and len(data.axis_names) >= 1
        default_xlabel = data.axis_names[0] if has_custom_axes else r"$k$"
        default_ylabel = data.name if has_custom_axes else r"$E(k)$"

        if self.props.compensated and not has_custom_axes:
            y = k ** (5.0 / 3.0) * e_k
            ylabel = r"$k^{5/3}\,E(k)$"
        else:
            y = e_k
            ylabel = default_ylabel

        # prefer body name for legend (multi-body PSD), then props, then ylabel
        body_label = data.body_names[0] if data.body_names else None
        label = self.props.label or body_label or ylabel

        line_kwargs = {"linewidth": self.props.linewidth, "label": label}
        if self.props.color:
            line_kwargs["color"] = self.props.color

        if self._main_line is None:
            self._main_line = self.ax.loglog(k, y, **line_kwargs)[0]
        else:
            self._main_line.set_data(k, y)

        # reference slopes only for spatial spectra (not temporal PSD)
        if (
            self.props.show_reference_slopes
            and not self.props.compensated
            and not has_custom_axes
        ):
            self._draw_reference_slopes(k, e_k)

        if self.ax.get_legend_handles_labels()[1]:
            self.ax.legend(loc="best")

        self.ax.set_xlabel(default_xlabel)
        self.ax.set_ylabel(ylabel)

        return RenderResult(
            artists={"line": self._main_line, "refs": self._ref_lines},
            metadata={"label": label, "is_line": True},
        )

    def _draw_reference_slopes(self, k: np.ndarray, e_k: np.ndarray) -> None:
        """overlay reference power-law slopes anchored to the mid-range of the spectrum."""
        # remove old reference lines
        for line in self._ref_lines:
            if line in self.ax.lines:
                line.remove()
        self._ref_lines = []

        n = len(k)
        if n < 4:
            return

        # use 10%-90% of the wavenumber range for reference lines
        i_lo = max(1, n // 10)
        i_hi = 9 * n // 10
        k_ref = k[i_lo:i_hi]

        # anchor at the geometric center of the spectrum
        i_mid = n // 2
        k_anchor = k[i_mid]
        e_anchor = e_k[i_mid]

        if e_anchor <= 0:
            return

        # find the max of the data in the reference range so slopes sit above it
        e_in_range = e_k[i_lo:i_hi]
        valid = e_in_range > 0
        if not np.any(valid):
            return
        peak = e_in_range[valid].max()

        colors = ["red", "blue", "green", "orange"]
        for ii, slope in enumerate(self.props.reference_slopes):
            color = colors[ii % len(colors)]
            slope_label = _SLOPE_LABELS.get(slope, rf"$k^{{{slope:.1f}}}$")
            raw_y = e_anchor * (k_ref / k_anchor) ** slope
            # shift up so the entire curve sits above the data
            scale = 3.0 * peak / raw_y.max() if raw_y.max() > 0 else 1.0
            ref_y = raw_y * scale
            ref_line = self.ax.loglog(
                k_ref,
                ref_y,
                linestyle="--",
                color=color,
                linewidth=1.0,
                alpha=0.7,
                label=slope_label,
            )[0]
            self._ref_lines.append(ref_line)

    def cleanup(self) -> None:
        if hasattr(self, "ax"):
            if self._main_line and self._main_line in self.ax.lines:
                self._main_line.remove()
            for line in self._ref_lines:
                if line in self.ax.lines:
                    line.remove()
        self._main_line = None
        self._ref_lines = []
