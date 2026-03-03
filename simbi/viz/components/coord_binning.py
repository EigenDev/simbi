from typing import List, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from simbi.viz.utility import get_field_str

from ..config import FigureConfig
from ..types import Array, FieldData, RenderResult
from .interface import Component, ComponentProps


def stripped_field_name(field_name: str) -> tuple[str, str]:
    field_base_name = field_name.split("_vs_r")[0]
    field_str = get_field_str(field_base_name)
    if field_str.startswith("$") and field_str.endswith("$"):
        field_str = field_str[1:-1]
    return field_base_name, rf"$\langle {field_str} \rangle$"


class CoordinateProfileProps(ComponentProps):
    """Properties for radial profile component."""

    label: Optional[str] = None
    color: Optional[str] = None
    linestyle: str = "-"
    linewidth: float = 2.0
    normalization: float = 1
    x_normalization: float = 1
    rbeg: float = 0.0  # Reference line start radius
    rend: float = 0.5  # Reference line end radius

    # broken power-law fit overlay
    show_reference_lines: bool = True
    reference_fields: tuple[str, ...] = ("rho",)

    x_scale: str = "linear"
    y_scale: str = "linear"


class CoordinateProfileComponent(Component[CoordinateProfileProps, FieldData]):
    """
    Renders a 1D radial profile and adds analysis-specific formatting.

    Note: components now return a RenderResult object. This ensures the
    Figure/Formatter can reliably inspect artists and metadata for
    colorbar/legend/label decisions.
    """

    def __init__(self, props: CoordinateProfileProps):
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

    def update(self, props: CoordinateProfileProps) -> None:
        self.props = props
        # We would re-render if props change
        pass

    def render(self, data: FieldData, style: FigureConfig) -> RenderResult:
        """Render the radial profile with guaranteed 1D data and return a RenderResult."""
        if not self._initialized:
            raise RuntimeError("Component not initialized.")

        if data.ndim != 1:
            raise ValueError("CoordinateProfileComponent expects 1D FieldData.")

        r_bins = data.domain[0] / self.props.x_normalization
        values = data.values
        _, field_str = stripped_field_name(data.name)
        norm = self.props.normalization
        line_label = self.props.label if self.props.label else field_str
        # --- Render Main Line ---
        good_bins = ~np.isnan(values)
        if self._main_line is None:
            self._main_line = self.ax.plot(
                r_bins[good_bins],
                values[good_bins] / norm,
                label=line_label,
            )[0]
        else:
            self._main_line.set_data(
                r_bins[good_bins], values[good_bins] / norm
            )

        if norm != 1:
            # if plotting a negative normalized quantity, we need to
            # account for whether the horizontal line is -1 or 1
            norm_loc = 1.0 if np.all(values / norm > 0) else -1.0
            self.ax.axhline(
                norm_loc, color="gray", linestyle="--", linewidth=0.5
            )

        # --- Apply Special Formatting ---
        self._format_axes(r_bins, values, data.name)
        # return a RenderResult so the Figure/Formatter can inspect artists and metadata
        return RenderResult(
            artists={"line": self._main_line, "refs": self._ref_lines},
            metadata={"label": field_str},
        )

    def _format_axes(self, r_bins: Array, values: Array, field_name: str):
        """Apply analysis-specific formatting."""
        self.ax.set_xscale(self.props.x_scale)
        self.ax.set_yscale(self.props.y_scale)
        field_base_name, field_str = stripped_field_name(field_name)

        self.ax.set_xlabel(
            r"$\tilde{r}$" if self.props.x_normalization != 1 else "$r$"
        )

        if (
            self.props.show_reference_lines
            and field_base_name in self.props.reference_fields
        ):
            good_bins = ~np.isnan(values)
            r_ref = r_bins[good_bins]
            val_ref = values[good_bins]
            positive = (r_ref > 0) & (val_ref > 0)

            if np.sum(positive) >= 6:
                self._fit_broken_power_law(
                    r_ref[positive],
                    val_ref[positive],
                    field_base_name,
                )

    def _fit_broken_power_law(
        self,
        rp: Array,
        vp: Array,
        field_base_name: str,
    ) -> None:
        """fit a 2-segment broken power law in [rbeg, rend], draw the result."""
        # interpolate full data onto a dense log-uniform grid,
        # then restrict to [rbeg, rend] for both fitting and drawing
        r_lo = max(rp[0], self.props.rbeg) if self.props.rbeg > 0 else rp[0]
        r_hi = min(rp[-1], self.props.rend) if self.props.rend > 0 else rp[-1]
        if r_hi <= r_lo:
            return

        n_resample = 200
        lr_full = np.log(rp)
        lv_full = np.log(vp)

        # dense log-uniform grid in [r_lo, r_hi]
        lr = np.linspace(np.log(r_lo), np.log(r_hi), n_resample)
        lv = np.interp(lr, lr_full, lv_full)

        # scan break points
        min_seg = 5
        best_ssr = np.inf
        best_break = n_resample // 2
        for kk in range(min_seg, n_resample - min_seg):
            s1, i1 = np.polyfit(lr[:kk], lv[:kk], 1)
            s2, i2 = np.polyfit(lr[kk:], lv[kk:], 1)
            r1 = lv[:kk] - (s1 * lr[:kk] + i1)
            r2 = lv[kk:] - (s2 * lr[kk:] + i2)
            ssr = np.sum(r1**2) + np.sum(r2**2)
            if ssr < best_ssr:
                best_ssr = ssr
                best_break = kk

        r_break = np.exp(lr[best_break])

        # fit each segment on the resampled grid
        s1, i1 = np.polyfit(lr[:best_break], lv[:best_break], 1)
        s2, i2 = np.polyfit(lr[best_break:], lv[best_break:], 1)

        print(f"[{field_base_name}] broken power law:")
        print(f"  inner slope: {s1:.3f}  (r < {r_break:.2f})")
        print(f"  outer slope: {s2:.3f}  (r > {r_break:.2f})")

        # draw lines offset slightly above data for visibility
        offset = 1.3
        n_draw = 100
        r_inner = np.geomspace(r_lo, r_break, n_draw)
        r_outer = np.geomspace(r_break, r_hi, n_draw)
        y_inner = offset * np.exp(i1) * r_inner ** s1
        y_outer = offset * np.exp(i2) * r_outer ** s2

        line_inner = self.ax.plot(
            r_inner, y_inner,
            linestyle="--", linewidth=1.5, color="gray",
        )[0]
        line_outer = self.ax.plot(
            r_outer, y_outer,
            linestyle="--", linewidth=1.5, color="gray",
        )[0]
        self._ref_lines.extend([line_inner, line_outer])

        # annotate above midpoint of each segment
        mid_inner = len(r_inner) // 2
        self.ax.annotate(
            rf"$r^{{{s1:.2f}}}$",
            xy=(r_inner[mid_inner], y_inner[mid_inner]),
            xytext=(0, 12), textcoords="offset points",
            fontsize=10,
            ha="center", va="bottom",
        )
        mid_outer = len(r_outer) // 2
        self.ax.annotate(
            rf"$r^{{{s2:.2f}}}$",
            xy=(r_outer[mid_outer], y_outer[mid_outer]),
            xytext=(0, 12), textcoords="offset points",
            fontsize=10,
            ha="center", va="bottom",
        )

        self.ax.axvline(
            r_break, color="gray", linestyle=":", linewidth=0.8
        )

    def cleanup(self) -> None:
        if hasattr(self, "ax"):
            if self._main_line and self._main_line in self.ax.lines:
                self._main_line.remove()
            for line in self._ref_lines:
                if line in self.ax.lines:
                    line.remove()
        self._main_line = None
        self._ref_lines = []
