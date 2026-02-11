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
    rbeg: float = 0.2  # Reference line start radius
    rend: float = 0.5  # Reference line end radius

    # reference power-law overlay
    show_reference_lines: bool = True
    reference_power_law: float = -1.5
    reference_scale: float = 1.5
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

            ref_beg_idx = np.argmax(r_ref > self.props.rbeg)
            ref_end_idx = np.argmax(r_ref > self.props.rend)
            power = self.props.reference_power_law

            if ref_beg_idx > 0 and ref_end_idx > ref_beg_idx:
                anchor = val_ref[ref_beg_idx] / (r_ref[ref_beg_idx] ** power)
                ref_vals = anchor * (r_ref**power)

                ref_line = self.ax.plot(
                    r_ref[ref_beg_idx:ref_end_idx],
                    ref_vals[ref_beg_idx:ref_end_idx]
                    * self.props.reference_scale,
                    linestyle="--",
                    color="red",
                    label=rf"$r^{{{power}}}$",
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
