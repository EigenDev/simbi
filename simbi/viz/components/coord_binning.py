from typing import List, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from simbi.viz.utility import get_field_str

from ..config import StyleConfig
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
    rbeg: float = 0.2  # Reference line start radius
    rend: float = 0.5  # Reference line end radius

    # Analysis-specific formatting
    show_reference_lines: bool = True
    x_scale: str = "log"
    y_scale: str = "log"


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

    def render(self, data: FieldData, style: StyleConfig) -> RenderResult:
        """Render the radial profile with guaranteed 1D data and return a RenderResult."""
        if not self._initialized:
            raise RuntimeError("Component not initialized.")

        if data.ndim != 1:
            raise ValueError("CoordinateProfileComponent expects 1D FieldData.")

        r_bins = data.domain[0]
        values = data.values
        _, field_str = stripped_field_name(data.name)
        norm = self.props.normalization
        # --- Render Main Line ---
        good_bins = ~np.isnan(values)
        if self._main_line is None:
            self._main_line = self.ax.plot(
                r_bins[good_bins],
                values[good_bins] / norm,
                label=field_str,
            )[0]
        else:
            self._main_line.set_data(
                r_bins[good_bins], values[good_bins] / norm
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

        self.ax.set_xlabel("$r/a$")
        # self.ax.set_ylabel(field_str)

        # Add reference lines
        if self.props.show_reference_lines:
            # self.ax.axvline(50.0, color="gray", linestyle="--", linewidth=0.5)
            # Add power-law reference for density
            if field_base_name == "rho":
                good_bins = ~np.isnan(values)
                r_ref = r_bins[good_bins]
                val_ref = values[good_bins]

                # Find a good anchor point for the reference line
                ref_beg_idx = np.argmax(r_ref > 2)
                ref_end_idx = np.argmax(r_ref > 30)
                if ref_beg_idx > 0:
                    norm = val_ref[ref_beg_idx] / (r_ref[ref_beg_idx] ** (-1.5))
                    ref_vals = norm * (r_ref ** (-1.5))

                    ref_line = self.ax.plot(
                        r_ref[ref_beg_idx:ref_end_idx],
                        ref_vals[ref_beg_idx:ref_end_idx] * 1.5,
                        linestyle="--",
                        color="red",
                        label=r"$r^{-3/2}$",
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
