"""Line plot component for visualization with functional programming principles."""

from typing import Optional, Any
import numpy as np
from pydantic import Field, field_validator, ValidationInfo
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ..core.types import PlotData, FieldData, Array
from .interface import ComponentProps
from simbi.functional.utilities import (
    map_with_index,
    for_each,
)


# ---- Pure styling functions ----


def create_line_style(
    linewidth: float,
    alpha: float,
    colors: list[str],
    linestyles: list[str],
    markers: list[str],
    marker_size: float,
    index: int,
) -> dict[str, Any]:
    """Create styling for a line based on index and style properties."""
    style: dict[str, str | float] = {"linewidth": linewidth, "alpha": alpha}

    # Add color if specified
    if index < len(colors):
        style["color"] = colors[index]

    # Add linestyle (cycling through available styles)
    if linestyles:
        style["linestyle"] = linestyles[index % len(linestyles)]

    # Add marker if specified
    if index < len(markers):
        style["marker"] = markers[index]
        style["markersize"] = marker_size

    return style


def get_line_label(
    field: FieldData, index: int, custom_labels: Optional[list[str]], use_legend: bool
) -> Optional[str]:
    """Get label for a line based on field, index, and label settings."""
    if not use_legend:
        return None

    # Use custom label if provided
    if custom_labels and index < len(custom_labels):
        return custom_labels[index]

    # Otherwise use field name
    return field.name


# ---- Data extraction functions ----


def extract_x_data(field: FieldData) -> Array:
    """Extract x-axis data from field."""
    # Use first domain array as x-coordinates
    if field.domain:
        return field.domain[0]

    # Fallback to index array if no domain provided
    return np.arange(field.values.size, dtype=np.floating)


def extract_y_data(field: FieldData, x_data: Array) -> Array:
    """Extract y-axis data from field."""
    y_data = field.values

    # Handle dimensionality
    if y_data.ndim > 1:
        if y_data.shape[0] == x_data.size:
            # First dimension matches x_data size
            y_data = y_data if y_data.ndim == 1 else y_data[:, 0]
        elif y_data.size == x_data.size:
            # Flatten if total size matches
            y_data = y_data.flatten()
        else:
            # Take first slice as fallback
            y_data = y_data[0] if y_data.ndim > 1 else y_data

    return y_data


# ---- Line creation and update functions ----


def create_line(
    ax: Axes, field: FieldData, style: dict[str, Any], label: Optional[str]
) -> Line2D:
    """Create a new line on the axes."""
    x_data = extract_x_data(field)
    y_data = extract_y_data(field, x_data)

    # Create the line with styling
    line = ax.plot(x_data, y_data, label=label, **style)[0]
    return line


def update_line(
    line: Line2D, field: FieldData, style: dict[str, Any], label: Optional[str]
) -> Line2D:
    """Update an existing line with new data and styling."""
    x_data = extract_x_data(field)
    y_data = extract_y_data(field, x_data)

    # Update data
    line.set_data(x_data, y_data)

    # Update label
    if label is not None:
        line.set_label(label)

    # Update styling
    for key, value in style.items():
        setter = getattr(line, f"set_{key}", None)
        if setter:
            setter(value)

    return line


# ---- Legend handling ----


def should_use_legend(show_legend: Optional[bool], field_count: int) -> bool:
    """Determine if legend should be used."""
    if show_legend is not None:
        return show_legend
    # Auto-determine based on number of fields
    return field_count > 1


def update_legend(ax: Axes, lines: list[Line2D]) -> None:
    """Update legend if any lines have labels."""
    if any(
        line.get_label() and not str(line.get_label()).startswith("_") for line in lines
    ):
        ax.legend()


# ---- Component class ----


class LinePlotProps(ComponentProps):
    """Properties for line plot component."""

    field_indices: list[int] = Field(default_factory=lambda: [0])
    labels: Optional[list[str]] = None
    colors: list[str] = Field(default_factory=list)
    linestyles: list[str] = Field(default_factory=lambda: ["-", "--", ":", "-."])
    linewidth: float = 2.0
    markers: list[str] = Field(default_factory=list)
    show_legend: Optional[bool] = None
    marker_size: float = 6.0
    alpha: float = 1.0

    @field_validator("field_indices")
    @classmethod
    def validate_field_indices(cls, v: list[int], info: ValidationInfo) -> list[int]:
        """Validate that field indices are non-negative."""
        if not v:
            raise ValueError("At least one field index must be specified")

        for idx in v:
            if idx < 0:
                raise ValueError(f"Field indices must be non-negative, got {idx}")

        return v

    @field_validator("linewidth", "marker_size", "alpha")
    @classmethod
    def validate_positive_float(cls, v: float, info: ValidationInfo) -> float:
        """Validate that numeric values are positive."""
        if v <= 0:
            field_name = info.field_name or "Value"
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v


class LinePlotComponent:
    """Line plot visualization component with functional approach."""

    def __init__(self, props: LinePlotProps):
        """Initialize the line plot component."""
        self.props = props
        self._lines: list[Line2D] = []
        self._initialized: bool = False

    def initialize(self, fig: Figure, ax: Axes) -> None:
        """Initialize the component with figure and axes."""
        self.fig = fig
        self.ax = ax
        self._initialized = True

    def update(self, props: LinePlotProps) -> None:
        """Update component properties."""
        self.props = props

    def render(self, data: PlotData) -> list[Line2D]:
        """Render the line plot with data using functional patterns."""
        if not self._initialized or not hasattr(self, "ax"):
            raise RuntimeError("Component not initialized. Call initialize() first.")

        ax = self.ax

        # Reset lines if field selection changed
        if self._should_reset_lines():
            self._reset_lines()

        # Determine if we should use legend
        use_legend = should_use_legend(
            self.props.show_legend, len(self.props.field_indices)
        )

        # Process each field and create/update lines
        processed_lines = self._process_fields(data, use_legend)

        # Filter out None values from any skipped fields
        self._lines = list(filter(lambda x: x is not None, processed_lines))

        # Show legend if needed
        if use_legend:
            update_legend(ax, self._lines)

        return self._lines

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, "ax"):
            for_each(
                lambda line: line.remove() if line in self.ax.lines else None,
                self._lines,
            )
        self._lines = []

    def _should_reset_lines(self) -> bool:
        """Determine if lines should be reset based on field count change."""
        return len(self._lines) != len(self.props.field_indices)

    def _reset_lines(self) -> None:
        """Reset all lines."""
        if hasattr(self, "ax"):
            for_each(
                lambda line: line.remove() if line in self.ax.lines else None,
                self._lines,
            )
        self._lines = []

    def _process_fields(
        self, data: PlotData, use_legend: bool
    ) -> list[Optional[Line2D]]:
        """Process each field and create or update corresponding line."""
        return map_with_index(
            lambda i, field_idx: self._process_field(data, field_idx, i, use_legend),
            self.props.field_indices,
        )

    def _process_field(
        self, data: PlotData, field_idx: int, line_idx: int, use_legend: bool
    ) -> Optional[Line2D]:
        """Process a single field and create or update its line."""
        # Skip if field index is out of range
        if field_idx >= len(data.fields):
            return None

        # Get field data
        field = data.fields[field_idx]

        # Get styling and label
        style = create_line_style(
            self.props.linewidth,
            self.props.alpha,
            self.props.colors,
            self.props.linestyles,
            self.props.markers,
            self.props.marker_size,
            line_idx,
        )

        label = get_line_label(field, line_idx, self.props.labels, use_legend)

        # Create or update line
        if line_idx < len(self._lines):
            # Update existing line
            return update_line(self._lines[line_idx], field, style, label)
        else:
            # Create new line
            return create_line(self.ax, field, style, label)
