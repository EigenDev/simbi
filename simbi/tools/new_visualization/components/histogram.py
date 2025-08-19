"""Histogram plot component for visualization."""

from typing import Optional, Any, Literal, cast
import numpy as np
from pydantic import field_validator, ValidationInfo
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.container import BarContainer

from ..core.types import PlotData, Array, Bounds
from .interface import ComponentProps


# ---- Pure data transformation functions ----


def calculate_histogram(
    values: Array,
    bins: int | Array,
    weights: Optional[Array] = None,
    density: bool = False,
    range_bounds: Optional[tuple[float, float]] = None,
) -> tuple[Array, Array]:
    """
    Calculate histogram data.

    Args:
        values: Data to histogram
        bins: Number of bins or bin edges
        weights: Optional weights for each value
        density: Whether to normalize histogram
        range_bounds: Optional range to limit histogram

    Returns:
        Tuple of (bin edges, histogram values)
    """
    # Filter invalid values if any
    valid_mask = np.isfinite(values)
    if not np.all(valid_mask):
        values = values[valid_mask]
        if weights is not None:
            weights = weights[valid_mask]

    # Calculate histogram
    hist_values, bin_edges = np.histogram(
        values, bins=bins, weights=weights, density=density, range=range_bounds
    )

    return bin_edges, hist_values


def calculate_bin_centers(bin_edges: Array) -> Array:
    """Calculate bin centers from bin edges."""
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def calculate_log_bins(data_min: float, data_max: float, num_bins: int = 50) -> Array:
    """Calculate logarithmically spaced bins."""
    # Ensure positive values for log spacing
    if data_min <= 0:
        data_min = max(np.min(data_max * 0.01), 1e-10)

    return np.geomspace(data_min, data_max, num_bins + 1)


def fit_power_law(
    x: Array, y: Array, x_min: Optional[float] = None, x_max: Optional[float] = None
) -> tuple[float, float, Array, Array]:
    """
    Fit a power law to data.

    Args:
        x: X values
        y: Y values
        x_min: Optional minimum x for fitting
        x_max: Optional maximum x for fitting

    Returns:
        Tuple of (alpha, amplitude, x_fit, y_fit) where:
            alpha: Power law exponent
            amplitude: Power law amplitude
            x_fit: X values for plotting the fit
            y_fit: Y values for plotting the fit
    """
    # Skip if not enough data
    if len(x) < 3 or len(y) < 3:
        return 0.0, 0.0, np.array([]), np.array([])

    # Apply range limits if provided
    if x_min is not None or x_max is not None:
        mask = np.ones_like(x, dtype=bool)
        if x_min is not None:
            mask &= x >= x_min
        if x_max is not None:
            mask &= x <= x_max

        if not np.any(mask):
            return 0.0, 0.0, np.array([]), np.array([])

        x_fit = x[mask]
        y_fit = y[mask]
    else:
        x_fit = x
        y_fit = y

    # Perform linear fit in log-log space
    log_x = np.log10(x_fit)
    log_y = np.log10(y_fit)

    # Filter out any invalid values
    valid = np.isfinite(log_x) & np.isfinite(log_y)
    if not np.any(valid):
        return 0.0, 0.0, np.array([]), np.array([])

    log_x = log_x[valid]
    log_y = log_y[valid]

    # Perform linear regression
    slope, intercept = np.polyfit(log_x, log_y, 1)

    # Calculate power law parameters
    alpha = -slope
    amplitude = 10**intercept

    # Generate fitted curve for plotting
    x_smooth = np.geomspace(x_fit.min(), x_fit.max(), 100)
    y_smooth = amplitude * x_smooth ** (-alpha)

    return alpha, amplitude, x_smooth, y_smooth


# ---- Component class ----


class HistogramPlotProps(ComponentProps):
    """Properties for histogram plot component."""

    field_index: int = 0
    nbins: int = 50
    log_bins: bool = True
    range: Optional[Bounds] = None
    density: bool = False
    histtype: Literal["bar", "step", "barstacked", "stepfilled"] = "bar"
    cumulative: bool = False
    color: Optional[str] = None
    edgecolor: Optional[str] = "black"
    alpha: float = 0.7
    linewidth: float = 1.0
    label: Optional[str] = None
    fit_power_law: bool = False
    power_law_range: Optional[Bounds] = None

    @field_validator("field_index")
    @classmethod
    def validate_field_index(cls, v: int, info: ValidationInfo) -> int:
        """Validate that field index is non-negative."""
        if v < 0:
            raise ValueError(f"Field index must be non-negative, got {v}")
        return v

    @field_validator("nbins")
    @classmethod
    def validate_nbins(cls, v: int, info: ValidationInfo) -> int:
        """Validate that nbins is positive."""
        if v <= 0:
            raise ValueError(f"Number of bins must be positive, got {v}")
        return v

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float, info: ValidationInfo) -> float:
        """Validate that alpha is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {v}")
        return v

    @field_validator("linewidth")
    @classmethod
    def validate_linewidth(cls, v: float, info: ValidationInfo) -> float:
        """Validate that linewidth is positive."""
        if v < 0:
            raise ValueError(f"Linewidth must be non-negative, got {v}")
        return v

    @field_validator("histtype")
    @classmethod
    def validate_histtype(cls, v: str, info: ValidationInfo) -> str:
        """Validate histtype option."""
        valid_options = ["bar", "barstacked", "step", "stepfilled"]
        if v not in valid_options:
            raise ValueError(f"Histtype must be one of {valid_options}, got {v}")
        return v


class HistogramPlotComponent:
    """Histogram plot visualization component."""

    def __init__(self, props: HistogramPlotProps):
        """Initialize the histogram plot component."""
        self.props = props
        self._histogram: Optional[BarContainer] = None
        self._power_law_line = None
        self._initialized: bool = False
        self._bin_edges: Optional[Array] = None
        self._hist_values: Optional[Array] = None

    def initialize(self, fig: Figure, ax: Axes) -> None:
        """Initialize the component with figure and axes."""
        self.fig = fig
        self.ax = ax
        self._initialized = True

    def update(self, props: HistogramPlotProps) -> None:
        """Update component properties."""
        # Check if we need to re-render due to property changes
        needs_rerender = (
            props.nbins != self.props.nbins
            or props.log_bins != self.props.log_bins
            or props.density != self.props.density
            or props.histtype != self.props.histtype
            or props.cumulative != self.props.cumulative
            or props.fit_power_law != self.props.fit_power_law
        )

        # Store new props
        self.props = props

        # Re-render if needed and we have already rendered before
        if (
            needs_rerender
            and self._histogram is not None
            and self._bin_edges is not None
        ):
            if not self._hist_values:
                raise RuntimeError(
                    "Cannot re-render histogram without existing bin edges and values."
                )
            # Re-render with existing data
            self._render_histogram(self._bin_edges, self._hist_values)

    def render(self, data: PlotData) -> dict[str, Any]:
        """Render the histogram plot with data."""
        if not self._initialized or not hasattr(self, "ax"):
            raise RuntimeError("Component not initialized. Call initialize() first.")

        # Skip if field index is out of range
        if self.props.field_index >= len(data.fields):
            return {"histogram": self._histogram, "power_law": self._power_law_line}

        # Get field data
        field = data.fields[self.props.field_index]

        # Extract values
        values = field.values.flatten()

        # Calculate histogram
        range_bounds = None
        if self.props.range:
            range_bounds = (self.props.range.min, self.props.range.max)

        # Determine bins
        if self.props.log_bins and np.min(values) > 0:
            bins = calculate_log_bins(
                float(np.min(values)), float(np.max(values)), self.props.nbins
            )
        else:
            bins = self.props.nbins

        # Calculate histogram
        bin_edges, hist_values = calculate_histogram(
            values, bins=bins, density=self.props.density, range_bounds=range_bounds
        )

        # Store for potential updates
        self._bin_edges = bin_edges
        self._hist_values = hist_values

        # Render the histogram
        hist_container = self._render_histogram(bin_edges, hist_values)

        return {"histogram": hist_container, "power_law": self._power_law_line}

    def _render_histogram(self, bin_edges: Array, hist_values: Array) -> BarContainer:
        """Render the histogram with the given data."""
        # Clear previous histogram if exists
        if self._histogram is not None:
            for patch in self._histogram.patches:
                patch.remove()

        if self._power_law_line is not None:
            self._power_law_line.remove()
            self._power_law_line = None

        # Calculate bin centers for step and line plots
        bin_centers = calculate_bin_centers(bin_edges)

        # Render histogram
        _, _, res = self.ax.hist(
            bin_centers,
            bins=[float(x) for x in bin_edges],
            weights=hist_values,
            histtype=self.props.histtype,
            color=self.props.color,
            edgecolor=self.props.edgecolor,
            alpha=self.props.alpha,
            linewidth=self.props.linewidth,
            label=self.props.label,
            cumulative=self.props.cumulative,
            log=True if self.props.log_bins else False,
        )
        self._histogram = cast(BarContainer, res)

        # Fit power law if requested
        if self.props.fit_power_law:
            self._add_power_law_fit(bin_centers, hist_values)

        # Add legend if label is provided
        if self.props.label or (
            self._power_law_line and self._power_law_line.get_label()
        ):
            self.ax.legend()

        return self._histogram

    def _add_power_law_fit(self, x: Array, y: Array) -> None:
        """Add power law fit to the histogram."""
        # Get range for power law fitting
        x_min = None
        x_max = None
        if self.props.power_law_range:
            x_min = self.props.power_law_range.min
            x_max = self.props.power_law_range.max

        # Fit power law
        alpha, amplitude, x_fit, y_fit = fit_power_law(x, y, x_min, x_max)

        # Skip if fit failed
        if len(x_fit) == 0 or len(y_fit) == 0:
            return

        # Plot the fit
        self._power_law_line = self.ax.plot(
            x_fit, y_fit, "--", color="red", linewidth=2, label=f"α = {alpha:.2f}"
        )[0]

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, "ax"):
            if self._histogram is not None:
                for patch in self._histogram.patches:
                    patch.remove()
                self._histogram = None

            if self._power_law_line is not None:
                self._power_law_line.remove()
                self._power_law_line = None
