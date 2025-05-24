from typing import Any, Sequence
from dataclasses import dataclass
import numpy as np
from matplotlib.lines import Line2D
from simbi.tools.visualization.constants.alias import FIELD_ALIASES
from simbi.tools.visualization.state.core import VisualizationState
from .base import Component
from ..bridge import SimbiDataBridge


@dataclass
class AccretionTimeSeriesData:
    """Container for accretion time series data"""

    times: Sequence[float]
    accreted_mass: dict[str, Sequence[float]]
    accretion_rate: dict[str, Sequence[float]]

    @property
    def array(self):
        return np.array(self.times)

    def get_body_data(self, body_id: str, data_type: str):
        """Get data for a specific body"""
        if data_type == "accreted_mass":
            return np.array(self.accreted_mass.get(body_id, []))
        elif data_type == "accretion_rate":
            return np.array(self.accretion_rate.get(body_id, []))
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    @property
    def body_ids(self):
        """Get list of all body IDs"""
        return list(self.accreted_mass.keys())


@dataclass
class TimeSeriesData:
    """Container for time series data"""

    times: Sequence[float]
    values: Sequence[float]
    weight_type: str
    field: str

    @property
    def array(self):
        return np.array(self.times), np.array(self.values)


class TemporalPlotComponent(Component):
    """Temporal plot visualization component"""

    def __init__(self, state: VisualizationState, id: str):
        super().__init__(state, id)
        self.times = None
        self.values = None
        self.is_initialized = False

    def setup(self) -> None:
        """Initialize temporal plot resources"""
        # Setup axis style
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        # Create bridge to access data
        self.bridge = SimbiDataBridge(self.state)

        # Initialize empty line
        self.line = self.ax.plot([], [], label=self.props.get("label"))[0]

        # Store reference in state
        self.state.plot_elements[f"{self.id}_line"] = self.line

        # Track whether we're showing accretion data
        self.is_accretion_data = False

    def render(self) -> Any:
        """Render the temporal plot with current data"""
        if not self.state.data:
            return self.line

        # Use pre-computed time series if available
        if self.times is not None and self.values is not None:
            # Use pre-computed time series
            self.line.set_data(self.times, self.values)

            # Update axis limits
            if self.props.get("auto_scale", True):
                self._update_limits(self.times, self.values)

            # Update labels
            field = self.props.get("field", "rho")
            weight_type = self.state.config.get("plot", {}).get("weight")

            # Set y-label based on field and weight
            if weight_type:
                weight_str = self.bridge.get_field_label(weight_type)
                field_str = self.bridge.get_field_label(field)
                self.ax.set_ylabel(
                    f"$\\langle$ {field_str} $\\rangle_{{{weight_str}}}$"
                )
            else:
                field_str = self.bridge.get_field_label(field)
                self.ax.set_ylabel(f"$\\langle$ {field_str} $\\rangle$")

            # Set x-label with units if appropriate
            if self.state.config.get("style", {}).get("orbital_params"):
                self.ax.set_xlabel("$t$ [orbit(s)]")
            else:
                self.ax.set_xlabel("$t$")

            # Show legend if needed
            if self.line.get_label() and self.state.config.get("style", {}).get(
                "legend", True
            ):
                self.ax.legend()

            return self.line

        # Get field and determine if it's accretion data
        field = self.props.get("field", "rho")
        if field in FIELD_ALIASES:
            field = FIELD_ALIASES[field]
        self.is_accretion_data = field in ["accreted_mass", "accretion_rate"]

        self.format_axis()
        if self.is_accretion_data:
            return self._render_accretion_data(field)
        else:
            return self._render_regular_field(field)

    def _render_accretion_data(self, field_type: str) -> Line2D:
        """Render accretion data"""
        # Get time series data
        accretion_series = self._compute_accretion_time_series()
        times = accretion_series.array

        # Apply orbital period scaling if configured
        if self.state.config.get("style", {}).get("orbital_params"):
            # Get orbital period
            p = self.state.config["style"]["orbital_params"]
            separation = p.get("separation")
            total_mass = p.get("mass")

            if separation is not None and total_mass is not None:
                import math

                orbital_period = (
                    2.0
                    * math.pi
                    * math.sqrt(float(separation) ** 3 / float(total_mass))
                )
                # Convert times to orbital periods
                times = np.array(times) / orbital_period

        # Get body ID to plot
        body_id = self.props.get("body_id")

        # If no specific body ID, use the first one
        if body_id is None and accretion_series.body_ids:
            body_id = accretion_series.body_ids[0]

        # Skip if no bodies
        if not body_id or not accretion_series.body_ids:
            return self.line

        # Get body data
        body_data = accretion_series.get_body_data(body_id, field_type)

        # Update line
        self.line.set_data(times, body_data)

        # Set label if not already set
        if not self.line.get_label() or self.line.get_label() == "_nolegend_":
            # Use provided label or generate default
            label = self.props.get("label") or f"Body {body_id.split('_')[-1]}"
            self.line.set_label(label)

        # Update axis limits
        if self.props.get("auto_scale", True):
            self._update_limits(times, body_data)

        # Set title based on what we're plotting
        title = (
            "Total Accreted Mass vs Time"
            if field_type == "accreted_mass"
            else "Accretion Rate vs Time"
        )
        self.ax.set_title(title)

        # Show legend
        self.ax.legend()

        return self.line

    def _render_regular_field(self, field: str) -> Line2D:
        """Render regular field time series"""
        # Compute time series for the field
        series = self._compute_time_series(field)
        times, values = series.array

        # Update line data
        self.line.set_data(times, values)

        # Update label if needed
        label = self.props.get("label")
        if label and not self.line.get_label():
            self.line.set_label(label)

        # Update axis limits
        if self.props.get("auto_scale", True):
            self._update_limits(times, values)

        # Set labels
        weight = series.weight_type
        if weight:
            weight_str = self.bridge.get_field_label(weight)
            field_str = self.bridge.get_field_label(field)
            self.ax.set_ylabel(f"$\\langle$ {field_str} $\\rangle_{{{weight_str}}}$")
        else:
            field_str = self.bridge.get_field_label(field)
            self.ax.set_ylabel(f"$\\langle$ {field_str} $\\rangle$")

        # Set x-label with units if appropriate
        if self.state.config.get("style", {}).get("orbital_params"):
            self.ax.set_xlabel("$t$ [orbit(s)]")
        else:
            self.ax.set_xlabel("$t$")

        # Show legend if needed
        if self.line.get_label() and self.state.config.get("style", {}).get(
            "legend", True
        ):
            self.ax.legend()

        return self.line

    def _compute_time_series(self, field: str) -> TimeSeriesData:
        """Compute time series for given field"""
        times, values = [], []
        weight_type = self.state.config.get("plot", {}).get("weight")

        data = self.state.data

        # Get variable and weights
        var = self.bridge.get_variable(field)

        if weight_type:
            weights = self.bridge.get_variable(weight_type)
        else:
            weights = None

        # Calculate weighted mean
        if weights is not None:
            dV = self._calc_volume()
            value = self._compute_weighted_mean(var, weights, dV)
        else:
            value = np.max(var)

        # Add data point
        times.append(data.setup.get("time", 0.0))
        values.append(value)

        return TimeSeriesData(times, values, weight_type, field)

    def _compute_accretion_time_series(self) -> AccretionTimeSeriesData:
        """Compute accretion time series data"""
        times = []
        accreted_mass = {}
        accretion_rate = {}

        data = self.state.data

        # Skip if no immersed bodies
        if not data.immersed_bodies:
            return AccretionTimeSeriesData([], {}, {})

        # Add current data point
        times.append(data.setup.get("time", 0.0))

        # Process immersed bodies data
        for body_id, body_data in data.immersed_bodies.items():
            # Check if body is an accretor
            if "total_accreted_mass" in body_data:
                # Initialize lists if this is the first time we're seeing this body
                if body_id not in accreted_mass:
                    accreted_mass[body_id] = []
                    accretion_rate[body_id] = []

                accreted_mass[body_id].append(body_data["total_accreted_mass"])
                accretion_rate[body_id].append(body_data["accretion_rate"])

        return AccretionTimeSeriesData(times, accreted_mass, accretion_rate)

    def _calc_volume(self) -> np.ndarray:
        """Calculate cell volumes"""
        from ....functional.helpers import calc_cell_volume

        mesh = self.state.data.mesh
        ndim = self.state.config.get("plot", {}).get("ndim", 1)

        coords = [mesh.get(f"x{i + 1}v") for i in range(ndim)]
        coord_system = self.state.data.setup.get("coord_system", "cartesian")

        return calc_cell_volume(coords=coords, coord_system=coord_system, vertices=True)

    def _compute_weighted_mean(
        self, var: np.ndarray, weights: np.ndarray, dV: np.ndarray
    ) -> float:
        """Compute weighted mean of variable"""
        if var.shape != weights.shape:
            # Handle shape mismatch - might need reshaping based on specifics
            return np.max(var)

        field = self.props.get("field", "rho")
        weight_type = self.state.config.get("plot", {}).get("weight")

        if weight_type == field:
            return np.max(var)

        return np.sum(weights * var * dV) / np.sum(weights * dV)

    def _update_limits(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Update axis limits based on data"""
        # Check if user defined limits
        xlims = self.state.config.get("style", {}).get("xlims", (None, None))
        ylims = self.state.config.get("style", {}).get("ylims", (None, None))

        # Set auto x-limits if not user-defined
        if not any(xlims):
            if len(x_data) > 1:
                margin = 0.05 * (np.max(x_data) - np.min(x_data))
                self.ax.set_xlim(np.min(x_data) - margin, np.max(x_data) + margin)
        else:
            self.ax.set_xlim(xlims)

        # Set auto y-limits if not user-defined
        if not any(ylims):
            if len(y_data) > 1:
                margin = 0.05 * (np.max(y_data) - np.min(y_data))
                self.ax.set_ylim(np.min(y_data) - margin, np.max(y_data) + margin)
        else:
            self.ax.set_ylim(ylims)

    def update(self, props: dict[str, Any]) -> None:
        """Update component properties"""
        super().update(props)

        # Store time series data if provided
        if "times" in props and "values" in props:
            self.times = np.array(props["times"])
            self.values = np.array(props["values"])

        # Apply any property updates to the line
        if hasattr(self, "line"):
            if "color" in props:
                self.line.set_color(props["color"])
            if "linestyle" in props:
                self.line.set_linestyle(props["linestyle"])
            if "linewidth" in props:
                self.line.set_linewidth(props["linewidth"])
            if "label" in props:
                self.line.set_label(props["label"])
                if self.state.config.get("style", {}).get("legend", True):
                    self.ax.legend()
