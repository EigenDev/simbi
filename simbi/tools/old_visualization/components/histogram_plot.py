from typing import Any
import numpy as np
from matplotlib.container import BarContainer
from .base import Component
from ..bridge import SimbiDataBridge


class HistogramComponent(Component):
    """Histogram visualization component for distribution functions"""

    def setup(self) -> None:
        """Initialize histogram plot resources"""
        # Setup axis style
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        # Set log scales by default for histogram
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")

        # Setup typical histogram ticks
        self.ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
        self.ax.set_xticklabels(["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])

        # Create bridge to access data
        self.bridge = SimbiDataBridge(self.state)

        # Initialize empty histogram
        self.hist_container = None

    def render(self) -> BarContainer:
        """Render the histogram with current data"""
        if not self.state.data:
            if self.hist_container:
                return self.hist_container
            else:
                # Return an empty container if we don't have one yet
                dummy_container = self.ax.hist(
                    [0], bins=[0, 1], weights=[0], histtype="step", linewidth=3.0
                )
                return dummy_container

        # Calculate cell volumes
        dV = self._calc_volume()

        # Compute variable to histogram
        var = self._compute_variable(dV)

        # Compute histogram data
        gbs, hist_data = self._compute_histogram(var)

        # Plot histogram
        label = self.props.get("label")
        self.hist_container = self.ax.hist(
            gbs,
            bins=gbs,
            weights=hist_data,
            label=label,
            histtype="step",
            rwidth=1.0,
            linewidth=3.0,
            color=self.props.get("color", "blue"),
        )

        # Fit power law if requested
        if self.state.config.get("plot", {}).get("powerfit"):
            self._add_power_law_fit(gbs, hist_data)

        # Update axis limits
        self._update_limits(gbs, hist_data)

        # Add legend if needed
        if label and self.state.config.get("style", {}).get("legend", True):
            self.ax.legend()

        return self.hist_container

    def _calc_volume(self) -> np.ndarray:
        """Calculate cell volumes"""
        from ....functional.helpers import calc_cell_volume

        mesh = self.state.data.mesh
        ndim = self.state.config.get("plot", {}).get("ndim", 1)

        coords = [mesh.get(f"x{i + 1}v") for i in range(ndim)]
        coord_system = self.state.data.setup.get("coord_system", "cartesian")

        return calc_cell_volume(coords=coords, coord_system=coord_system, vertices=True)

    def _compute_variable(self, dV: np.ndarray) -> np.ndarray:
        """Compute variable to histogram based on histogram type"""
        hist_type = self.state.config.get("plot", {}).get("hist_type", "kinetic")
        fields = self.state.data.fields

        from ... import utility as util

        if hist_type == "kinetic":
            # Kinetic energy
            mass = (
                dV
                * fields.get("W", np.ones_like(dV))
                * fields.get("rho", np.ones_like(dV))
            )
            var = (fields.get("W", np.ones_like(dV)) - 1.0) * mass * util.e_scale.value
        elif hist_type == "enthalpy":
            # Enthalpy
            adiabatic_index = fields.get("adiabatic_index", 5 / 3)
            enthalpy = 1.0 + adiabatic_index * fields.get("p", np.zeros_like(dV)) / (
                fields.get("rho", np.ones_like(dV)) * (adiabatic_index - 1.0)
            )
            var = (enthalpy - 1.0) * dV * util.e_scale.value
        elif hist_type == "mass":
            # Mass distribution
            var = (
                dV
                * fields.get("W", np.ones_like(dV))
                * fields.get("rho", np.ones_like(dV))
                * util.mass_scale.value
            )
        else:
            # Total energy density
            from ...utility import prims2var

            edens_total = prims2var(fields, "energy")
            var = edens_total * dV * util.e_scale.value

        return var

    def _compute_histogram(self, var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute histogram data from variable"""
        fields = self.state.data.fields
        gamma_beta = fields.get("gamma_beta", np.zeros_like(var))

        # Create logarithmically spaced bins
        gbs = np.geomspace(1e-5, gamma_beta.max() + 1.0e-4, 128)

        # Compute cumulative distribution
        hist_data = np.array([var[gamma_beta > gb].sum() for gb in gbs])

        return gbs, hist_data

    def _add_power_law_fit(self, gbs: np.ndarray, hist_data: np.ndarray) -> None:
        """Add power law fit to histogram"""
        from ....functional.helpers import find_nearest

        # Calculate ratios for power law determination
        E_seg_rat = hist_data[1:] / hist_data[:-1]
        gb_seg_rat = gbs[1:] / gbs[:-1]
        E_seg_rat[E_seg_rat == 0] = 1

        # Find power law region
        slope = (hist_data[1:] - hist_data[:-1]) / (gbs[1:] - gbs[:-1])
        power_law_idx = np.argmin(slope)
        up_min = find_nearest(gbs, 2 * gbs[power_law_idx:][0])[0]
        upower = gbs[up_min:]

        # Calculate power law index
        power_law_range = slice(up_min, np.argmin(E_seg_rat > 0.8))
        epower_law_seg = E_seg_rat[power_law_range]
        gbpower_law_seg = gb_seg_rat[power_law_range]

        alpha = 1.0 - np.mean(np.log10(epower_law_seg) / np.log10(gbpower_law_seg))
        E_0 = hist_data[up_min] * upower[0] ** (alpha - 1)

        # Plot the fit
        self.ax.plot(
            upower, E_0 * upower ** (-(alpha - 1)), "--", label=f"α = {alpha:.2f}"
        )

    def _update_limits(self, gbs: np.ndarray, hist_data: np.ndarray) -> None:
        """Update axis limits based on data"""
        # Check if user defined limits
        xlims = self.state.config.get("style", {}).get("xlims", (None, None))
        ylims = self.state.config.get("style", {}).get("ylims", (None, None))

        # Set auto x-limits if not user-defined
        if not any(xlims):
            self.ax.set_xlim(gbs.min() * 0.8, gbs.max() * 1.5)
        else:
            self.ax.set_xlim(xlims)

        # Set auto y-limits if not user-defined
        if not any(ylims):
            if hist_data.min() <= 0:
                self.ax.set_ylim(None, hist_data.max() * 1.5)
            else:
                self.ax.set_ylim(hist_data.min() * 0.9, hist_data.max() * 1.5)
        else:
            self.ax.set_ylim(ylims)

    def update(self, props: dict[str, Any]) -> None:
        """Update component properties"""
        super().update(props)

        # Apply updates to histogram
