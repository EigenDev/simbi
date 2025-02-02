import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from ..core.mixins import DataHandlerMixin, AnimationMixin, CoordinatesMixin
from typing import Optional, Tuple, Any
from numpy.typing import NDArray
from ..core.base import BasePlotter
from ..utils.io import DataManager
from ..utils.formatting import PlotFormatter
from ....detail.helpers import find_nearest, calc_cell_volume
from ... import utility as util


class HistogramPlotter(BasePlotter, DataHandlerMixin, AnimationMixin, CoordinatesMixin):
    """Histogram plotter for distribution functions"""
    
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser)
        self.data_manager = DataManager(
            self.config["plot"].files, movie_mode=self.config["plot"].kind == "movie"
        )
        self.formatter = PlotFormatter()

    def create_figure(self):
        """Create figure with appropriate styling"""
        self.fig, self.axes = plt.subplots(
            self.config["plot"].nplots,
            1,
            figsize=self.config["style"].fig_dims,
            sharex=True,
        )
        if self.config["plot"].nplots == 1:
            self.axes = [self.axes]

        self._setup_axes()

    def _setup_axes(self):
        """Configure axes style"""
        for ax in self.axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
            ax.set_xticklabels(["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])

    def _calc_volume(
        self, mesh: dict[str, NDArray[np.float64]], setup: dict[str, Any]
    ) -> np.ndarray:
        """Calculate cell volumes"""
        dV = calc_cell_volume(
            coords=[mesh[f"x{i+1}v"] for i in range(self.config["plot"].ndim)],
            coord_system=setup["coord_system"],
            vertices=True,
        )
        return dV

    def _compute_variable(self, fields: dict, dV: np.ndarray) -> np.ndarray:
        """Compute variable to histogram"""
        if self.config["plot"].hist_type == "kinetic":
            mass = dV * fields["W"] * fields["rho"]
            var = (fields["W"] - 1.0) * mass * util.e_scale.value
        elif self.config["plot"].hist_type == "enthalpy":
            enthalpy = 1.0 + fields["ad_gamma"] * fields["p"] / (
                fields["rho"] * (fields["ad_gamma"] - 1.0)
            )
            var = (enthalpy - 1.0) * dV * util.e_scale.value
        elif self.config["plot"].hist_type == "mass":
            var = dV * fields["W"] * fields["rho"] * util.mass_scale.value
        else:
            edens_total = util.prims2var(fields, "energy")
            var = edens_total * dV * util.e_scale.value

        return var

    def _compute_histogram(
        self, fields: dict, var: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute histogram data"""
        gamma_beta = fields["gamma_beta"]
        gbs = np.geomspace(1e-5, gamma_beta.max() + 1.e-4, 128)

        # Compute cumulative distribution
        hist_data = np.array([var[gamma_beta > gb].sum() for gb in gbs])

        return gbs, hist_data

    def _fit_power_law(
        self, gbs: np.ndarray, hist_data: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Fit power law to histogram data"""
        if not self.powerfit:
            return None

        # Calculate ratios
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

        return upower, E_0, alpha

    def _plot_histogram(
        self,
        ax: plt.Axes,
        gbs: np.ndarray,
        hist_data: np.ndarray,
        label: Optional[str] = None,
    ):
        """Plot histogram data"""
        # if self.xfill_scale:
        #     util.fill_below_intersec(gbs, hist_data, self.xfill_scale, axis="x")
        # elif self.yfill_scale:
        #     util.fill_below_intersec(
        #         gbs, hist_data, self.yfill_scale * hist_data.max(), axis="y"
        #     )

        self.frames.append(
            ax.hist(
                gbs,
                bins=gbs,
                weights=hist_data,
                label=label,
                histtype="step",
                rwidth=1.0,
                linewidth=3.0,
            )
        )

    def plot(self):
        """Main plotting method"""
        for ax_idx, ax in enumerate(self.axes):
            for file_idx, data in enumerate(self.data_manager.iter_files()):

                # Handle multiple subplots
                if self.config["plot"].nplots > 1 and file_idx == len(self.flist) // 2:
                    ax = self.axes[ax_idx + 1]
                    ax_idx += 1

                # Calculate distributions
                dV = self._calc_volume(data.mesh, data.setup)
                var = self._compute_variable(data.fields, dV)
                gbs, hist_data = self._compute_histogram(data.fields, var)

                # Plot histogram
                label = None  # self.labels[file_idx] if self.labels else None
                self._plot_histogram(ax, gbs, hist_data, label)

                # Fit power law if requested
                if self.config["plot"].powerfit:
                    fit = self._fit_power_law(gbs, hist_data)
                    if fit:
                        upower, E_0, alpha = fit
                        ax.plot(
                            upower,
                            E_0 * upower ** (-(alpha - 1)),
                            "--",
                            label=f"α = {alpha:.2f}",
                        )

            # Set axis properties
            if not any(self.config["style"].xlims):
                xlims = (gbs.min()*0.8, gbs.max()*1.5)
            else:
                xlims = self.config["style"].xlims
            
            if not any(self.config["style"].ylims):
                ylims = (hist_data.min()*0.9, hist_data.max() * 1.5)
                if hist_data.min() <= 0:
                    ylims = (None, hist_data.max() * 1.5)
            else:
                ylims = self.config["style"].ylims
                
            self.formatter.set_axes_properties(self.fig, ax, data.setup, self.config)
            self.formatter.setup_axis_style(
                ax, xlim=xlims, ylim=ylims
            )

            # if self.legend:
            #     alpha = 0.0 if self.transparent else 1.0
            #     ax.legend(loc=self.legend_loc, framealpha=alpha)
