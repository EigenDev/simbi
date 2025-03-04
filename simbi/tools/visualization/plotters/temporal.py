import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional, Any
from numpy.typing import NDArray
from ..utils.io import DataManager
from ..utils.formatting import PlotFormatter
from ..core.base import BasePlotter
from ....functional.helpers import calc_cell_volume
from ... import utility as util
from ..core.constants import DERIVED


@dataclass
class TimeSeriesData:
    """Container for time series data"""

    times: list[float]
    values: list[float]
    weight_type: str
    field: str

    @property
    def array(self):
        return np.array(self.times), np.array(self.values)


class TemporalPlotter(BasePlotter):
    """Plots mean quantities vs time"""

    def __init__(self, parser: ArgumentParser) -> None:
        super().__init__(parser)
        self.data_manager = DataManager(
            self.config["plot"].files, movie_mode=self.config["plot"].kind == "movie"
        )
        self.formatter = PlotFormatter()

    def create_figure(self):
        """Create figure with appropriate styling"""
        self.fig, self.axes = plt.subplots(1, 1, figsize=self.config["style"].fig_dims)

    def _calc_volume(
        self, mesh: dict[str, NDArray[np.float64]], setup: dict[str, Any]
    ) -> np.ndarray:
        """Calculate cell volumes based on dimension"""
        dV = calc_cell_volume(
            coords=[mesh[f"x{i+1}v"] for i in range(self.config["plot"].ndim)],
            coord_system=setup["coord_system"],
            vertices=True,
        )
        return dV

    def _compute_weighted_mean(
        self, var: np.ndarray, weights: np.ndarray, dV: np.ndarray
    ) -> float:
        """Compute weighted mean of variable"""
        if self.config["plot"].weight == self.config["plot"].fields[0]:
            return np.max(var)
        return np.sum(weights * var * dV) / np.sum(weights * dV)

    def _compute_time_series(self) -> TimeSeriesData:
        """Compute time series for given dataset"""
        times, values = [], []

        for data in self.data_manager.iter_files():
            # Get variable and weights
            var = (
                util.prims2var(data.fields, self.config["plot"].fields[0])
                if self.config["plot"].fields[0] in DERIVED
                else data.fields[self.config["plot"].fields[0]]
            )

            weights = (
                util.prims2var(data.fields, self.config["plot"].weight)
                if self.config["plot"].weight != self.config["plot"].fields[0]
                else None
            )

            # Calculate weighted mean
            if weights is not None:
                dV = self._calc_volume(data.mesh, data.setup)
                value = self._compute_weighted_mean(var, weights, dV)
            else:
                value = np.max(var)

            times.append(data.setup["time"])
            values.append(value)

        return TimeSeriesData(
            times, values, self.config["plot"].weight, self.config["plot"].fields[0]
        )

    def _fit_power_law(
        self, times: np.ndarray, data: np.ndarray, t_break: Optional[float] = None
    ) -> dict:
        """Fit power law to data"""
        if not t_break:
            return {
                "time": times,
                "fit": data[0] * (times / times[0]) ** (-3 / 2),
                "label": r"$\propto t^{-3/2}$",
            }

        idx = int(np.argmin(np.abs(times - t_break)))
        t_ref = times[idx:]
        return {
            "time": t_ref,
            "fits": [
                (data[idx] * np.exp(1 - t_ref / t_ref[0]), r"$\propto \exp(-t)$"),
                (data[idx] * (t_ref / t_ref[0]) ** (-3), r"$\propto t^{-3}$"),
            ],
        }

    def plot(self):
        """Main plotting method"""
        # Compute time series
        series = self._compute_time_series()
        times, data = series.array

        # Plot main data
        label = self.config["style"].labels or None
        self.frames.append(self.axes.plot(times, data, label=label, alpha=1.0)[0])
        self.formatter.set_axes_properties(
            self.fig, self.axes, {"dimensions": 1}, self.config
        )
        self.formatter.setup_axis_style(self.axes)
        # Add power law fits if requested
        # if (
        #     self.pictorial
        #     and series.field in ["gamma_beta", "u1", "u"]
        #     and key == len(self.flist.keys()) - 1
        # ):

        #     fit = self._fit_power_law(times, data, self.break_time)

        #     if self.break_time:
        #         for fit_data, label in fit["fits"]:
        #             self.ax.plot(
        #                 fit["time"],
        #                 fit_data,
        #                 label=label,
        #                 color="grey",
        #                 linestyle="--",
        #             )
        #     else:
        #         self.ax.plot(
        #             fit["time"],
        #             fit["fit"],
        #             label=fit["label"],
        #             color="grey",
        #             linestyle=":",
        #         )
