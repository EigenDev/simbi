import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional, Any
from numpy.typing import NDArray
from simbi.tools.visualization.core.mixins import (
    AnimationMixin,
    CoordinatesMixin,
    DataHandlerMixin,
)
from ..utils.io import DataManager
from ..utils.formatting import PlotFormatter
from ..core.base import BasePlotter
from ....functional.helpers import calc_cell_volume
from ... import utility as util
from ..core.constants import DERIVED, FIELD_ALIASES


@dataclass
class AccretionTimeSeriesData:
    """Container for accretion time series data"""

    times: list[float]
    # body ID -> list of accreted masses over time
    # for both accretion rate and mass, the list is ordered by time
    # and the body ID is the key
    accreted_mass: dict[str, list[float]]
    accretion_rate: dict[str, list[float]]

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

    times: list[float]
    values: list[float]
    weight_type: str
    field: str

    @property
    def array(self):
        return np.array(self.times), np.array(self.values)


class TemporalPlotter(BasePlotter, DataHandlerMixin, AnimationMixin, CoordinatesMixin):
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
        self, mesh: dict[str, NDArray[np.floating[Any]]], setup: dict[str, Any]
    ) -> np.ndarray:
        """Calculate cell volumes based on dimension"""
        dV = calc_cell_volume(
            coords=[mesh[f"x{i + 1}v"] for i in range(self.config["plot"].ndim)],
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
        plot_weight = self.config["plot"].weight

        for data in self.data_manager.iter_files():
            # Get variable and weights
            var = self.get_variable(data.fields, self.config["plot"].fields[0])

            if plot_weight is not None:
                weights = self.get_variable(data.fields, self.config["plot"].weight)
            else:
                weights = None

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

    def _compute_accretion_time_series(self) -> AccretionTimeSeriesData:
        """Compute time series for accretion data"""
        times = []
        # body ID -> list of accreted masses
        accreted_mass = {}
        accretion_rate = {}

        for data in self.data_manager.iter_files():
            times.append(data.setup["time"])

            # process immersed bodies data
            if data.immersed_bodies:
                for body_id, body_data in data.immersed_bodies.items():
                    # check if body is an accretor
                    if "total_accreted_mass" in body_data:
                        # init lists if this is the first time we're seeing this body
                        if body_id not in accreted_mass:
                            accreted_mass[body_id] = []
                            accretion_rate[body_id] = []

                        accreted_mass[body_id].append(body_data["total_accreted_mass"])
                        accretion_rate[body_id].append(body_data["accretion_rate"])

        return AccretionTimeSeriesData(times, accreted_mass, accretion_rate)

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

    def _plot_accretion_data(self, field_type: str):
        """Plot accretion data for immersed bodies"""
        # compute accretion time series
        accretion_series = self._compute_accretion_time_series()
        times = accretion_series.array

        if self.config["style"].orbital_params:
            # get orbital period
            separation = self.config["style"].orbital_params.get("separation")
            total_mass = self.config["style"].orbital_params.get("mass")
            if separation is None or total_mass is None:
                raise ValueError("Separation or total mass not provided in config.")

            orbital_period = (
                2.0 * np.pi * np.sqrt(float(separation) ** 3 / float(total_mass))
            )
            # convert times to orbital periods
            times = np.array(times) / orbital_period

        # get labels from config or generate defaults
        labels = self.config["style"].labels or []
        # plot data for each body
        for i, body_id in enumerate(accretion_series.body_ids):
            body_data = accretion_series.get_body_data(body_id, field_type)

            # use provided label or generate default
            label = labels[i] if i < len(labels) else f"Body {body_id.split('_')[-1]}"

            # plot this body's data
            self.frames.append(
                self.axes.plot(times, body_data, label=label, alpha=1.0, marker="o")[0]
            )

        # set title based on what we're plotting
        title = (
            "Total Accreted Mass vs Time"
            if field_type == "accreted_mass"
            else "Accretion Rate vs Time"
        )
        self.axes.set_title(title)

        # add legend if multiple bodies
        if len(accretion_series.body_ids) > 1:
            self.axes.legend()

        metadata = self.data_manager.get_metadata()
        # apply formatter settings
        self.formatter.set_axes_properties(self.fig, self.axes, metadata, self.config)
        self.formatter.setup_axis_style(self.axes)

    def plot(self):
        """Main plotting method"""
        # check if we're plotting regular field data or accretion data
        field = self.config["plot"].fields[0]
        if field in FIELD_ALIASES:
            field = FIELD_ALIASES[field]

        # check if we're plotting accretion data
        if field in ["accreted_mass", "accretion_rate"]:
            self._plot_accretion_data(field)
        else:
            series = self._compute_time_series()
            times, data = series.array

            # plot main data
            label = self.config["style"].labels or None
            self.frames.append(self.axes.plot(times, data, label=label, alpha=1.0)[0])
            metadata = self.data_manager.get_metadata()
            self.formatter.set_axes_properties(
                self.fig, self.axes, metadata, self.config
            )
            self.formatter.setup_axis_style(self.axes)
