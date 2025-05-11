from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from ..core.base import BasePlotter
from ..core.mixins import DataHandlerMixin, AnimationMixin, CoordinatesMixin
from ..utils.formatting import PlotFormatter
from ..utils.io import DataManager
from typing import Sequence, Any
from numpy.typing import NDArray


class LinePlotter(BasePlotter, DataHandlerMixin, AnimationMixin, CoordinatesMixin):
    """Line plot implementation with mixins"""

    def __init__(self, parser):
        super().__init__(parser)
        self.formatter = PlotFormatter()
        self.data_manager = DataManager(
            self.config["plot"].files, movie_mode=self.config["plot"].kind == "movie"
        )

    def create_figure(self) -> None:
        self.fig, self.axes = plt.subplots(
            self.config["plot"].nplots,
            1,
            figsize=self.config["style"].fig_dims,
            # sharex=True,
        )
        if not isinstance(self.axes, Sequence):
            self.axes = [self.axes]

    def plot(self):
        """Main plotting method"""
        if self.config["style"].labels:
            labels = self.config["style"].labels
        else:
            labels = [None] * len(self.config["plot"].fields)
        label_cycle = cycle(labels)
        for ax in self.axes:
            for data in self.data_manager.iter_files():
                for field_idx, field in enumerate(self.config["plot"].fields):
                    # Use DataHandlerMixin
                    var = self.get_variable(data.fields, field)
                    label = self.get_label(field, next(label_cycle))

                    # Use CoordinatesMixin
                    x, indices = self.transform_coordinates(data.mesh, data.setup)
                    if self.config["multidim"].slice_along:
                        sliced_var = self.get_slice_data(var, data.mesh, data.setup)
                        self._plot_slice(ax, x, sliced_var, field, label)
                    else:
                        self._plot_line(ax, x, var, field, label)

            self.formatter.set_axes_properties(self.fig, ax, data.setup, self.config)
            self.formatter.setup_axis_style(
                ax, xlim=self.config["style"].xlims, ylim=self.config["style"].ylims
            )

    def _plot_line(
        self,
        ax: plt.Axes,
        x: NDArray[np.floating[Any]],
        var: NDArray[np.floating[Any]],
        field: str,
        label: str | None = None,
    ):
        """Plot 1D line using transformed coordinates"""
        yvar = var if self.config["plot"].ndim == 1 else var[:, 0]
        line = ax.plot(x, yvar, label=label)[0]
        self.frames.append(line)

    def _plot_slice(self, ax, x, var, field: str, label: str | None = None):
        """Plot 1D slice using transformed coordinates"""
        line = ax.plot(x, var.flat, label=label)[0]
        self.frames.append(line)
