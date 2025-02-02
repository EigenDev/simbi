import matplotlib.pyplot as plt
from ..core.base import BasePlotter
from ..core.mixins import DataHandlerMixin, AnimationMixin, CoordinatesMixin
from ..utils.formatting import PlotFormatter
from ..utils.io import DataManager
from typing import Sequence
from ...utility import get_field_str


class LinePlotter(BasePlotter, DataHandlerMixin, AnimationMixin, CoordinatesMixin):
    """Line plot implementation with mixins"""

    def __init__(self, parser):
        super().__init__(parser)
        self.formatter = PlotFormatter()
        self.data_manager = DataManager(
            self.config["plot"].files, 
            movie_mode=self.config["plot"].kind == "movie"
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
        for ax in self.axes:
            for data in self.data_manager.iter_files():
                for field in self.config["plot"].fields:
                    # Use DataHandlerMixin
                    var = self.get_variable(data.fields, field)

                    # Use CoordinatesMixin
                    x, indices = self.transform_coordinates(data.mesh, data.setup)
                    if self.config['multidim'].slice_along:
                        sliced_var = self.get_slice_data(var, data.mesh, data.setup)
                        self._plot_slice(ax, x, var)
                    else:
                        self._plot_line(ax, x, var, field)

            self.formatter.set_axes_properties(self.fig, ax, data.setup, self.config)
            self.formatter.setup_axis_style(ax)

    def _plot_line(self, ax, x, var, field):
        """Plot 1D line using transformed coordinates"""
        yvar = var if self.config["plot"].ndim == 1 else var[:, 0]
        field_string = get_field_str(field)
        line = ax.plot(x, yvar, label=field_string)[0]
        self.frames.append(line)

    def _plot_slice(self, ax, x, var):
        """Plot 1D slice using transformed coordinates"""
        line = ax.plot(x, var.flat)[0]
        self.frames.append(line)
