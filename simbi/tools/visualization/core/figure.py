from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure

from simbi.tools.visualization.components.multidim import MultidimPlotComponent
from simbi.tools.visualization.components.temporal import TemporalPlotComponent

from ..animation import AnimationController
from ..components.interface import Component
from ..core.types import PlotData
from .config import VisualizationConfig
from .state import VisualizationState


class Figure:
    """Main visualization container"""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.fig: Optional[MplFigure] = None
        self.axes: dict[str, Axes] = {}
        self.components: list[Component] = []
        self._current_data: Optional[PlotData] = None
        self._time_series: Optional[list[PlotData]] = None
        self.animation: Optional[AnimationController] = None

    def create_figure(self) -> None:
        """Create the matplotlib figure and axes"""
        figsize = self.config.style.fig_size
        self.fig, ax = plt.subplots(figsize=figsize)
        self.axes["main"] = ax

    def add_component(self, component: Component) -> None:
        """Add a visualization component"""
        self.components.append(component)

        if not component.initialized:
            component.initialize(self.fig, self.axes["main"])

    def load_data(self, data: PlotData) -> None:
        """Load data for rendering"""
        self._current_data = data

    def load_time_series(self, series: list[PlotData]) -> None:
        self._time_series = series

    def render(self) -> None:
        """Render all components"""
        if not self.fig:
            self.create_figure()
            if self.fig is None:
                raise RuntimeError("Failed to create figure")

            for component in self.components:
                if not component.initialized:
                    component.initialize(self.fig, self.axes["main"])

        if not self._current_data:
            raise ValueError(
                "No data loaded. Call load_data() or load_time_series() first."
            )

        for component in self.components:
            component.render(
                self._current_data,
                self.config.style,
            )

    def save(self, filename: str) -> None:
        """Save figure to file"""
        if not self.fig:
            raise RuntimeError("Figure not initialized. Call render() first.")

        if self.animation:
            self.animation.save(filename=filename, dpi=self.config.style.dpi)
        else:
            output_str = filename.replace("-", "_")
            extension = ""
            for comp in self.components:
                if isinstance(comp, MultidimPlotComponent):
                    extension += ".png"
                    break
                extension += ".pdf"

            output_str += extension
            self.fig.savefig(output_str, bbox_inches="tight")
            print(f"Saved figure to {filename}")

    def show(self) -> None:
        """Display the figure"""
        plt.show()

    def clear(self) -> None:
        """Clear the figure and components"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.axes = {}
        self.components = []
        self._current_data = None

    def tight_layout(self) -> None:
        """Adjust layout to prevent overlap"""
        if self.fig and not any(
            ax.name == "polar" for ax in self.fig.get_axes()
        ):
            self.fig.tight_layout()

    def get_frame(self, file_path: str) -> PlotData:
        """Placeholder for frame handler function"""
        from ..pipeline import create_plot_data, load_data

        data = load_data(file_path)
        return create_plot_data(data, self.config.plot.fields, self.config)

    def animate(
        self,
        files: Sequence[str],
        interval: int = 33,
    ) -> FuncAnimation:
        """Create animation from files"""
        if not self.fig:
            self.create_figure()
            self.load_data(self.get_frame(files[0]))
            if self.fig is None:
                raise RuntimeError("Failed to create figure")

            for component in self.components:
                if not component.initialized:
                    component.initialize(self.fig, self.axes["main"])

        state = VisualizationState(data=self._current_data)
        controller = AnimationController(state, self.components)
        controller.initialize(self.fig)
        self.animation = controller

        return controller.animate(
            files,
            style=self.config.style,
            interval=interval,
            frame_handler=self.get_frame,
        )
