from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure
from matplotlib.axes import Axes
from .types import SimulationData
from .config import VisualizationConfig
from ..components.interface import Component


class Figure:
    """Main visualization container"""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.fig: Optional[MplFigure] = None
        self.axes: Dict[str, Axes] = {}
        self.components: List[Component] = []
        self._current_data: Optional[SimulationData] = None

    def create_figure(self) -> None:
        """Create the matplotlib figure and axes"""
        figsize = self.config.style.fig_size
        self.fig, ax = plt.subplots(figsize=figsize)
        self.axes["main"] = ax

    def add_component(self, component: Component) -> None:
        """Add a visualization component"""
        self.components.append(component)

        if self.fig and "main" in self.axes:
            component.initialize(self.fig, self.axes["main"])

    def load_data(self, data: SimulationData) -> None:
        """Load data for rendering"""
        self._current_data = data

    def render(self) -> None:
        """Render all components"""
        if not self.fig:
            self.create_figure()
            if self.fig is None:
                raise RuntimeError("Failed to create figure")

            for component in self.components:
                if not hasattr(component, "_initialized") or not component._initialized:
                    component.initialize(self.fig, self.axes["main"])

        if not self._current_data:
            raise ValueError("No data loaded. Call load_data() first.")

        for component in self.components:
            component.render(self._current_data)

    def save(self, filename: str) -> None:
        """Save figure to file"""
        if not self.fig:
            raise RuntimeError("Figure not initialized. Call render() first.")

        self.fig.savefig(filename, dpi=self.config.style.dpi, bbox_inches="tight")
        print(f"Saved figure to {filename}")

    def show(self) -> None:
        """Display the figure"""
        plt.show()
