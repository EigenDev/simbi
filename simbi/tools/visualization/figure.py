from .styling import ThemeManager
from typing import Any, Sequence, Type
import matplotlib.pyplot as plt
from .state.core import VisualizationState
from .components.base import Component
from .pipeline.core import DataPipeline
from .animation.controller import AnimationController
from .bridge import SimbiDataBridge


class Figure:
    """Main visualization container"""

    def __init__(
        self,
        config: dict[str, Any],
        nfiles: int = 1,
        nfields: int = 1,
        theme: str = "default",
    ):
        self.config = config
        self.state = VisualizationState(config=config)
        self.pipeline = DataPipeline(self.state)
        self.components = []
        self.fig = None
        self.axes = {}
        self.bridge = SimbiDataBridge(self.state)

        # Set and apply theme :D (!)
        self.theme = theme
        ThemeManager.set_theme(self.theme, nfiles=nfiles, nfields=nfields)

    def create_figure(self) -> None:
        """Create the matplotlib figure and axes"""
        figsize = self.config.get("style", {}).get("fig_dims", (10, 6))

        # Check if polar projection is needed
        is_cartesian = True  # Default
        if self.state.data:
            is_cartesian = self.state.data.setup.get("is_cartesian", True)

        # Create figure with appropriate projection
        if is_cartesian:
            self.fig, ax = plt.subplots(figsize=figsize)
        else:
            self.fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={"projection": "polar"}
            )

        self.axes["main"] = ax

    def add(
        self, component_class: Type[Component], component_id: str, **props
    ) -> Component:
        """Add a visualization component"""
        component = component_class(self.state, component_id)
        component.update(props)
        self.components.append(component)
        return component

    def initialize(self) -> None:
        """Initialize all components"""
        if not self.fig:
            self.create_figure()

        for component in self.components:
            # Get the appropriate axes for this component
            ax = self.axes.get("main")
            component.initialize(self.fig, ax)

    def load_data(self, file_path: str) -> None:
        """Load data from file into state"""
        data = self.bridge.load_file(file_path)
        self.state.update_data(data)

    def render(self) -> None:
        """Render all components"""
        if not self.fig:
            self.initialize()

        # Render each component
        for component in self.components:
            component.render()

    def animate(self, files: Sequence[str], interval: int = 33) -> None:
        """Create animation from files"""
        self.state.data_files = files
        self.load_data(files[0])  # Load first frame

        # Initialize if needed
        if not self.fig:
            self.initialize()

        # Create animation controller
        controller = AnimationController(self.state, self.components, self.bridge)
        self.animation = controller.animate(interval=interval)

    def save(self, filename: str) -> None:
        """Save figure or animation with progress feedback"""
        if not self.fig:
            raise RuntimeError("Figure not initialized. Call render() first.")

        dpi = self.config["style"]["dpi"]
        if hasattr(self, "animation"):
            print(f"Saving animation to {filename}.mp4...")
            print(f"This may take a while for {len(self.state.data_files)} frames.")

            # Create progress bar callback
            from tqdm import tqdm

            progress_bar = tqdm(total=len(self.state.data_files))

            def update_progress(current, total):
                progress_bar.update(1)
                return

            self.animation.save(
                f"{filename}.mp4", dpi=dpi, progress_callback=update_progress
            )
            progress_bar.close()
            print(f"Successfully saved animation to {filename}.mp4")
        else:
            self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure to {filename}")

    def set_theme(self, theme_name: str) -> None:
        """Change the figure's theme"""
        if ThemeManager.set_theme(theme_name):
            self.theme = theme_name

            # Apply to all components
            for component in self.components:
                component.theme_name = theme_name
                component.apply_theme()

            # Re-render (rerender (?))
            self.render()

    def tight_layout(self) -> None:
        """Adjust layout to prevent overlap"""
        if self.fig:
            self.fig.tight_layout()

    def show(self) -> None:
        """Display the figure or animation"""
        plt.show()
