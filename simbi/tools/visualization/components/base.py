from abc import ABC, abstractmethod
from typing import Any
import matplotlib.pyplot as plt
from ..state.core import VisualizationState
from ..styling import ThemeManager
from ..styling import axis_formatter


class Component(ABC):
    """Base class for all visualization components"""

    def __init__(self, state: VisualizationState, id: str):
        self.state = state
        self.id = id
        self.is_initialized = False
        self.props = {}

        # Theme name to use for this component (can be overridden)
        self.theme_name = None

    def initialize(self, fig: plt.Figure, ax: plt.Axes) -> None:
        """Initialize the component with a figure and axes"""
        self.fig = fig
        self.ax = ax
        self.is_initialized = True

        # Apply theme styling to axis
        self.apply_theme()

        # Setup component-specific resources
        self.setup()

    def apply_theme(self) -> None:
        """Apply theme styling to this component's axis"""
        if hasattr(self, "ax"):
            # Check if polar axis
            if hasattr(self.ax, "name") and self.ax.name == "polar":
                ThemeManager.style_polar_axis(self.ax, self.theme_name)
            else:
                ThemeManager.style_axis(self.ax, self.theme_name)

    def format_axis(self) -> None:
        """Format axis based on data and component type"""
        if not hasattr(self, "ax") or not self.state.data:
            return

        # Get basic information needed for formatting
        # plot_type = self.state.config["plot"]["plot_type"]
        field = self.props["field"]

        # Get field label information
        if field:
            from ...utility import get_field_str

            field_info = get_field_str(field)
        else:
            field_info = ""

        # Apply formatting based on axis type
        if hasattr(self.ax, "name") and self.ax.name == "polar":
            pass  # [TODO] implement specific polar formatting if needed
        else:
            # Use formatter for Cartesian axes
            axis_formatter.format_cartesian_axis(
                self.ax, self.state.data.setup, self.state.config, field_info
            )

        # self.ax.set_aspect("equal")

    @abstractmethod
    def setup(self) -> None:
        """Setup component-specific resources"""
        pass

    @abstractmethod
    def render(self) -> Any:
        """Render the component based on current state"""
        pass

    @abstractmethod
    def update(self, props: dict[str, Any]) -> None:
        """Update the component with new properties"""
        # Update theme if specified
        if "theme" in props:
            self.theme_name = props["theme"]
            self.apply_theme()

        self.props.update(props)

    def cleanup(self) -> None:
        """Release resources when component is removed"""
        pass
