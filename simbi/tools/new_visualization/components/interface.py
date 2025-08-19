"""Component interface definitions."""

from typing import Protocol, TypeVar, Generic, Any
from pydantic import BaseModel
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..core.types import PlotData


class ComponentProps(BaseModel):
    """Base class for component properties."""

    model_config = {
        "frozen": True,  # Make instances immutable
    }


P = TypeVar("P", bound=ComponentProps)


class Component(Protocol, Generic[P]):
    """Component protocol defining the interface for all visualization components."""

    props: P

    def initialize(self, fig: Figure, ax: Axes) -> None:
        """Initialize the component with figure and axes."""
        ...

    def update(self, props: P) -> None:
        """Update component properties."""
        ...

    def render(self, data: PlotData) -> Any:
        """Render the component with the given data."""
        ...

    def cleanup(self) -> None:
        """Clean up resources."""
        ...
