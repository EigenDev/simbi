"""Component interface definitions."""

from typing import Any, Generic, Protocol, TypeVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel

from simbi.tools.visualization.core.config import StyleConfig

from ..core.types import PlotData


class ComponentProps(BaseModel):
    """Base class for component properties."""

    model_config = {
        "frozen": True,
        "arbitrary_types_allowed": True,
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

    def render(self, data: PlotData, style: StyleConfig) -> Any:
        """Render the component with the given data."""
        ...

    def cleanup(self) -> None:
        """Clean up resources."""
        ...

    @property
    def initialized(self) -> bool:
        """Check if the component has been initialized."""
        ...
