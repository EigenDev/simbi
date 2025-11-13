"""Component interface definitions."""

from typing import Any, Generic, Protocol, TypeVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel

from ..config import StyleConfig

# --- TypeVars ---
P = TypeVar("P", bound="ComponentProps")  # Generic for Props
D = TypeVar("D", contravariant=True)  # Generic for the data payload
# -----------------


class ComponentProps(BaseModel):
    """Base class for component properties."""

    model_config = {
        "frozen": True,
        "arbitrary_types_allowed": True,
    }


class Component(Protocol, Generic[P, D]):
    """
    Component protocol defining the interface for all visualization components.

    It is generic over:
     - P (Props): The pydantic model for styling.
     - D (Data): The data payload type the render() method expects.
    """

    props: P

    def initialize(self, fig: Figure, ax: Axes) -> None:
        """Initialize the component with figure and axes."""
        ...

    def update(self, props: P) -> None:
        """Update component properties."""
        ...

    def render(self, data: D, style: StyleConfig) -> Any:
        """Render the component with the given data."""
        ...

    def cleanup(self) -> None:
        """Clean up resources."""
        ...

    @property
    def initialized(self) -> bool:
        """Check if the component has been initialized."""
        ...
