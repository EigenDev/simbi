"""
State management for the visualization system.

This module provides state management functionality for tracking visualization state,
settings, and data across components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .types import PlotData


class PlotState(Enum):
    """Enumeration of plot states."""

    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    RENDERED = "rendered"
    ANIMATING = "animating"
    ERROR = "error"


@dataclass
class ViewportState:
    """State for viewport information."""

    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    locked: bool = False
    auto_scale: bool = True


@dataclass
class AnimationState:
    """State for animation information."""

    playing: bool = False
    frame_index: int = 0
    total_frames: int = 0
    frame_rate: int = 30
    loop: bool = False
    data_files: List[str] = field(default_factory=list)


@dataclass
class VisualizationState:
    """Central state container for visualization components."""

    # Overall plot state
    plot_state: PlotState = PlotState.UNINITIALIZED

    # Data state
    data: Optional[PlotData] = None
    data_files: Sequence[str] = field(default_factory=list)

    # Viewport state
    viewport: ViewportState = field(default_factory=ViewportState)

    # Animation state
    animation: AnimationState = field(default_factory=AnimationState)

    # Component state registry
    component_states: Dict[str, Any] = field(default_factory=dict)

    def update_data(self, new_data: PlotData) -> None:
        """Update the visualization data."""
        self.data = new_data

        # Auto-update viewport limits if not locked
        if not self.viewport.locked and self.viewport.auto_scale:
            self._update_viewport_from_data()

    def _update_viewport_from_data(self) -> None:
        """Update viewport limits based on current data."""
        if not self.data or not self.data.fields:
            return

        # Initialize with first field domain
        field = self.data.fields[0]

        # Update x limits if domain exists
        if field.domain and len(field.domain) > 0:
            x_domain = field.domain[0]
            self.viewport.x_min = float(np.min(x_domain))
            self.viewport.x_max = float(np.max(x_domain))

        # Update y limits if domain exists
        if field.domain and len(field.domain) > 1:
            y_domain = field.domain[1]
            self.viewport.y_min = float(np.min(y_domain))
            self.viewport.y_max = float(np.max(y_domain))

        # Update z limits if domain exists
        if field.domain and len(field.domain) > 2:
            z_domain = field.domain[2]
            self.viewport.z_min = float(np.min(z_domain))
            self.viewport.z_max = float(np.max(z_domain))

    def register_component_state(self, component_id: str, state: Any) -> None:
        """Register a component's state."""
        self.component_states[component_id] = state

    def get_component_state(self, component_id: str) -> Optional[Any]:
        """Get a component's state."""
        return self.component_states.get(component_id)

    def advance_frame(self) -> bool:
        """Advance to the next animation frame."""
        if not self.animation.playing or not self.animation.data_files:
            return False

        if self.animation.frame_index < self.animation.total_frames - 1:
            self.animation.frame_index += 1
            return True
        elif self.animation.loop:
            self.animation.frame_index = 0
            return True
        else:
            self.animation.playing = False
            return False

    def set_animation_files(self, files: Sequence[str]) -> None:
        """Set the animation files."""
        self.animation.data_files = list(files)
        self.animation.total_frames = len(files)
        self.animation.frame_index = 0

    def lock_viewport(self) -> None:
        """Lock the viewport to prevent auto-scaling."""
        self.viewport.locked = True

    def unlock_viewport(self) -> None:
        """Unlock the viewport to enable auto-scaling."""
        self.viewport.locked = False
