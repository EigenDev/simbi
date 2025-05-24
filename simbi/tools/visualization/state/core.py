from dataclasses import dataclass, field
from typing import Any, Sequence, Optional
import numpy as np


@dataclass
class SimulationData:
    """Container for simulation data"""

    fields: dict[str, np.ndarray]
    setup: dict[str, Any]
    mesh: dict[str, np.ndarray]
    immersed_bodies: Optional[dict[str, Any]] = None


@dataclass
class VisualizationState:
    """Central state container for visualization components"""

    # Input data
    data: Optional[SimulationData] = None
    data_files: Sequence[str] = field(default_factory=list)
    current_frame: int = 0

    # Plot configuration
    config: dict[str, Any] = field(default_factory=dict)

    # Generated visualization elements
    plot_elements: dict[str, Any] = field(default_factory=dict)

    # Animation state
    is_animating: bool = False
    animation_progress: float = 0.0

    def update_data(self, new_data: SimulationData) -> None:
        """Update the data state"""
        self.data = new_data
        # Trigger data-dependent recalculations

    def advance_frame(self) -> None:
        """Move to the next frame"""
        if self.current_frame < len(self.data_files) - 1:
            self.current_frame += 1
            # Load new data if needed
