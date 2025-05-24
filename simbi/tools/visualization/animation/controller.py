from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from typing import Sequence
from ..state.core import VisualizationState
from ..components.base import Component
from ..bridge import SimbiDataBridge


class AnimationController:
    """Controls animation of visualization components"""

    def __init__(
        self,
        state: VisualizationState,
        components: Sequence[Component],
        bridge: SimbiDataBridge,
    ):
        self.state = state
        self.components = components
        self.bridge = bridge
        self.animation = None

    def update_frame(self, frame: int) -> tuple:
        """Update all components for a new frame"""
        try:
            # Load data for this frame
            self.state.current_frame = frame
            data_file = self.state.data_files[frame]

            # Load the data
            data = self.bridge.load_file(data_file)
            self.state.update_data(data)

            # Clear any existing titles to prevent overlap
            # for ax in plt.gcf().axes:
            #     ax.set_title("")
            # if hasattr(plt.gcf(), "_suptitle"):
            #     plt.gcf()._suptitle.set_text("")

            # Update all components
            frames = []
            for component in self.components:
                if component.is_initialized:
                    frame_element = component.render()
                    if frame_element:
                        # If we get multiple artists (like list/tuple), add them all
                        if isinstance(frame_element, (list, tuple)):
                            frames.extend(frame_element)
                        else:
                            frames.append(frame_element)

            # Ensure figure updates by drawing
            self.components[0].fig.canvas.draw_idle()

            # Return all frame elements for blitting
            return tuple(frames)
        except Exception as e:
            print(f"Error updating frame {frame}: {e}")
            # Return empty tuple to prevent animation from crashing
            return ()

    def animate(self, frames: int = 0, interval: int = 33) -> FuncAnimation:
        """Create animation across all components"""
        if not self.components:
            raise ValueError("No components to animate")

        # Determine number of frames
        if frames == 0:
            frames = len(self.state.data_files)

        # Create animation
        self.animation = FuncAnimation(
            self.components[0].fig,  # Assume all components share the same figure
            self.update_frame,
            frames=range(frames),
            interval=interval,
            blit=False,
        )

        return self.animation
