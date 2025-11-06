"""
Animation controller for the visualization system.

This module provides animation functionality for visualizations.
"""

from typing import Any, Callable, Optional, Sequence

from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from simbi.tools.visualization.core.config import StyleConfig

from ..components.interface import Component
from ..core.state import PlotState, VisualizationState


class AnimationController:
    """Controls animation of visualization components."""

    def __init__(
        self,
        state: VisualizationState,
        components: Sequence[Component],
    ) -> None:
        """
        Initialize the animation controller.

        Args:
            state: Visualization state
            components: Visualization components to animate
        """
        self.state = state
        self.components = components
        self.animation = None
        self.fig = None
        self._frame_handler = None

    def initialize(self, fig: Figure) -> None:
        """
        Initialize the animation controller with a figure.

        Args:
            fig: Matplotlib figure to animate
        """
        self.fig = fig

        # Update animation state
        self.state.animation.playing = False
        self.state.animation.frame_index = 0

        # Set the state to initialized
        self.state.plot_state = PlotState.INITIALIZED

    def update_frame(self, frame_idx: int) -> tuple[Any, ...]:
        """
        Update all components for the given frame index.

        Args:
            frame_idx: Frame index to update

        Returns:
            Tuple of artists that were updated
        """
        self.state.animation.frame_index = frame_idx
        file_path = self.state.animation.data_files[frame_idx]

        # Load data for this frame
        if self._frame_handler:
            plot_data = self._frame_handler(file_path)
            self.state.update_data(plot_data)

        # Collect artists from all components
        artists = []

        for component in self.components:
            component.cleanup()

        # Update each component
        for component in self.components:
            if not self.state.data:
                continue
            frame_element = component.render(self.state.data, self.style)
            if frame_element:
                # Handle different return types
                if isinstance(frame_element, (list, tuple)):
                    artists.extend(frame_element)
                else:
                    artists.append(frame_element)

        # Return tuple of artists for blitting
        return tuple(artists)

    def animate(
        self,
        files: Sequence[str],
        style: StyleConfig,
        interval: int = 33,
        blit: bool = False,
        frame_handler: Optional[Callable] = None,
    ) -> FuncAnimation:
        """
        Create animation across all components.

        Args:
            files: Sequence of file paths to load for each frame
            interval: Time interval between frames in milliseconds
            blit: Whether to use blitting optimization
            frame_handler: Optional function to handle frame loading

        Returns:
            Matplotlib FuncAnimation object
        """
        if not self.fig:
            raise ValueError(
                "Animation controller not initialized. Call initialize() first."
            )

        if not self.components:
            raise ValueError("No components to animate")

        self.state.set_animation_files(files)
        self._frame_handler = frame_handler
        self.style = style

        # Create animation
        self.animation = FuncAnimation(
            self.fig,
            self.update_frame,
            frames=range(len(files)),
            interval=interval,
            blit=blit,
        )

        # Update state
        self.state.animation.playing = True
        self.state.animation.frame_rate = int(1000 / interval)
        self.state.plot_state = PlotState.ANIMATING

        return self.animation

    def save(
        self,
        filename: str,
        fps: int = 30,
        dpi: int = 300,
        **kwargs,
    ) -> None:
        """
        Save the animation to a file.

        Args:
            filename: Output file path
            writer: Animation writer ('ffmpeg', 'pillow', or a writer instance)
            fps: Frames per second
            dpi: Resolution in dots per inch
            **kwargs: Additional arguments for the writer
        """
        if not self.animation:
            raise ValueError("No animation to save")

        output_str = filename.replace("-", "_")
        extension = ""
        if not output_str.endswith(".mp4"):
            extension += ".mp4"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.description]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task(
                "[green] Saving animation...",
                total=self.state.animation.total_frames,
            )

            def prog(frame: int, total_frames: int) -> None:
                progress.update(task_id, advance=1)

            self.animation.save(
                output_str + extension,
                dpi=dpi,
                progress_callback=prog,
            )
        print(f"File saved as {output_str + extension}!")
