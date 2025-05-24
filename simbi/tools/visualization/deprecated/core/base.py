import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Optional, Any, List
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.axes import Axes

try:
    import cmasher as cmr
except ImportError:
    cmr = None


class BasePlotter(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.frames: Sequence[Any] = []
        self.fig: Optional[Figure] = None
        self.axes: Optional[Sequence[Axes]] = None

    @abstractmethod
    def create_figure(self) -> None:
        """Create and setup figure"""
        pass

    @abstractmethod
    def plot(self) -> None:
        """Main plotting method"""
        pass

    def setup(self) -> None:
        """Setup plotting environment"""
        self.create_figure()
        if not self.fig or not self.axes:
            raise RuntimeError("Figure and axes must be created")

    def save(self) -> None:
        """Save figure or animation"""
        path = self.config["plot"].save_as
        if not self.fig:
            raise RuntimeError("No figure to save")

        self.fig.tight_layout()
        if hasattr(self, "animation"):
            self.animation.save(
                f"{path}.mp4",
                dpi=self.config["style"].dpi,
                progress_callback=lambda i, n: print(
                    f"Saving frame {i} of {n}", end="\r"
                ),
            )
        else:
            if self.config["plot"].plot_type == "multidim":
                path = path + ".png"
            else:
                path = path + ".pdf"
            self.fig.savefig(
                path,
                dpi=self.config["style"].dpi,
                bbox_inches="tight",
                transparent=self.config["style"].transparent,
            )

    def show(self) -> None:
        """Display plot"""
        if not self.fig:
            raise RuntimeError("No figure to display")
        self.fig.tight_layout()
        plt.show()

    def __enter__(self):
        """Context manager entry"""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        plt.close(self.fig)
