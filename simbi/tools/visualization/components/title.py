from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from .base import Component


class TitleComponent(Component):
    """Handles title display and formatting"""

    def setup(self) -> None:
        """Initialize title"""
        self.title_artist = (
            self.ax.set_title("") if self.props.get("ax_title", True) else None
        )
        self.suptitle_artist = (
            self.fig.suptitle("") if self.props.get("fig_title", False) else None
        )

    def render(self) -> Any:
        """Update title with current data"""
        if not self.state.data:
            return self.title_artist or self.suptitle_artist

        # Get setup info
        setup = self.state.data.setup
        time = setup.get("time", 0.0)

        # Apply time scaling if using orbital parameters
        time_unit = ""
        if self.state.config["style"]["orbital_params"]:
            import math

            p = self.state.config["style"]["orbital_params"]
            orbital_period = (
                2.0
                * math.pi
                * math.sqrt(
                    float(p.get("separation", 1)) ** 3 / float(p.get("mass", 1))
                )
            )
            time = setup["time"] / orbital_period
            time_unit = "orbit(s)"

        # Format title text
        setup_name = self.state.config["plot"].get("setup", "Simulation")
        title_text = f"{setup_name} t = {time:.2f} {time_unit}"

        # For animation, it's better to remove and recreate the title
        # rather than update an existing one
        if setup["is_cartesian"] or self.state.config["multidim"]["slice_along"]:
            # Remove old title if it exists
            # if self.title_artist:
            # self.title_artist.remove()

            # Create new title
            self.title_artist = self.ax.set_title(title_text)
            return self.title_artist
        else:
            # For polar plots, handle suptitle
            # if self.suptitle_artist:
            # self.suptitle_artist.remove()

            y_pos = 0.95 if setup.get("x2max", np.pi) == np.pi else 0.92
            self.suptitle_artist = self.fig.suptitle(title_text, y=y_pos)
            return self.suptitle_artist

    def update(self, props: dict[str, Any]) -> None:
        """Update component properties"""
        super().update(props)

        # Handle any title-specific property updates
        if hasattr(self, "title_artist") and self.title_artist:
            if "fontsize" in props:
                self.title_artist.set_fontsize(props["fontsize"])
            if "color" in props:
                self.title_artist.set_color(props["color"])
