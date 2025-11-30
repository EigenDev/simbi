import math
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from simbi.viz.components.coord_binning import (
    CoordinateProfileComponent,
)
from simbi.viz.components.time_series import TimeSeriesPlotComponent
from simbi.viz.utility import get_field_str, map_coordinate_label

from . import formatting
from .components import (
    LinePlotComponent,
    PolygonPlotComponent,
    QuadPlotComponent,
)
from .components.interface import Component
from .config import VisualizationConfig
from .types import CoordSystem, FieldData


class Figure:
    """Manages the matplotlib figure, axes, and components."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.fig: Optional[MplFigure] = None
        self.axes: dict[str, Axes] = {}
        self._components: list[tuple[Component, Any]] = []
        self.coord_system: CoordSystem = CoordSystem.CARTESIAN

    def add_component(self, component: Component, data: Any = None):
        """Adds a component and its associated data payload."""
        self._components.append((component, data))

    def render(self):
        """Renders all components and applies formatting."""
        if not self.fig or not self.axes:
            raise RuntimeError("Figure has not been prepared.")

        main_ax = self.axes["main"]

        # --- PRE-RENDER FORMATTING ---
        formatting.apply_scaling(main_ax, self.config.style)

        # --- RENDER DATA ---
        rendered_artists = []
        for component, data in self._components:
            if not component.initialized:
                raise RuntimeError("Component not initialized before render.")

            # The component renders its artist
            artist_dict = component.render(data, self.config.style)
            rendered_artists.append(artist_dict)

            # set axis limits for quad/polygon plots since relim() doesn't work on mesh collections
            if isinstance(component, (QuadPlotComponent, PolygonPlotComponent)):
                if isinstance(data, FieldData) and data.domain:
                    x_data = (
                        data.domain[1]
                        if len(data.domain) > 1
                        else data.domain[0]
                    )
                    y_data = data.domain[0] if len(data.domain) > 1 else None
                    main_ax.set_xlim(x_data.min(), x_data.max())
                    if y_data is not None:
                        main_ax.set_ylim(y_data.min(), y_data.max())

        main_ax.relim()
        main_ax.autoscale_view()

        # Get context from the *first* component
        first_component, first_data = (
            self._components[0] if self._components else (None, None)
        )
        if first_data is None:
            return  # Nothing to format

        has_special_formatting = isinstance(
            first_component,
            (CoordinateProfileComponent,),
        )
        # Set Title
        time = first_data.time if hasattr(first_data, "time") else None
        formatting.set_title(main_ax, self.fig, self.config.style, time)

        # Set Axis Labels
        xlabel, ylabel = None, None
        multid_plot = isinstance(
            first_component, (QuadPlotComponent, PolygonPlotComponent)
        )

        if isinstance(first_data, FieldData):
            if first_data.axis_names and multid_plot:
                xlabel = map_coordinate_label(
                    first_data.axis_names[0], self.coord_system
                )
                if len(first_data.axis_names) > 1:
                    ylabel = map_coordinate_label(
                        first_data.axis_names[1], self.coord_system
                    )
            if isinstance(
                first_component, (LinePlotComponent, TimeSeriesPlotComponent)
            ):
                ylabel = get_field_str(first_data.name)
                if first_data.axis_names:
                    xlabel = first_data.axis_names[0]

        if not has_special_formatting:
            formatting.apply_axis_labels(
                main_ax, self.config.style, xlabel, ylabel
            )
        formatting.apply_axis_limits(main_ax, self.config.style)

        # Add a colorbar only if a 2D component was rendered
        mappable_artist = None
        if any(
            isinstance(c, (QuadPlotComponent, PolygonPlotComponent))
            for c, d in self._components
        ):
            # Find the primary mappable artist
            mappable_artist = rendered_artists[0].get(
                "mesh"
            ) or rendered_artists[0].get("collection")

        if mappable_artist:
            self._format_colorbar(main_ax, mappable_artist, first_data)

        # Add a legend only if line components were rendered
        if any(
            isinstance(
                c,
                (
                    LinePlotComponent,
                    CoordinateProfileComponent,
                    TimeSeriesPlotComponent,
                ),
            )
            for c, d in self._components
        ):
            formatting.remove_spines(main_ax)
            formatting.apply_legend(main_ax, self.config.style)

    def _format_colorbar(self, ax: Axes, artist: Any, field_data: FieldData):
        """
        "Smart" internal method for colorbar formatting.
        Contains the layout logic that was (incorrectly) in the formatter.
        """
        label = field_data.name
        if self.fig is None:
            return

        if "_polygons" in label:
            label = label.split("_polygons")[0]

        if ax.name == "polar":
            # --- Polar Colorbar Logic ---
            theta = field_data.domain[1]
            max_angle = theta[-1]
            half_sphere = max_angle == 0.5 * math.pi
            orientation = "horizontal" if half_sphere else "vertical"

            polar_pos = ax.get_position()
            nfields = 1  # TODO: This should be derived from component count

            if orientation == "horizontal":
                width = min(0.6, 0.78 / nfields)
                x = polar_pos.x0 + (polar_pos.width - width) / 2 - 0.01
                cax = self.fig.add_axes((x, 0.2, width, 0.03))
            else:
                height = 0.8 / (2 if max_angle < math.pi else 1)
                x = polar_pos.x0 + polar_pos.width + 0.05
                y = polar_pos.y0 + (polar_pos.height - height) / 2
                cax = self.fig.add_axes((x, y, 0.03, height))

        else:
            # --- Cartesian Colorbar Logic ---
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            orientation = "vertical"

        formatting.add_colorbar(self.fig, artist, cax, label, orientation)

    def save(self, path: str):
        if self.fig:
            self.fig.savefig(path, dpi=self.config.style.dpi)

    def show(self):
        if self.fig:
            plt.show()

    def animate(self, files: Sequence[str], interval: int):
        print("Animation logic would go here.")

    def tight_layout(self):
        if self.fig:
            self.fig.tight_layout()
