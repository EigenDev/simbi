# =============================================================================
# figure.py
#
# orchestrator for visualization components.
#
# responsibilities:
#   - prepare and own the matplotlib Figure/Axes
#   - manage component lifecycle (initialize, render, cleanup)
#   - collect component outputs and delegate formatting to FigureFormatter
#   - drive animations via a pluggable data pipeline
#
# the prepare_figure() factory lives here (not in pipeline/) to avoid
# circular imports between the data pipeline and presentation layers.
# =============================================================================
import logging
from typing import Any, Callable, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure

from simbi.viz.components.coord_binning import CoordinateProfileComponent
from simbi.viz.components.time_series import TimeSeriesPlotComponent

from . import formatting
from .components import (
    LinePlotComponent,
    PolygonPlotComponent,
    QuadPlotComponent,
)
from .components.interface import Component
from .config import VisualizationConfig
from .types import CoordSystem, FieldData, PlotData

logger = logging.getLogger(__name__)


def _normalize_render_output(
    result: Any,
) -> Tuple[dict, Optional[dict]]:
    """normalize component.render() output to (artists_dict, metadata)."""
    if result is None:
        return {}, None

    if hasattr(result, "artists"):
        artists = getattr(result, "artists", {}) or {}
        metadata = getattr(result, "metadata", None)
        if isinstance(artists, dict):
            return artists, metadata

    if isinstance(result, (list, tuple)) and len(result) >= 1:
        artists_candidate = result[0] if len(result) > 0 else {}
        metadata_candidate = result[1] if len(result) > 1 else None
        if isinstance(artists_candidate, dict):
            return artists_candidate, metadata_candidate
        return {}, metadata_candidate

    if isinstance(result, dict):
        return result, None

    return {}, None


def prepare_figure(
    config: VisualizationConfig,
    nfiles: int = 1,
    projection: Literal["polar", "cartesian"] | None = None,
    nlvls: int = 1,
    coord_system: CoordSystem = CoordSystem.CARTESIAN,
    formatter: Optional[object] = None,
    overlay_mode: bool = False,
) -> "Figure":
    """create and prepare a figure based on configuration."""
    config.theme.apply(
        nfiles=nfiles,
        nfields=len(config.plot.fields) * nlvls,
        overlay_mode=overlay_mode,
    )
    if projection == "polar":
        fig, ax = plt.subplots(
            1,
            1,
            figsize=config.figure.fig_size,
            subplot_kw={"projection": "polar"},
            layout="constrained",
        )
    else:
        fig = plt.figure(figsize=config.figure.fig_size)
        ax = fig.add_subplot(111)

    figure = Figure(config, formatter=formatter)
    figure.fig = fig
    figure.axes["main"] = ax
    figure.coord_system = coord_system
    return figure


def _find_field(plot_data: PlotData, name: str) -> Optional[FieldData]:
    """find a field by name in plot data (exact match, then prefix)."""
    for f in plot_data.fields:
        if f.name == name:
            return f
    for f in plot_data.fields:
        if f.name.startswith(name):
            return f
    return None


def _dispatch_to_component(
    component: Component,
    signature: Any,
    plot_data: PlotData,
    style: Any,
) -> Optional[Any]:
    """dispatch updated data to a component based on its original payload signature."""
    if isinstance(signature, FieldData):
        new_field = _find_field(plot_data, signature.name)
        if new_field is not None:
            return component.render(new_field, style)
        if plot_data.fields:
            return component.render(plot_data.fields[0], style)
        return None

    if isinstance(signature, (list, tuple)):
        new_payload = []
        for elt in signature:
            if hasattr(elt, "name"):
                found = _find_field(plot_data, elt.name)
                if found is not None:
                    new_payload.append(found)
        if new_payload:
            return component.render(new_payload, style)
        if plot_data.fields:
            return component.render(plot_data.fields, style)
        return None

    return component.render(plot_data, style)


class Figure:
    """manages the matplotlib figure, axes, and components."""

    def __init__(
        self,
        config: VisualizationConfig,
        formatter: Optional[formatting.FigureFormatter] = None,
    ):
        self.config = config
        self.fig: Optional[MplFigure] = None
        self.axes: dict[str, Axes] = {}
        self._components: list[
            tuple[Component, Any, bool]
        ] = []  # (component, data, is_overlay)
        self._anim: Optional[FuncAnimation] = None
        self.coord_system: CoordSystem = CoordSystem.CARTESIAN

        if formatter is not None:
            self.formatter = formatter
        else:
            self.formatter = formatting.FigureFormatter(self.config.figure)

    def add_component(
        self, component: Component, data: Any = None, is_overlay: bool = False
    ):
        """
        adds a component and its associated data payload.

        args:
            component: the component to add
            data: the data payload for the component
            is_overlay: if True, marks this as an overlay (no colorbar, etc.)
        """
        self._components.append((component, data, is_overlay))

    def render(self):
        """renders all components and applies formatting."""
        if not self.fig or not self.axes:
            raise RuntimeError("figure has not been prepared.")

        main_ax = self.axes["main"]
        formatting.apply_scaling(main_ax, self.config.figure)

        rendered_artists = []
        has_mesh_collection = False

        for component, data, is_overlay in self._components:
            if not component.initialized:
                raise RuntimeError("component not initialized before render.")

            result = component.render(data, self.config.figure)
            artist_dict, metadata = _normalize_render_output(result)

            # mark overlay metadata so formatter can skip colorbars
            if is_overlay:
                if metadata is None:
                    metadata = {}
                metadata["is_overlay"] = True

            rendered_artists.append((artist_dict, metadata))

            if isinstance(component, (QuadPlotComponent, PolygonPlotComponent)):
                has_mesh_collection = True
                if isinstance(data, FieldData) and len(data.domain) > 0:
                    self._set_mesh_limits(main_ax, component, data)

        if not has_mesh_collection:
            main_ax.relim()
            main_ax.autoscale_view()

        # apply user-specified limits (overrides auto limits)
        style = self.config.figure
        if style.xlims is not None:
            if style.xlims.min is not None or style.xlims.max is not None:
                main_ax.set_xlim(style.xlims.min, style.xlims.max)
        if style.ylims is not None:
            if style.ylims.min is not None or style.ylims.max is not None:
                main_ax.set_ylim(style.ylims.min, style.ylims.max)

        # get first non-overlay component data for formatting
        first_data = None
        for _, data, is_overlay in self._components:
            if not is_overlay:
                first_data = data
                break

        if first_data is None and self._components:
            first_data = self._components[0][1]

        if first_data is None:
            return

        self.formatter.apply_figure_formatting(
            self.fig,
            main_ax,
            rendered_artists,
            first_data,
            coord_system=self.coord_system,
        )

    def _set_mesh_limits(
        self, ax: Axes, component: Component, data: FieldData
    ) -> None:
        """set axis limits from mesh/polygon domain data."""
        import numpy as np

        if isinstance(component, PolygonPlotComponent):
            patches = np.asarray(data.domain)
            all_x = patches[:, :, 0].flatten()
            all_y = patches[:, :, 1].flatten()
            ax.set_xlim(float(all_x.min()), float(all_x.max()))
            ax.set_ylim(float(all_y.min()), float(all_y.max()))
        else:
            x_data = data.domain[1] if len(data.domain) > 1 else data.domain[0]
            y_data = data.domain[0] if len(data.domain) > 1 else None
            if x_data is None or y_data is None:
                return
            if ax.name == "polar":
                x_data, y_data = y_data, x_data
            ax.set_xlim(x_data.min(), x_data.max())
            ax.set_ylim(y_data.min(), y_data.max())

    def save(self, path: str):
        """save figure with smart extension based on plot type."""
        from simbi.reader import logger as reader_logger
        from simbi.reader.progress import create_progress_bar

        if not self.fig:
            raise RuntimeError("figure has not been prepared")

        import os

        base, ext = os.path.splitext(path)
        base = base.strip().replace(" ", "_")

        if self._anim is not None:
            if ext not in (".mp4", ".avi", ".mov", ".gif"):
                ext = ".mp4"
            save_path = base + ext

            prog_bar = create_progress_bar()
            with prog_bar:
                task = prog_bar.add_task(
                    f"[green]saving animation to {save_path}...",
                    total=self.config.animation.total_frames,
                )

                def prog_callback(current: int, total: int) -> None:
                    prog_bar.update(task, advance=1)

                self._anim.save(
                    save_path,
                    dpi=self.config.figure.dpi,
                    progress_callback=prog_callback,
                )
            reader_logger.info(f"animation saved: {save_path}")
        else:
            is_line_like = any(
                isinstance(
                    comp,
                    (
                        LinePlotComponent,
                        CoordinateProfileComponent,
                        TimeSeriesPlotComponent,
                    ),
                )
                for comp, _, _ in self._components
            )
            is_2d = any(
                isinstance(comp, (QuadPlotComponent, PolygonPlotComponent))
                for comp, _, _ in self._components
            )

            if ext:
                save_path = base + ext
            elif is_line_like and not is_2d:
                save_path = base + ".pdf"
            else:
                save_path = base + ".png"

            self.fig.savefig(
                save_path,
                dpi=self.config.figure.dpi,
                bbox_inches="tight",
                transparent=self.config.figure.transparent,
            )
            reader_logger.info(f"figure saved: {save_path}")

    def show(self):
        if self.fig:
            plt.show()

    def _animate_with_pipeline(
        self,
        files: Sequence[str],
        data_pipeline: Callable[[str], PlotData],
        config: VisualizationConfig,
        fps: int = 30,
    ) -> None:
        """unified animation driver parameterized by a data pipeline.

        both animate() and animate_coordinate_profile() delegate here.
        the only difference between them is the data_pipeline callable.
        """
        if not files:
            raise ValueError("no input files provided for animation")
        if not self.fig:
            raise RuntimeError("figure must be prepared before animating")

        nframes = len(files)
        signatures = [
            (payload, is_overlay) for _, payload, is_overlay in self._components
        ]

        def _init():
            self.fig.canvas.draw()
            return []

        def _render_frame(plot_data: PlotData) -> list:
            """render all components for a single frame."""
            rendered = []
            for (component, _, is_overlay), (sig, _) in zip(
                self._components, signatures
            ):
                result = _dispatch_to_component(
                    component, sig, plot_data, config.figure
                )
                if result is not None:
                    artist_dict, metadata = _normalize_render_output(result)
                    if is_overlay:
                        if metadata is None:
                            metadata = {}
                        metadata["is_overlay"] = True
                    rendered.append((artist_dict, metadata))
            return rendered

        def _update(ii: int):
            plot_data = data_pipeline(files[ii])
            _render_frame(plot_data)

            main_ax = self.axes.get("main")
            if main_ax is not None and plot_data.fields:
                time = getattr(plot_data.fields[0], "time", None)
                formatting.set_title(main_ax, self.fig, config.figure, time)

            self.fig.canvas.draw_idle()
            return []

        # render frame 0 with full formatting
        frame0_data = data_pipeline(files[0])
        rendered_frame0 = _render_frame(frame0_data)

        main_ax = self.axes.get("main")
        if main_ax is not None and frame0_data.fields:
            self.formatter.apply_figure_formatting(
                self.fig,
                main_ax,
                rendered_frame0,
                frame0_data.fields[0],
                coord_system=self.coord_system,
            )

        self.fig.canvas.draw_idle()

        self._anim = FuncAnimation(
            self.fig,
            _update,
            frames=nframes,
            init_func=_init,
            blit=False,
            interval=int(1000 / fps),
        )

    def animate(
        self,
        files: Sequence[str],
        output_path: str | None = None,
        fps: int = 30,
        save_all_frames: bool = False,
    ):
        """create an animation from a sequence of checkpoint files."""
        from simbi.viz.pipeline import load_data
        from simbi.viz.pipeline.plot_data import create_plot_data
        from simbi.viz.pipeline.transforms import compose_fields_for_render

        field_names = self.config.plot.fields

        def pipeline(file_path: str) -> PlotData:
            sim_data = load_data(file_path)
            raw = create_plot_data(sim_data, field_names, self.config)
            composed = compose_fields_for_render(raw.fields, self.config)
            return PlotData(
                fields=composed,
                body_collection=raw.body_collection,
                time=raw.time,
                dimensions=composed[0].ndim if composed else 0,
                coord_system=raw.coord_system,
                hierarchy=raw.hierarchy,
            )

        self._animate_with_pipeline(files, pipeline, self.config, fps)

    def animate_coordinate_profile(
        self,
        files: Sequence[str],
        fields: Sequence[str],
        config: VisualizationConfig,
        output_path: str | None = None,
        fps: int = 30,
        save_all_frames: bool = False,
    ):
        """create an animation of coordinate-binned profiles."""
        from simbi.viz.pipeline import load_data
        from simbi.viz.pipeline.coord_binning import (
            create_coordinate_profile_data,
        )

        def pipeline(file_path: str) -> PlotData:
            sim_data = load_data(file_path)
            return create_coordinate_profile_data(sim_data, fields, config)

        self._animate_with_pipeline(files, pipeline, config, fps)

    def tight_layout(self):
        if self.fig:
            self.fig.tight_layout()
