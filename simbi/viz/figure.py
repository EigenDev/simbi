# =============================================================================
# figure.py
#
# orchestrator for visualization components.
#
# responsibilities:
#   - prepare and own the matplotlib Figure/Axes
#   - manage component lifecycle (initialize, render, cleanup)
#   - collect component outputs and delegate formatting to FigureFormatter
#
# contract with components:
#   a component implements the Component protocol (see components/interface.py)
#   and returns a `RenderResult`: the artists it drew, plus the facts the
#   component alone holds -- which artist a colorbar describes, what to call
#   it, the extent it drew, whether it is a vector overlay riding on another
#   artist.
#
#   the figure composes the view from those extents and delegates every
#   formatting decision (title, axis labels, colorbar, legend, spines) to
#   `FigureFormatter.apply_figure_formatting`.
# =============================================================================

from typing import Any, Optional, Sequence, Tuple

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
from .types import CoordSystem, FieldData


_warn_once = formatting.warn_once

# a checkpoint can be short or truncated and cost one frame, but a run of failed
# frames is a movie frozen from that point on, written out with a zero exit code
MAX_CONSECUTIVE_FRAME_FAILURES = 10



class Figure:
    """Manages the matplotlib figure, axes, and components.

    Notes:
      - Components should be initialized (initialize(fig, ax)) by the caller
        using prepare_figure (not implemented here) before calling render().
      - The figure delegates all layout/formatting responsibilities to the
        injected `FigureFormatter`. This keeps orchestration separate from
        presentation logic.
    """

    def __init__(
        self,
        config: VisualizationConfig,
        formatter: Optional[formatting.FigureFormatter] = None,
    ):
        """
        Args:
            config: VisualizationConfig for the figure
            formatter: Optional FigureFormatter. If omitted, a default is constructed
                       from `config.figure`.
        """
        self.config = config
        self.fig: Optional[MplFigure] = None
        self.axes: dict[str, Axes] = {}
        self._components: list[tuple[Component, Any]] = []
        self._anim: Optional[FuncAnimation] = None
        self.coord_system: CoordSystem = CoordSystem.CARTESIAN
        # matplotlib settings this figure draws under; see styled()
        self.theme_rc: Optional[dict] = None

        # formatter encapsulates all figure-level formatting (title, labels, colorbar, legend)
        # allow injection for testability and alternate layout policies
        if formatter is not None:
            self.formatter = formatter
        else:
            self.formatter = formatting.FigureFormatter(self.config.figure)

    def styled(self):
        """Draw under this figure's theme.

        matplotlib reads its settings when an artist is created, so every path
        that creates one runs inside this. the settings are scoped to the block
        rather than pushed into the global rcParams, where they would outlive
        this figure and restyle every later one in the session.
        """
        if self.theme_rc is None:
            from contextlib import nullcontext

            return nullcontext()

        import matplotlib.style

        return matplotlib.style.context(["default", self.theme_rc])

    def add_component(self, component: Component, data: Any = None):
        """Adds a component and its associated data payload."""
        self._components.append((component, data))

    def render(self):
        """Render every component against the payload it was given, and format.

        this is the still-image path: it draws the payloads the caller attached
        rather than a frame's worth of freshly loaded data, and otherwise takes
        the same route a movie frame does -- draw, compose the view, format.
        """
        if not self.fig or not self.axes:
            raise RuntimeError("Figure has not been prepared.")

        main_ax = self.axes["main"]

        with self.styled():
            formatting.apply_scaling(main_ax, self.config.figure)

            rendered = self._draw_components(
                [payload for _, payload in self._components]
            )

            self._autoscale_lines(main_ax)
            self._apply_view(main_ax, rendered)

            _, first_data = (
                self._components[0] if self._components else (None, None)
            )
            if first_data is not None:
                self._format_fully(main_ax, rendered, first_data)

    def _autoscale_lines(self, main_ax: Axes) -> None:
        """Fit the view to artists that carry a data limit.

        a mesh collection carries none, so relim cannot see it and would zero
        the view out from under it; those components report their drawn extent
        instead (see _apply_view).
        """
        if any(
            isinstance(component, (QuadPlotComponent, PolygonPlotComponent))
            for component, _ in self._components
        ):
            return

        # a polar or otherwise specialized axes may not support autoscaling,
        # and there is nothing to fall back to but the limits already set
        try:
            main_ax.relim()
            main_ax.autoscale_view()
        except Exception as exc:
            _warn_once(
                f"autoscale:{type(exc).__name__}", f"autoscale failed: {exc}"
            )

    def _apply_view(self, main_ax: Axes, rendered_artists: list) -> None:
        """Set the view to hold everything drawn on the shared axes.

        the axes carries every component, so no one of them owns the limits: a
        field laid into one angular sector would otherwise crop the sector
        beside it. the extent is recomputed from the artists of each frame,
        which is what keeps a moving mesh in view as it expands. an axis the
        user fixed is left alone.
        """
        boxes = [
            result.view_bounds
            for result in rendered_artists
            if result.view_bounds is not None
        ]
        if not boxes:
            return

        style = self.config.figure
        if not style.xlims_pinned:
            main_ax.set_xlim(
                min(box[0] for box in boxes), max(box[1] for box in boxes)
            )
        if not style.ylims_pinned:
            main_ax.set_ylim(
                min(box[2] for box in boxes), max(box[3] for box in boxes)
            )

    def save(self, path: str):
        """
        Save figure with smart extension based on plot type:
          - line/time_series/coordinate_profile -> .pdf (vector)
          - quad/polygon (2d) -> .png (hi-res raster)
          - animation -> .mp4
        """
        from simbi.reader import logger
        from simbi.reader.progress import create_progress_bar

        if not self.fig:
            raise RuntimeError("figure has not been prepared")

        # strip extension if provided, normalize path
        import os

        base, ext = os.path.splitext(path)
        base = base.strip().replace(" ", "_")

        if self._anim is not None:
            # animation: default to mp4
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
            logger.info(f"animation saved: {save_path}")
        else:
            # determine extension from component types
            is_line_like = any(
                isinstance(
                    comp,
                    (
                        LinePlotComponent,
                        CoordinateProfileComponent,
                        TimeSeriesPlotComponent,
                    ),
                )
                for comp, _ in self._components
            )
            is_2d = any(
                isinstance(comp, (QuadPlotComponent, PolygonPlotComponent))
                for comp, _ in self._components
            )

            if ext:
                # user specified extension, respect it
                save_path = base + ext
            elif is_line_like and not is_2d:
                # line plots -> pdf (vector graphics)
                save_path = base + ".pdf"
            else:
                # 2d plots -> png (hi-res raster)
                save_path = base + ".png"

            self.fig.savefig(
                save_path,
                dpi=self.config.figure.dpi,
                bbox_inches="tight",
                transparent=self.config.figure.transparent,
            )
            logger.info(f"figure saved: {save_path}")

    def show(self):
        if self.fig:
            plt.show()

    def animate(
        self,
        files: Sequence[str],
        fps: int = 30,
        frame_data: Optional[Any] = None,
    ):
        """
        create an animation from a sequence of checkpoint files.

        strategy:
          - reuse existing component instances and their in-place update/render paths
          - drive per-frame updates with matplotlib.animation.FuncAnimation
          - save using matplotlib's ffmpeg writer (requires ffmpeg on PATH)

        args:
          files: ordered list of checkpoint file paths (frames)
          fps: frames per second for the movie
          frame_data: maps a checkpoint path to the PlotData drawn for it. this
                      is the whole of what separates one kind of animation from
                      another; it defaults to the field pipeline.
        """
        from matplotlib.animation import FuncAnimation

        if not files:
            raise ValueError("no input files provided for animation")

        if not self.fig:
            raise RuntimeError("figure must be prepared before animating")

        self._frame_failures = 0
        build = frame_data or self._field_frame_data

        # components are held against the payload they were built with, and each
        # frame's fields are new objects, so they are matched back by name
        self._signatures = [payload for _, payload in self._components]

        # the immersed-body silhouette artists from the preceding frame, removed
        # before the next frame's overlay so a tumbling body shows one
        # silhouette per frame of the movie
        self._body_artists = []
        self._horizon_artists = []

        def _init():
            assert self.fig is not None
            self.fig.canvas.draw()
            return []

        def _update(index: int):
            self._animate_frame(files[index], build, full=index == 0)
            return []

        # the first frame is drawn here rather than left to the writer, so a
        # setup failure is raised before an output file is opened
        _update(0)

        self._anim = FuncAnimation(
            self.fig,
            _update,
            frames=len(files),
            init_func=_init,
            blit=False,
            interval=int(1000 / fps),
        )

    def animate_coordinate_profile(
        self,
        files: Sequence[str],
        fields: Sequence[str],
        config: "VisualizationConfig",
        fps: int = 30,
    ):
        """
        create an animation of coordinate-binned profiles from multiple files.

        the profile pipeline replaces the field one: it bins along a coordinate
        and can average over a sphere, which a per-cell field render cannot.
        everything after that -- dispatch, view, formatting -- is the animation
        path every other movie takes.
        """
        from simbi.viz.pipeline.coord_binning import (
            create_coordinate_profile_data,
        )

        def profile_frame(file_path: str):
            from simbi.viz.pipeline import load_data

            return create_coordinate_profile_data(
                load_data(file_path), fields, config
            )

        self.animate(files, fps=fps, frame_data=profile_frame)

    def _field_frame_data(self, file_path: str):
        """the fields of one checkpoint, composed as they are drawn.

        composition runs per frame because it is what lays several quantities
        into their sectors, and a moving mesh puts them at new vertices each
        time."""
        from simbi.viz.pipeline import load_data
        from simbi.viz.pipeline.plot_data import create_plot_data
        from simbi.viz.pipeline.transforms import compose_fields_for_render
        from simbi.viz.types import PlotData

        sim_data = load_data(file_path)
        plot_data = create_plot_data(
            sim_data, getattr(self.config.plot, "fields", []), self.config
        )
        composed = compose_fields_for_render(plot_data.fields, self.config)

        return PlotData(
            fields=composed,
            time=plot_data.time,
            dimensions=composed[0].ndim if composed else 0,
            coord_system=plot_data.coord_system,
            hierarchy=plot_data.hierarchy,
        )

    def _animate_frame(self, file_path: str, build, *, full: bool) -> None:
        """draw one frame of an animation.

        a single failed frame is tolerated -- a checkpoint can be short or
        truncated -- but a run of them is a movie frozen from that point on,
        shipped with a zero exit code, so it aborts instead.
        """
        try:
            self.draw_frame(build(file_path), full=full, file_path=file_path)
            self._frame_failures = 0
        except Exception as exc:
            self._frame_failures += 1
            _warn_once("frame-update", f"animation frame failed: {exc}")
            if self._frame_failures >= MAX_CONSECUTIVE_FRAME_FAILURES:
                # an interactive backend drives frames from a gui timer, which
                # swallows whatever the callback raises and fires again: the
                # timer has to be stopped or the same failure reprints forever
                self.stop_animation()
                raise RuntimeError(
                    f"animation aborted: {self._frame_failures} consecutive frame"
                    f" failures (last: {exc})"
                ) from exc

    def stop_animation(self) -> None:
        """Halt the frame timer, if one is running."""
        source = getattr(self._anim, "event_source", None)
        if source is not None:
            source.stop()

    def draw_frame(
        self,
        plot_data: Any,
        *,
        full: bool = True,
        file_path: Optional[str] = None,
    ) -> None:
        """Draw one frame of data onto the prepared axes.

        the order is load-bearing: the components draw, the view is composed
        from what they drew, and only then does the formatting that reads the
        view run -- a colorbar is placed against the extent of the wedge, so a
        view fixed afterwards would place it against the previous frame's.

        `full` formats everything (labels, colorbar, legend); otherwise only
        the parts that change between frames are touched, which is what keeps a
        movie's layout from shifting under itself.
        """
        if not self.fig or not self.axes:
            raise RuntimeError("Figure has not been prepared.")

        main_ax = self.axes["main"]
        signatures = getattr(self, "_signatures", None) or [
            payload for _, payload in self._components
        ]

        with self.styled():
            rendered = self._draw_components(
                [self._payload_for(sig, plot_data) for sig in signatures]
            )

            self._apply_view(main_ax, rendered)

            first_data = plot_data.fields[0] if plot_data.fields else None
            if first_data is not None:
                if full:
                    self._format_fully(main_ax, rendered, first_data)
                else:
                    self._format_between_frames(main_ax, rendered, first_data)

            if file_path is not None:
                self._overlay_on_frame(main_ax, file_path)

        try:
            self.fig.canvas.draw_idle()
        except Exception:
            self.fig.canvas.draw()

    def _draw_components(self, payloads: Sequence[Any]) -> list:
        """Hand each component its payload and collect the normalized returns."""
        rendered = []
        for (component, _), payload in zip(self._components, payloads):
            if not component.initialized:
                raise RuntimeError("Component not initialized before render.")
            if payload is None:
                continue

            result = component.render(payload, self.config.figure)
            if result is not None:
                rendered.append(result)

        return rendered

    @staticmethod
    def _find_field(plot_data: Any, name: str):
        """the field of this frame that answers to `name`, preferring an exact
        match over the level-suffixed ones."""
        for field in plot_data.fields:
            if field.name == name:
                return field
        for field in plot_data.fields:
            if field.name.startswith(name):
                return field
        return None

    def _payload_for(self, signature: Any, plot_data: Any):
        """this frame's payload for a component set up against `signature`.

        the fields of each frame are new objects carrying the same names, so a
        component is matched to its quantity rather than to a position. that
        holds whichever way the quantity is drawn -- a field on a mesh or a
        composed set of polygons -- since both carry the name."""
        from .types import FieldData, PolygonData

        if isinstance(signature, (FieldData, PolygonData)):
            found = self._find_field(plot_data, signature.name)
            if found is not None:
                return found
            return plot_data.fields[0] if plot_data.fields else None

        if isinstance(signature, (list, tuple)):
            matched = [
                found
                for elt in signature
                if hasattr(elt, "name")
                and (found := self._find_field(plot_data, elt.name)) is not None
            ]
            return matched or list(plot_data.fields)

        # a component given the whole plot data keeps taking the whole of it
        return plot_data

    def _format_fully(self, main_ax: Axes, rendered: list, first_data) -> None:
        assert self.fig is not None
        try:
            self.formatter.apply_figure_formatting(
                self.fig,
                main_ax,
                rendered,
                first_data,
                coord_system=self.coord_system,
                xlabel=None,
                ylabel=None,
                show_legend=True,
            )
        except Exception as exc:
            # a silent formatting failure ships an unlabeled plot as a success
            _warn_once(
                f"figure-format:{type(exc).__name__}",
                f"figure formatting failed: {exc}",
            )
            raise

    def _format_between_frames(
        self, main_ax: Axes, rendered: list, first_data
    ) -> None:
        """the parts of the formatting that change from frame to frame.

        the rest is left alone, which holds the layout fixed under the movie."""
        assert self.fig is not None
        from simbi.viz.formatting import set_title

        set_title(
            main_ax,
            self.fig,
            self.config.figure,
            getattr(first_data, "time", None),
        )

        # a mesh whose vertices moved is rebuilt rather than refilled, so the
        # colorbar is left addressing a discarded artist
        self.formatter.refresh_colorbars(
            self.fig, main_ax, rendered, first_data
        )

    def _overlay_on_frame(self, main_ax: Axes, file_path: str) -> None:
        """redraw the per-frame overlays at this checkpoint's state.

        the previous frame's artists are removed first, so a tumbling body does
        not smear across the movie."""
        style = self.config.figure

        if getattr(style, "draw_bodies", False):
            from simbi.viz.bodies import overlay_bodies_on_slice
            from simbi.viz.pipeline import load_data

            for artist in self._body_artists:
                artist.remove()
            self._body_artists = overlay_bodies_on_slice(
                main_ax,
                file_path,
                self.config.plot.slice,
                load_data(file_path).metadata.coord_system,
            )

        # the horizon is fixed (a black hole at the origin, constant mass), but
        # it is redrawn per frame so it survives an axis clear between frames
        if getattr(style, "draw_horizon", False):
            from simbi.viz.horizon import overlay_horizon_on_slice
            from simbi.viz.pipeline import load_data

            for artist in self._horizon_artists:
                artist.remove()
            metadata = load_data(file_path).metadata
            self._horizon_artists = overlay_horizon_on_slice(
                main_ax,
                metadata,
                self.config.plot.slice,
                metadata.coord_system,
            )

    def tight_layout(self):
        if self.fig:
            self.fig.tight_layout()
