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
#   components must implement the Component protocol (see components/interface.py)
#   and should return a `RenderResult` from `render()` whenever possible.
#
#   RenderResult:
#     - artists: dict[str, object] mapping semantic keys -> matplotlib artists
#     - metadata: optional dict[str, object] containing hints used by the
#       FigureFormatter (common keys: 'mappable', 'label', 'is_line', 'is_vector',
#       'preferred_cmap', 'color_range')
#
#   the Figure accepts legacy returns (plain dict/list) but normalizes them
#   into (artists_dict, metadata) tuples using `_normalize_render_output`.
#   formatting decisions (title, axis labels, colorbar, legend, spines, limits)
#   are delegated to `FigureFormatter.apply_figure_formatting`.
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


# formatting/frame failures are otherwise swallowed silently (unlabeled
# plots, frozen movies exiting 0); each distinct failure warns once with
# the real exception so the defect is visible without spamming per frame.
_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        import warnings

        warnings.warn(msg, stacklevel=3)



def _normalize_render_output(
    result: Any,
) -> Tuple[dict, Optional[dict]]:
    """
    Normalize the possible outputs of a component.render(...) call into a
    canonical (artists_dict, metadata_dict_or_none) tuple.

    Accepted input forms:
      - None -> ({}, None)
      - RenderResult-like object (has .artists attribute) -> (artists, metadata)
      - dict -> (dict, None)
      - mapping-like object coercible to dict -> (dict(result), None)
      - tuple/list of (artists, metadata) -> same normalized
      - any other -> ({}, None)

    Rationale:
      - centralizes coercion logic so Figure and tests rely on one implementation.
      - ensures the FigureFormatter receives consistent inputs.
    """
    if result is None:
        return {}, None

    # explicit RenderResult or RenderResult-like objects
    if hasattr(result, "artists"):
        try:
            artists = getattr(result, "artists", {}) or {}
            metadata = getattr(result, "metadata", None)
            if isinstance(artists, dict):
                return artists, metadata
        except Exception:
            # fallthrough to other coercions
            pass

    # tuple/list of (artists, metadata)
    if isinstance(result, (list, tuple)) and len(result) >= 1:
        artists_candidate = result[0] if len(result) > 0 else {}
        metadata_candidate = result[1] if len(result) > 1 else None
        if isinstance(artists_candidate, dict):
            return artists_candidate, metadata_candidate
        # if first element is mapping-like, attempt to coerce
        try:
            return dict(artists_candidate), metadata_candidate
        except Exception:
            return {}, metadata_candidate

    # plain dict -> legacy artists-only return
    if isinstance(result, dict):
        return result, None

    # lastly, best-effort coercion to dict
    try:
        coerced = dict(result)
        return coerced, None
    except Exception:
        return {}, None


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

        # formatter encapsulates all figure-level formatting (title, labels, colorbar, legend)
        # allow injection for testability and alternate layout policies
        if formatter is not None:
            self.formatter = formatter
        else:
            self.formatter = formatting.FigureFormatter(self.config.figure)

    def add_component(self, component: Component, data: Any = None):
        """Adds a component and its associated data payload."""
        self._components.append((component, data))

    def render(self):
        """Renders all components and applies formatting.

        Workflow:
          - apply pre-render scaling (log/semilog) from style
          - call component.render(payload, style) for each component
          - normalize outputs with _normalize_render_output
          - set axis limits for quad/polygon components (mesh-collections)
          - delegate the composited formatting to FigureFormatter.apply_figure_formatting
        """
        if not self.fig or not self.axes:
            raise RuntimeError("Figure has not been prepared.")

        main_ax = self.axes["main"]

        # --- pre-render formatting ---
        formatting.apply_scaling(main_ax, self.config.figure)

        # --- render data ---
        rendered_artists = []
        has_mesh_collection = False
        for component, data in self._components:
            if not component.initialized:
                raise RuntimeError("Component not initialized before render.")

            # the component renders its artist (components should return RenderResult,
            # but legacy dict/list returns are tolerated)
            result = component.render(data, self.config.figure)
            artist_dict, metadata = _normalize_render_output(result)

            # store normalized tuple (artists_dict, metadata) for the formatter
            rendered_artists.append((artist_dict, metadata))

            # set axis limits for quad/polygon plots since relim() doesn't work on mesh collections
            if isinstance(component, (QuadPlotComponent, PolygonPlotComponent)):
                has_mesh_collection = True
                if isinstance(data, FieldData) and len(data.domain) > 0:
                    try:
                        # for polygon data, domain contains patches (vertices)
                        # shape: (n_patches, 4, 2) where 4 is vertices per patch, 2 is (x, y)
                        if isinstance(component, PolygonPlotComponent):
                            import numpy as np

                            patches = np.asarray(data.domain)
                            # extract all x and y coordinates using numpy
                            all_x = patches[:, :, 0].flatten()
                            all_y = patches[:, :, 1].flatten()
                            main_ax.set_xlim(
                                float(all_x.min()), float(all_x.max())
                            )
                            main_ax.set_ylim(
                                float(all_y.min()), float(all_y.max())
                            )
                        else:
                            # quadmesh: domain contains coordinate arrays
                            x_data = (
                                data.domain[1]
                                if len(data.domain) > 1
                                else data.domain[0]
                            )

                            y_data = (
                                data.domain[0] if len(data.domain) > 1 else None
                            )
                            if x_data is None:
                                raise ValueError("empty x_data in domain")

                            if y_data is None:
                                raise ValueError("empty y_data in domain")

                            if main_ax.name == "polar":
                                x_data, y_data = y_data, x_data

                            main_ax.set_xlim(x_data.min(), x_data.max())
                            if y_data is not None:
                                main_ax.set_ylim(y_data.min(), y_data.max())
                    except Exception as e:
                        # log but don't break rendering
                        import logging

                        logging.getLogger(__name__).debug(
                            f"failed to set axis limits from domain: {e}"
                        )

        # only use relim/autoscale for non-mesh collections (lines, scatter, etc.)
        if not has_mesh_collection:
            try:
                main_ax.relim()
                main_ax.autoscale_view()
            except Exception:
                # some axes (polar, specialized) may not support relim/autoscale_view
                pass

        # apply user-specified limits from style config (overrides auto limits)
        style = self.config.figure
        if style.xlims is not None:
            if style.xlims.min is not None or style.xlims.max is not None:
                main_ax.set_xlim(style.xlims.min, style.xlims.max)
        if style.ylims is not None:
            if style.ylims.min is not None or style.ylims.max is not None:
                main_ax.set_ylim(style.ylims.min, style.ylims.max)

        # get context from the *first* component
        first_component, first_data = (
            self._components[0] if self._components else (None, None)
        )
        if first_data is None:
            return  # nothing to format

        # delegate all figure-level formatting to the FigureFormatter instance.
        try:
            assert self.fig is not None
            self.formatter.apply_figure_formatting(
                self.fig,
                main_ax,
                rendered_artists,
                first_data,
                coord_system=self.coord_system,
                xlabel=None,
                ylabel=None,
                show_legend=True,
            )
        except Exception as exc:
            # a silent formatting failure ships an unlabeled plot as a success —
            # surface it: warn with the real error, then fail the static render.
            _warn_once(f"figure-format:{type(exc).__name__}", f"figure formatting failed: {exc}")
            raise

    def _format_colorbar(self, ax: Axes, artist: Any, field_data: FieldData):
        # deprecated: colorbar placement is now the responsibility of FigureFormatter.
        # keep a lightweight no-op to avoid accidental direct calls.
        return

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
        output_path: str | None = None,
        fps: int = 30,
        save_all_frames: bool = False,
    ):
        """
        create an animation from a sequence of checkpoint files.

        strategy:
          - reuse existing component instances and their in-place update/render paths
          - drive per-frame updates with matplotlib.animation.FuncAnimation
          - save using matplotlib's ffmpeg writer (requires ffmpeg on PATH)
          - fallback: if writer is unavailable, write frames to a temp dir and call ffmpeg externally

        args:
          files: ordered list of checkpoint file paths (frames)
          output_path: output movie path (e.g., 'out.mp4'). if None, uses './animation.mp4'
          fps: frames per second for the movie
          save_all_frames: if True, keep intermediate frame PNGs
        """
        # lazy imports so module import remains cheap
        from typing import List

        from matplotlib.animation import FuncAnimation

        from simbi.viz.pipeline import load_data

        # pipeline helpers
        from simbi.viz.pipeline.plot_data import create_plot_data
        from simbi.viz.types import FieldData as PlotFieldData

        if not files:
            raise ValueError("no input files provided for animation")

        if not self.fig:
            raise RuntimeError("figure must be prepared before animating")

        output_path = output_path or "animation.mp4"
        nframes = len(files)

        # the immersed-body silhouette artists from the PREVIOUS frame, removed before
        # the next frame's overlay so a tumbling body does not smear across the movie.
        self._body_artists: list = []
        self._horizon_artists: list = []

        # build a map of the original component payload signatures used to
        # request the same plotted fields for each frame.
        component_signatures: List[object] = []
        for _, payload in self._components:
            component_signatures.append(payload)

        def _find_field(plot_data, name: str):
            # try exact match first, then prefix match
            for f in plot_data.fields:
                if f.name == name:
                    return f
            for f in plot_data.fields:
                if f.name.startswith(name):
                    return f
            return None

        # init function - draw baseline once
        def _init():
            # initial draw; components are assumed already initialized
            if self.fig is None:
                return []
            self.fig.canvas.draw()
            return []

        # update function called by FuncAnimation for frame i
        def _update(i: int):
            file_path = files[i]
            sim_data = load_data(file_path)

            # prepare full plot payloads for this frame: use all fields referenced
            # by the current visualization config
            field_names = getattr(self.config.plot, "fields", [])
            frame_plot_data = create_plot_data(
                sim_data, field_names, self.config
            )

            # apply composition to match component initialization
            from simbi.viz.pipeline.plot_data import PlotData
            from simbi.viz.pipeline.transforms import compose_fields_for_render

            composed_fields = compose_fields_for_render(
                frame_plot_data.fields, self.config
            )

            # create new PlotData with composed fields (PlotData is frozen)
            frame_plot_data = PlotData(
                fields=composed_fields,
                time=frame_plot_data.time,
                dimensions=composed_fields[0].ndim if composed_fields else 0,
                coord_system=frame_plot_data.coord_system,
                hierarchy=frame_plot_data.hierarchy,
            )

            # dispatch updated payloads to components and collect returned artist dicts
            rendered_artists_frame = []
            for (component, orig_payload), signature in zip(
                self._components, component_signatures
            ):
                try:
                    artist = None
                    if isinstance(signature, PlotFieldData):
                        # single-field component: match by name
                        name = signature.name
                        new_field = _find_field(frame_plot_data, name)
                        if new_field is not None:
                            artist = component.render(
                                new_field, self.config.figure
                            )
                        else:
                            # fallback: try rendering first available field
                            if frame_plot_data.fields:
                                artist = component.render(
                                    frame_plot_data.fields[0],
                                    self.config.figure,
                                )
                    elif isinstance(signature, (list, tuple)):
                        # vector or multi-field payload: map by element names
                        new_payload = []
                        for elt in signature:
                            if hasattr(elt, "name"):
                                found = _find_field(frame_plot_data, elt.name)
                                if found is not None:
                                    new_payload.append(found)
                        if new_payload:
                            artist = component.render(
                                new_payload, self.config.figure
                            )
                        else:
                            # fallback: render all available fields
                            artist = component.render(
                                frame_plot_data.fields, self.config.figure
                            )
                    else:
                        # unknown payload type: give the component the full PlotData
                        artist = component.render(
                            frame_plot_data, self.config.figure
                        )

                    if artist is not None:
                        # normalize different possible return types into (artists_dict, metadata)
                        artists_dict, metadata = _normalize_render_output(
                            artist
                        )
                        rendered_artists_frame.append((artists_dict, metadata))
                except Exception as exc:
                    # one component's failure does not end the movie, but it is
                    # never silent: a repeated per-frame render error is a frozen
                    # artist masquerading as a finished animation.
                    _warn_once("frame-component", f"frame component render failed: {exc}")
                    continue

            # per-frame immersed-body overlay: remove the previous frame's silhouettes
            # and redraw at THIS frame's poses (read from this frame's checkpoint), so a
            # spinning / tumbling body tracks its rotation across the movie.
            if getattr(self.config.figure, "draw_bodies", False):
                for art in self._body_artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                from simbi.viz.bodies import overlay_bodies_on_slice

                self._body_artists = overlay_bodies_on_slice(
                    self.axes.get("main") if hasattr(self, "axes") else None,
                    file_path,
                    self.config.plot.slice,
                    sim_data.metadata.coord_system,
                )

            # the horizon is fixed (BH at the origin, constant mass), but redraw it per
            # frame so it survives an axis clear between frames.
            if getattr(self.config.figure, "draw_horizon", False):
                for art in self._horizon_artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                from simbi.viz.horizon import overlay_horizon_on_slice

                self._horizon_artists = overlay_horizon_on_slice(
                    self.axes.get("main") if hasattr(self, "axes") else None,
                    sim_data.metadata,
                    self.config.plot.slice,
                    sim_data.metadata.coord_system,
                )

            # update title with current time (only dynamic element)
            main_ax = self.axes.get("main") if hasattr(self, "axes") else None
            if main_ax is not None and frame_plot_data.fields:
                from simbi.viz.formatting import set_title

                try:
                    assert self.fig is not None
                    time = getattr(frame_plot_data.fields[0], "time", None)
                    set_title(main_ax, self.fig, self.config.figure, time)
                except Exception as exc:
                    _warn_once("frame-title", f"frame title update failed: {exc}")

            # redraw canvas for this frame
            if self.fig is not None:
                # use draw_idle for responsive interactive backends
                try:
                    self.fig.canvas.draw_idle()
                except Exception:
                    self.fig.canvas.draw()
            return []

        # render first frame with full formatting (colorbar, limits, labels)
        # subsequent frames only update data and title
        try:
            file_path = files[0]
            sim_data = load_data(file_path)
            field_names = getattr(self.config.plot, "fields", [])
            frame_plot_data = create_plot_data(
                sim_data, field_names, self.config
            )

            from simbi.viz.pipeline.plot_data import PlotData
            from simbi.viz.pipeline.transforms import compose_fields_for_render

            composed_fields = compose_fields_for_render(
                frame_plot_data.fields, self.config
            )

            frame_plot_data = PlotData(
                fields=composed_fields,
                time=frame_plot_data.time,
                dimensions=composed_fields[0].ndim if composed_fields else 0,
                coord_system=frame_plot_data.coord_system,
                hierarchy=frame_plot_data.hierarchy,
            )

            # render components for frame 0
            rendered_artists_frame = []
            for (component, orig_payload), signature in zip(
                self._components, component_signatures
            ):
                try:
                    artist = None
                    if isinstance(signature, PlotFieldData):
                        name = signature.name
                        new_field = _find_field(frame_plot_data, name)
                        if new_field is not None:
                            artist = component.render(
                                new_field, self.config.figure
                            )
                        elif frame_plot_data.fields:
                            artist = component.render(
                                frame_plot_data.fields[0], self.config.figure
                            )
                    elif isinstance(signature, (list, tuple)):
                        new_payload = []
                        for elt in signature:
                            if hasattr(elt, "name"):
                                found = _find_field(frame_plot_data, elt.name)
                                if found is not None:
                                    new_payload.append(found)
                        if new_payload:
                            artist = component.render(
                                new_payload, self.config.figure
                            )
                        else:
                            artist = component.render(
                                frame_plot_data.fields, self.config.figure
                            )
                    else:
                        artist = component.render(
                            frame_plot_data, self.config.figure
                        )

                    if artist is not None:
                        artists_dict, metadata = _normalize_render_output(
                            artist
                        )
                        rendered_artists_frame.append((artists_dict, metadata))
                except Exception as exc:
                    _warn_once("frame-component", f"frame component render failed: {exc}")
                    continue

            # apply full formatting once for first frame
            main_ax = self.axes.get("main") if hasattr(self, "axes") else None
            if main_ax is not None and frame_plot_data.fields:
                try:
                    assert self.fig is not None
                    self.formatter.apply_figure_formatting(
                        self.fig,
                        main_ax,
                        rendered_artists_frame,
                        frame_plot_data.fields[0],
                        coord_system=self.coord_system,
                        xlabel=None,
                        ylabel=None,
                        show_legend=True,
                    )
                except Exception as exc:
                    _warn_once("frame-format", f"frame formatting failed: {exc}")

            if self.fig is not None:
                try:
                    self.fig.canvas.draw_idle()
                except Exception:
                    self.fig.canvas.draw()
            self._frame_failures = 0
        except Exception as exc:
            # a whole-frame failure repeated every frame is a frozen frame-0
            # movie exiting 0: warn on the first, abort after a streak.
            self._frame_failures = getattr(self, "_frame_failures", 0) + 1
            _warn_once("frame-update", f"animation frame update failed: {exc}")
            if self._frame_failures >= 10:
                raise RuntimeError(
                    f"animation aborted: {self._frame_failures} consecutive frame "
                    f"failures (last: {exc})"
                ) from exc

        # construct the animation
        self._anim = FuncAnimation(
            self.fig,
            _update,
            frames=nframes,
            init_func=_init,
            blit=False,
            interval=int(1000 / fps),
        )

    def animate_coordinate_profile(
        self,
        files: Sequence[str],
        fields: Sequence[str],
        config: "VisualizationConfig",
        output_path: str | None = None,
        fps: int = 30,
        save_all_frames: bool = False,
    ):
        """
        create an animation of coordinate-binned profiles from multiple files.

        uses the coordinate profile data pipeline (create_coordinate_profile_data)
        instead of the standard create_plot_data, enabling spherical averaging
        and mass flux calculations across frames.

        args:
          files: ordered list of checkpoint file paths (frames)
          fields: field names to compute profiles for (e.g., ['mdot', 'rho'])
          config: visualization configuration
          output_path: output movie path (e.g., 'out.mp4'). if None, uses './animation.mp4'
          fps: frames per second for the movie
          save_all_frames: if True, keep intermediate frame PNGs
        """
        from typing import List

        from matplotlib.animation import FuncAnimation

        from simbi.viz.pipeline import load_data
        from simbi.viz.pipeline.coord_binning import (
            create_coordinate_profile_data,
        )
        from simbi.viz.types import FieldData as PlotFieldData

        if not files:
            raise ValueError("no input files provided for animation")

        if not self.fig:
            raise RuntimeError("figure must be prepared before animating")

        output_path = output_path or "animation.mp4"
        nframes = len(files)

        # build component signatures from initial payloads
        component_signatures: List[object] = []
        for _, payload in self._components:
            component_signatures.append(payload)

        def _find_field(plot_data, name: str):
            for f in plot_data.fields:
                if f.name == name:
                    return f
            for f in plot_data.fields:
                if f.name.startswith(name):
                    return f
            return None

        def _init():
            if self.fig is None:
                return []
            self.fig.canvas.draw()
            return []

        def _update(i: int):
            file_path = files[i]
            sim_data = load_data(file_path)

            # build frame data through the coordinate profile pipeline
            frame_plot_data = create_coordinate_profile_data(
                sim_data, fields, config
            )

            # dispatch updated payloads to components
            rendered_artists_frame = []
            for (component, orig_payload), signature in zip(
                self._components, component_signatures
            ):
                try:
                    artist = None
                    if isinstance(signature, PlotFieldData):
                        name = signature.name
                        new_field = _find_field(frame_plot_data, name)
                        if new_field is not None:
                            artist = component.render(new_field, config.figure)
                        elif frame_plot_data.fields:
                            artist = component.render(
                                frame_plot_data.fields[0], config.figure
                            )
                    elif isinstance(signature, (list, tuple)):
                        new_payload = []
                        for elt in signature:
                            if hasattr(elt, "name"):
                                found = _find_field(frame_plot_data, elt.name)
                                if found is not None:
                                    new_payload.append(found)
                        if new_payload:
                            artist = component.render(
                                new_payload, config.figure
                            )
                        else:
                            artist = component.render(
                                frame_plot_data.fields, config.figure
                            )
                    else:
                        artist = component.render(
                            frame_plot_data, config.figure
                        )

                    if artist is not None:
                        artists_dict, metadata = _normalize_render_output(
                            artist
                        )
                        rendered_artists_frame.append((artists_dict, metadata))
                except Exception as exc:
                    _warn_once("frame-component", f"frame component render failed: {exc}")
                    continue

            # update title with current time
            main_ax = self.axes.get("main") if hasattr(self, "axes") else None
            if main_ax is not None and frame_plot_data.fields:
                from simbi.viz.formatting import set_title

                try:
                    assert self.fig is not None
                    time = getattr(frame_plot_data.fields[0], "time", None)
                    set_title(main_ax, self.fig, config.figure, time)
                except Exception as exc:
                    _warn_once("frame-title", f"frame title update failed: {exc}")

            if self.fig is not None:
                try:
                    self.fig.canvas.draw_idle()
                except Exception:
                    self.fig.canvas.draw()
            return []

        # render first frame with full formatting
        try:
            file_path = files[0]
            sim_data = load_data(file_path)
            frame_plot_data = create_coordinate_profile_data(
                sim_data, fields, config
            )

            rendered_artists_frame = []
            for (component, orig_payload), signature in zip(
                self._components, component_signatures
            ):
                try:
                    artist = None
                    if isinstance(signature, PlotFieldData):
                        name = signature.name
                        new_field = _find_field(frame_plot_data, name)
                        if new_field is not None:
                            artist = component.render(new_field, config.figure)
                        elif frame_plot_data.fields:
                            artist = component.render(
                                frame_plot_data.fields[0], config.figure
                            )
                    elif isinstance(signature, (list, tuple)):
                        new_payload = []
                        for elt in signature:
                            if hasattr(elt, "name"):
                                found = _find_field(frame_plot_data, elt.name)
                                if found is not None:
                                    new_payload.append(found)
                        if new_payload:
                            artist = component.render(
                                new_payload, config.figure
                            )
                        else:
                            artist = component.render(
                                frame_plot_data.fields, config.figure
                            )
                    else:
                        artist = component.render(
                            frame_plot_data, config.figure
                        )

                    if artist is not None:
                        artists_dict, metadata = _normalize_render_output(
                            artist
                        )
                        rendered_artists_frame.append((artists_dict, metadata))
                except Exception as exc:
                    _warn_once("frame-component", f"frame component render failed: {exc}")
                    continue

            # apply full formatting once for first frame
            main_ax = self.axes.get("main") if hasattr(self, "axes") else None
            if main_ax is not None and frame_plot_data.fields:
                try:
                    assert self.fig is not None
                    self.formatter.apply_figure_formatting(
                        self.fig,
                        main_ax,
                        rendered_artists_frame,
                        frame_plot_data.fields[0],
                        coord_system=self.coord_system,
                        xlabel=None,
                        ylabel=None,
                        show_legend=True,
                    )
                except Exception as exc:
                    _warn_once("frame-format", f"frame formatting failed: {exc}")

            if self.fig is not None:
                try:
                    self.fig.canvas.draw_idle()
                except Exception:
                    self.fig.canvas.draw()
            self._frame_failures = 0
        except Exception as exc:
            # a whole-frame failure repeated every frame is a frozen frame-0
            # movie exiting 0: warn on the first, abort after a streak.
            self._frame_failures = getattr(self, "_frame_failures", 0) + 1
            _warn_once("frame-update", f"animation frame update failed: {exc}")
            if self._frame_failures >= 10:
                raise RuntimeError(
                    f"animation aborted: {self._frame_failures} consecutive frame "
                    f"failures (last: {exc})"
                ) from exc

        # construct the animation
        self._anim = FuncAnimation(
            self.fig,
            _update,
            frames=nframes,
            init_func=_init,
            blit=False,
            interval=int(1000 / fps),
        )

    def tight_layout(self):
        if self.fig:
            self.fig.tight_layout()
