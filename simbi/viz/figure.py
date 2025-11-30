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
#
# why this file changed:
#   - factor normalization logic into a single helper for clarity and testability
#   - document the Figure <-> RenderResult contract in the module header
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
                       from `config.style`.
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
            self.formatter = formatting.FigureFormatter(self.config.style)

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

        # --- PRE-RENDER FORMATTING ---
        formatting.apply_scaling(main_ax, self.config.style)

        # --- RENDER DATA ---
        rendered_artists = []
        for component, data in self._components:
            if not component.initialized:
                raise RuntimeError("Component not initialized before render.")

            # The component renders its artist (components should return RenderResult,
            # but legacy dict/list returns are tolerated)
            result = component.render(data, self.config.style)
            artist_dict, metadata = _normalize_render_output(result)

            # store normalized tuple (artists_dict, metadata) for the formatter
            rendered_artists.append((artist_dict, metadata))

            # set axis limits for quad/polygon plots since relim() doesn't work on mesh collections
            if isinstance(component, (QuadPlotComponent, PolygonPlotComponent)):
                if isinstance(data, FieldData) and data.domain:
                    x_data = (
                        data.domain[1]
                        if len(data.domain) > 1
                        else data.domain[0]
                    )
                    y_data = data.domain[0] if len(data.domain) > 1 else None
                    try:
                        main_ax.set_xlim(x_data.min(), x_data.max())
                        if y_data is not None:
                            main_ax.set_ylim(y_data.min(), y_data.max())
                    except Exception:
                        # don't let domain issues break rendering
                        pass

        try:
            main_ax.relim()
            main_ax.autoscale_view()
        except Exception:
            # some axes (polar, specialized) may not support relim/autoscale_view
            pass

        # Get context from the *first* component
        first_component, first_data = (
            self._components[0] if self._components else (None, None)
        )
        if first_data is None:
            return  # Nothing to format

        # Delegate all figure-level formatting to the FigureFormatter instance.
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
        except Exception:
            # Formatting should never break rendering; swallow errors here.
            pass

    def _format_colorbar(self, ax: Axes, artist: Any, field_data: FieldData):
        # deprecated: colorbar placement is now the responsibility of FigureFormatter.
        # keep a lightweight no-op to avoid accidental direct calls.
        return

    def save(self, path: str):
        from simbi.reader import logger
        from simbi.reader.progress import create_progress_bar

        if not self.fig:
            raise RuntimeError("Figure has not been prepared.")

        clean_path = (
            path.lower()
            .strip()
            .split(".")[0]
            .replace(" ", "_")
            .replace("-", "_")
        )
        if self._anim is not None:
            if not clean_path.endswith((".mp4", ".avi", ".mov", ".gif")):
                clean_path += ".mp4"  # default to mp4 if no extension

            prog_bar = create_progress_bar()
            with prog_bar:
                task = prog_bar.add_task(
                    f"[green]Saving animation to path {clean_path}...",
                    total=self.config.animation.total_frames,
                )

                # beautiful progress callback for animation saving using
                # the rich library if available
                def prog_callback(current: int, total: int) -> None:
                    prog_bar.update(task, advance=1)

                self._anim.save(
                    clean_path,
                    dpi=self.config.style.dpi,
                    progress_callback=prog_callback,
                )
        else:
            if any(
                isinstance(
                    x,
                    (
                        LinePlotComponent,
                        CoordinateProfileComponent,
                        TimeSeriesPlotComponent,
                    ),
                )
                for x, d in self._components
            ):
                if not clean_path.endswith(".pdf"):
                    # save line plots as vector graphics by default
                    path += ".pdf"
            else:
                clean_path += ".png"
            self.fig.savefig(clean_path, dpi=self.config.style.dpi)
            logger.info(f"Figure saved to path: {clean_path}!")

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

        # build a map of the original component payload \"signatures\" so we can
        # request the same plotted fields for each frame.
        component_signatures: List[object] = []
        for component, payload in self._components:
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
                                new_field, self.config.style
                            )
                        else:
                            # fallback: try rendering first available field
                            if frame_plot_data.fields:
                                artist = component.render(
                                    frame_plot_data.fields[0], self.config.style
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
                                new_payload, self.config.style
                            )
                        else:
                            # fallback: render all available fields
                            artist = component.render(
                                frame_plot_data.fields, self.config.style
                            )
                    else:
                        # unknown payload type: give the component the full PlotData
                        artist = component.render(
                            frame_plot_data, self.config.style
                        )

                    if artist is not None:
                        # Normalize different possible return types into (artists_dict, metadata)
                        artists_dict, metadata = _normalize_render_output(
                            artist
                        )
                        rendered_artists_frame.append((artists_dict, metadata))
                except Exception:
                    # do not break the animation loop for a single component failure
                    continue

            # per-frame formatting: delegate to the FigureFormatter so animation
            # matches single-file rendering behavior
            main_ax = self.axes.get("main") if hasattr(self, "axes") else None
            if main_ax is not None:
                # prefer the first available plotted field for formatting context
                first_plot_field = (
                    frame_plot_data.fields[0]
                    if frame_plot_data.fields
                    else None
                )
                try:
                    assert self.fig is not None
                    self.formatter.apply_figure_formatting(
                        self.fig,
                        main_ax,
                        rendered_artists_frame,
                        first_plot_field,
                        coord_system=self.coord_system,
                        xlabel=None,
                        ylabel=None,
                        show_legend=True,
                    )
                except Exception:
                    # don't let formatting errors stop the animation
                    pass

            # redraw canvas for this frame
            if self.fig is not None:
                # use draw_idle for responsive interactive backends
                try:
                    self.fig.canvas.draw_idle()
                except Exception:
                    self.fig.canvas.draw()
            return []

        # render the first frame immediately so the displayed figure is the
        # exact first-frame formatting (and not a blank canvas)
        try:
            _update(0)
            if self.fig is not None:
                try:
                    self.fig.canvas.draw_idle()
                except Exception:
                    self.fig.canvas.draw()
        except Exception:
            # if first frame render fails, continue to construct the animation
            pass

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
