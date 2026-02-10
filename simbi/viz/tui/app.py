# =============================================================================
# simbi/viz/tui/app.py
#
# main tui application for interactive plot configuration.
# collects parameters via widgets, builds plot requests, then exits
# and hands off to matplotlib for rendering. no concurrent event loops.
# =============================================================================
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Rule,
    Select,
    Static,
)

from .widgets.config_panel import ConfigPanel
from .widgets.file_browser import FileBrowser
from .widgets.plot_queue import PlotQueue, plot_request_t
from .widgets.slice_selector import SliceSelector


def _detect_plot_mode(
    files: list[Path],
    effective_ndim: int,
    has_refinement: bool,
) -> tuple[str, str, bool, bool]:
    """determine component type and mode from file selection and data.

    returns (component_type, render_mode, is_animation, is_overlay).
    """
    n = len(files)
    is_animation = n > 1
    is_overlay = False

    if effective_ndim == 1:
        component_type = "line"
        render_mode = "pcolormesh"
    elif effective_ndim >= 2:
        if has_refinement:
            component_type = "polygon"
            render_mode = "polygons"
        else:
            component_type = "quad"
            render_mode = "pcolormesh"
    else:
        component_type = "line"
        render_mode = "pcolormesh"

    return component_type, render_mode, is_animation, is_overlay


class PlotTUI(App):
    """interactive tui for plot parameter selection."""

    TITLE = "simbi plot"

    CSS = """
    Screen {
        background: $surface;
    }

    #main-layout {
        width: 100%;
        height: 100%;
    }

    #left-panel {
        width: 35;
        height: 100%;
        border-right: solid $primary;
        padding: 0 1;
    }

    #right-panel {
        width: 1fr;
        height: 100%;
        padding: 0 1;
    }

    #config-scroll {
        height: 1fr;
    }

    .section-title {
        text-style: bold;
        color: $primary;
        margin-top: 1;
        margin-bottom: 1;
    }

    #data-summary {
        background: $panel;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary;
        height: auto;
    }

    #status-bar {
        dock: bottom;
        height: 2;
        background: $panel;
        padding: 0 1;
        color: $warning;
    }

    #field-select {
        width: 100%;
        margin-bottom: 1;
    }

    #plot-type-select {
        width: 100%;
        margin-bottom: 1;
    }

    #action-buttons {
        height: auto;
        margin-top: 1;
        margin-bottom: 1;
    }

    #action-buttons Button {
        margin-right: 1;
    }

    #right-panel-placeholder {
        width: 1fr;
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("ctrl+a", "add_to_queue", "Add to Queue", show=True),
        Binding("ctrl+d", "add_and_plot", "Add & Plot", show=True),
        Binding("ctrl+p", "plot_all", "Plot Queue", show=True),
        Binding("escape", "quit", "Quit", show=True),
    ]

    def __init__(self, initial_path: Optional[Path] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._initial_path = initial_path
        self._selected_files: list[Path] = []
        self._available_fields: list[str] = []
        self._effective_ndim: int = 1
        self._has_refinement: bool = False
        self._coord_system: str = "cartesian"
        self._coord_ranges: dict[str, tuple[float, float]] = {}
        self.result: Optional[list[plot_request_t]] = None

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="main-layout"):
            with Vertical(id="left-panel"):
                yield FileBrowser(
                    initial_path=self._initial_path, id="file-browser"
                )

            with Vertical(id="right-panel"):
                # placeholder shown before files are selected
                yield Static(
                    "select checkpoint file(s) from the left panel to begin",
                    id="right-panel-placeholder",
                )

                # data summary (hidden initially)
                yield Static("", id="data-summary")

                with ScrollableContainer(id="config-scroll"):
                    yield Label("Plot Configuration", classes="section-title")

                    # field selection
                    yield Select(
                        [("rho", "rho")],
                        value="rho",
                        id="field-select",
                        prompt="Field",
                    )

                    # plot type
                    yield Select(
                        [
                            ("Auto", "auto"),
                            ("Snapshot", "snapshot"),
                            ("Animation", "animation"),
                            ("Overlay", "overlay"),
                            ("Coordinate Profile", "coordinate_bin"),
                            ("Time Series", "time_series"),
                        ],
                        value="auto",
                        id="plot-type-select",
                        prompt="Plot Type",
                    )

                    # slice selector (hidden until 3d data detected)
                    yield SliceSelector(id="slice-selector")

                    # component props panel (populated on file load)
                    yield Static("", id="component-props-container")

                    # action buttons
                    yield Rule()
                    with Horizontal(id="action-buttons"):
                        yield Button(
                            "Add to Queue",
                            id="btn-add-queue",
                            variant="primary",
                        )
                        yield Button(
                            "Add & Plot",
                            id="btn-add-plot",
                            variant="success",
                        )

                    # plot queue
                    yield Rule()
                    yield PlotQueue(id="plot-queue")

        yield Static(
            "workflow: select files -> pick field -> configure -> Add to Queue -> Plot All",
            id="status-bar",
        )
        yield Footer()

    def on_mount(self) -> None:
        # hide config panel until files are selected
        self.query_one("#slice-selector", SliceSelector).display = False
        self.query_one("#data-summary").display = False
        self.query_one("#config-scroll").display = False
        self.query_one("#action-buttons").display = False
        self.query_one("#plot-queue").display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-add-queue":
            self.action_add_to_queue()
        elif event.button.id == "btn-add-plot":
            self.action_add_and_plot()

    def on_file_browser_files_selected(
        self, event: FileBrowser.FilesSelected
    ) -> None:
        self._selected_files = event.files
        if event.files:
            self._introspect_checkpoint(event.files[0])
        else:
            self._clear_data_summary()

    def _introspect_checkpoint(self, path: Path) -> None:
        """load a checkpoint and discover fields, dims, refinement."""
        status = self.query_one("#status-bar", Static)
        status.update("loading checkpoint...")

        try:
            from simbi.viz.pipeline.transforms import load_data

            sim_data = load_data(str(path))
            self._available_fields = sorted(sim_data.available_fields())
            self._effective_ndim = sum(r > 1 for r in sim_data.mesh.shape)
            self._has_refinement = sim_data.has_refinement()
            self._coord_system = sim_data.metadata.coord_system

            # extract coordinate ranges for slice selector
            mesh = sim_data.mesh
            coord_accessors = [
                ("x1", mesh.x1v),
                ("x2", mesh.x2v),
                ("x3", mesh.x3v),
            ]
            self._coord_ranges = {}
            for axis, coords in coord_accessors[: len(mesh.shape)]:
                if len(coords) > 1:
                    self._coord_ranges[axis] = (
                        float(coords[0]),
                        float(coords[-1]),
                    )

            self._update_ui_after_introspection(sim_data)
        except Exception as e:
            status.update(f"error loading checkpoint: {e}")

    def _update_ui_after_introspection(self, sim_data) -> None:
        # show the config panel
        self.query_one("#right-panel-placeholder").display = False
        self.query_one("#data-summary").display = True
        self.query_one("#config-scroll").display = True
        self.query_one("#action-buttons").display = True
        self.query_one("#plot-queue").display = True

        # update data summary
        summary = self.query_one("#data-summary", Static)
        n = len(self._selected_files)
        shape_str = " x ".join(str(s) for s in sim_data.mesh.shape)
        mode = "animation" if n > 1 else "snapshot"
        summary.update(
            f"files: {n} ({mode}) | dims: {self._effective_ndim} | "
            f"shape: {shape_str} | coord: {self._coord_system} | "
            f"AMR: {'yes' if self._has_refinement else 'no'} | "
            f"t = {sim_data.metadata.time:.4g}"
        )

        # update field selector
        field_select = self.query_one("#field-select", Select)
        field_select.set_options([(f, f) for f in self._available_fields])
        if self._available_fields:
            field_select.value = self._available_fields[0]

        # show/hide slice selector
        slice_sel = self.query_one("#slice-selector", SliceSelector)
        if self._effective_ndim >= 3:
            slice_sel.display = True
            slice_sel.set_ranges(self._coord_ranges)
        else:
            slice_sel.display = False

        # populate component props panel
        self._populate_props_panel()

        status = self.query_one("#status-bar", Static)
        status.update(
            f"{len(self._available_fields)} fields | "
            f"pick a field, configure props, then 'Add to Queue' or 'Add & Plot'"
        )

    def _populate_props_panel(self) -> None:
        """create the appropriate props panel based on data dimensionality."""
        container = self.query_one("#component-props-container", Static)

        component_type, _, _, _ = _detect_plot_mode(
            self._selected_files, self._effective_ndim, self._has_refinement
        )

        from simbi.viz.components import (
            LinePlotProps,
            PolygonPlotProps,
            QuadPlotProps,
        )

        props_map = {
            "line": ("Line Props", LinePlotProps),
            "quad": ("Quad Props", QuadPlotProps),
            "polygon": ("Polygon Props", PolygonPlotProps),
        }

        if component_type in props_map:
            title, model = props_map[component_type]
            for old in self.query("ConfigPanel"):
                old.remove()
            panel = ConfigPanel(title, model, id="active-props-panel")
            container.mount(panel)

    def _clear_data_summary(self) -> None:
        self.query_one("#right-panel-placeholder").display = True
        self.query_one("#data-summary").display = False
        self.query_one("#config-scroll").display = False
        self.query_one("#action-buttons").display = False
        status = self.query_one("#status-bar", Static)
        status.update(
            "workflow: select files -> pick field -> configure -> Add to Queue -> Plot All"
        )

    def _build_current_request(self) -> Optional[plot_request_t]:
        """build a plot_request_t from current widget state. returns None on validation failure."""
        if not self._selected_files:
            self.query_one("#status-bar", Static).update(
                "no files selected -- pick files from the left panel"
            )
            return None

        field_select = self.query_one("#field-select", Select)
        field = field_select.value
        if field is None or field == Select.BLANK:
            self.query_one("#status-bar", Static).update(
                "no field selected -- pick a field from the dropdown"
            )
            return None

        # enforce slice for 3d
        slice_config = None
        if self._effective_ndim >= 3:
            slice_sel = self.query_one("#slice-selector", SliceSelector)
            slice_config = slice_sel.get_slice_config()
            if slice_config is None:
                self.query_one("#status-bar", Static).update(
                    "3D data requires a slice -- set axis and position above"
                )
                return None

        # determine plot mode
        plot_type_select = self.query_one("#plot-type-select", Select)
        plot_type = plot_type_select.value

        component_type, render_mode, is_animation, is_overlay = (
            _detect_plot_mode(
                self._selected_files, self._effective_ndim, self._has_refinement
            )
        )

        if plot_type == "overlay":
            is_animation = False
            is_overlay = True
        elif plot_type == "animation":
            is_animation = True
            is_overlay = False
        elif plot_type == "snapshot":
            is_animation = False
            is_overlay = False
        elif plot_type == "coordinate_bin":
            component_type = "coordinate_profile"
        elif plot_type == "time_series":
            component_type = "time_series"

        # collect component props
        component_props = {}
        try:
            panels = self.query("ConfigPanel")
            for panel in panels:
                component_props = panel.collect_values()
        except Exception:
            pass

        return plot_request_t(
            files=list(self._selected_files),
            fields=[str(field)],
            component_type=component_type,
            component_props=component_props,
            figure_props={},
            effective_ndim=self._effective_ndim,
            slice_config=slice_config,
            render_mode=render_mode,
            is_animation=is_animation,
            is_overlay=is_overlay,
        )

    def action_add_to_queue(self) -> None:
        """add current configuration to the plot queue."""
        request = self._build_current_request()
        if request is None:
            return

        queue = self.query_one("#plot-queue", PlotQueue)
        queue.add_request(request)

        self.query_one("#status-bar", Static).update(
            f"added '{request.fields[0]}' to queue -- "
            f"add more or hit 'Plot All' / ctrl+p"
        )

    def action_add_and_plot(self) -> None:
        """add current config to queue and immediately plot everything."""
        request = self._build_current_request()
        if request is None:
            return

        queue = self.query_one("#plot-queue", PlotQueue)
        queue.add_request(request)
        self.result = queue.requests
        self.exit()

    def action_plot_all(self) -> None:
        """plot everything in the queue."""
        queue = self.query_one("#plot-queue", PlotQueue)
        if not queue.requests:
            self.query_one("#status-bar", Static).update(
                "queue is empty -- add something first with 'Add to Queue'"
            )
            return
        self.result = queue.requests
        self.exit()

    def on_plot_queue_plot_all_requested(
        self, event: PlotQueue.PlotAllRequested
    ) -> None:
        self.result = event.requests
        self.exit()

    def action_quit(self) -> None:
        self.result = None
        self.exit()


def _execute_requests(requests: list[plot_request_t]) -> None:
    """execute plot requests after tui exits."""

    import matplotlib.pyplot as plt

    from simbi.viz import api
    from simbi.viz.config import (
        AnimationConfig,
        CoordinateConfig,
        FigureConfig,
        PlotConfig,
        RefinementConfig,
        VisualizationConfig,
    )

    for req in requests:
        files = [str(f) for f in req.files]
        fields = req.fields

        if req.component_type == "coordinate_profile":
            plot_type = "coordinate_bin"
        elif req.component_type == "time_series":
            plot_type = "time_series"
        elif req.component_type == "line":
            plot_type = "line"
        else:
            plot_type = "multidim"

        # ndim after slicing: subtract the number of slice axes
        n_sliced = len(req.slice_config) if req.slice_config else 0
        ndim = max(1, req.effective_ndim - n_sliced)

        config = VisualizationConfig(
            plot=PlotConfig(
                plot_type=plot_type,
                fields=fields,
                ndim=ndim,
                slice=req.slice_config,
            ),
            figure=FigureConfig(**req.figure_props),
            refinement=RefinementConfig(render_mode=req.render_mode),
            coordinate=CoordinateConfig(),
            animation=AnimationConfig(),
        )

        component_props = None
        if req.component_props:
            component_props = _build_component_props(
                req.component_type, req.component_props
            )

        try:
            if req.is_overlay:
                if req.component_type == "coordinate_profile":
                    api.plot_coordinate_profile_overlay(
                        config=config,
                        files=files,
                        fields=fields,
                        component_props=component_props,
                        show=False,
                    )
                else:
                    api.plot_overlay(
                        config=config,
                        files=files,
                        fields=fields,
                        component_props=component_props,
                        show=False,
                    )
            elif req.is_animation:
                if req.component_type == "coordinate_profile":
                    api.animate_coordinate_profile(
                        config=config,
                        files=files,
                        fields=fields,
                        component_props=component_props,
                        show=False,
                    )
                else:
                    api.animate(
                        config=config,
                        files=files,
                        fields=fields,
                        component_props=component_props,
                        show=False,
                    )
            else:
                if req.component_type == "coordinate_profile":
                    api.plot_coordinate_profile(
                        config=config,
                        files=files,
                        fields=fields,
                        component_props=component_props,
                        show=False,
                    )
                elif req.component_type == "time_series":
                    api.plot_time_series(
                        config=config,
                        files=files,
                        fields=fields,
                        component_props=component_props,
                        show=False,
                    )
                else:
                    api.plot(
                        config=config,
                        files=files,
                        fields=fields,
                        component_props=component_props,
                        show=False,
                    )
        except Exception as e:
            print(f"error plotting {fields}: {e}")

    plt.show()


def _build_component_props(component_type: str, values: dict) -> dict:
    """build a component_props dict from collected values."""
    from simbi.viz.components import (
        CoordinateProfileProps,
        LinePlotProps,
        PolygonPlotProps,
        QuadPlotProps,
    )
    from simbi.viz.components.time_series import TimeSeriesPlotProps

    props_map = {
        "line": ("line", LinePlotProps),
        "quad": ("quad", QuadPlotProps),
        "polygon": ("polygon", PolygonPlotProps),
        "coordinate_profile": ("coordinate_profile", CoordinateProfileProps),
        "time_series": ("time_series", TimeSeriesPlotProps),
    }

    if component_type not in props_map:
        return {}

    key, cls = props_map[component_type]
    valid = {k: v for k, v in values.items() if k in cls.model_fields}
    try:
        return {key: cls(**valid)}
    except Exception:
        return {}


def run_plot_tui(initial_path: Optional[Path] = None) -> None:
    """run the plot tui and execute any queued plots."""
    app = PlotTUI(initial_path=initial_path)
    app.run()

    if app.result:
        _execute_requests(app.result)
