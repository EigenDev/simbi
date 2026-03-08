# =============================================================================
# simbi/cli/tui/plot.py
#
# interactive terminal ui for plot parameter selection.
# collects visualization parameters, then hands off to the existing
# viz api for rendering. follows the same pattern as skymap.py.
#
# the form adapts based on plot type and data properties (AMR, ndim).
# each plot type surfaces its component's specific props.
#
# usage:
#   params = run_plot_tui(files)
#   if params is not None:
#       # dispatch to viz api
# =============================================================================
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Select,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
)


@dataclass
class plot_params_t:
    """parameters collected from TUI for plot generation"""

    # data
    fields: list[str] = field(default_factory=lambda: ["rho"])
    plot_type: str = "multidim"
    slice: Optional[dict[str, float]] = None
    overlay: bool = False

    # file filtering
    tmin: Optional[float] = None
    tmax: Optional[float] = None
    stride: int = 1

    # figure
    fig_size: tuple[float, float] = (8.0, 6.0)
    dpi: int = 300
    xlims: Optional[tuple[Optional[float], Optional[float]]] = None
    ylims: Optional[tuple[Optional[float], Optional[float]]] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    xscale: str = "linear"
    yscale: str = "linear"
    title: Optional[str] = None
    transparent: bool = False
    draw_bodies: bool = False
    theme: str = "default"
    use_tex: bool = False

    # output
    save_as: Optional[str] = None
    animate: bool = False
    frame_rate: int = 10

    # 2d shared (quad + polygon)
    cmap: str = "viridis"
    log_scale: bool = False
    color_min: Optional[float] = None
    color_max: Optional[float] = None
    power: float = 1.0
    alpha: float = 1.0

    # quad-specific
    shading: str = "auto"
    show_mesh_grid: bool = False

    # polygon-specific
    show_level_bounds: bool = False

    # line-specific
    linewidth: float = 1.0
    marker: Optional[str] = None

    # coordinate profile
    coord_linestyle: str = "-"
    coord_linewidth: float = 1.0
    coord_normalization: float = 1.0
    coord_rend: float = 0.5
    coord_show_ref_lines: bool = True
    coord_x_scale: str = "linear"
    coord_y_scale: str = "linear"

    # time series
    ts_linestyle: str = "-"
    ts_linewidth: float = 1.0
    ts_marker: Optional[str] = None
    ts_alpha: float = 0.6
    ts_normalization: Optional[float] = None
    ts_show_moving_avg: bool = False
    ts_moving_avg_window: int = 5
    ts_show_trend: bool = False

    # vector overlay
    vector_fields: Optional[list[str]] = None
    vector_type: str = "quiver"
    quiver_color: str = "white"
    quiver_skip: int = 5
    quiver_alpha: float = 1.0
    stream_color: str = "white"
    stream_linewidth: float = 0.5
    stream_density: float = 1.0
    stream_alpha: float = 0.6

    # refinement
    render_mode: str = "pcolormesh"
    composite_view: bool = False
    active_levels: Optional[set[int]] = None

    # binning
    n_bins: int = 64

    # time series weight
    weight: Optional[str] = None

    # time display
    time_scale: Optional[float] = None
    time_units: str = ""


class PlotTUI(App):
    """interactive TUI for plot parameter selection"""

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }

    .section-title {
        text-style: bold;
        color: $primary;
        margin-top: 1;
        margin-bottom: 1;
    }

    .data-summary {
        background: $panel;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary;
    }

    .param-row {
        height: auto;
        margin-bottom: 1;
    }

    .param-label {
        width: 20;
        padding-right: 1;
    }

    .param-input {
        width: 30;
    }

    .param-hint {
        width: 30;
        color: $text-muted;
        padding-left: 1;
    }

    #field-select {
        height: 10;
        margin-bottom: 1;
    }

    #buttons {
        margin-top: 2;
        height: 3;
    }

    #buttons Button {
        margin-right: 2;
    }

    #status {
        margin-top: 1;
        height: 2;
        color: $warning;
    }

    Select {
        width: 40;
        margin-bottom: 1;
    }

    Checkbox {
        margin-bottom: 1;
    }

    Input:focus {
        border: tall $accent;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        Binding("ctrl+p", "plot", "Plot", show=True),
        Binding("escape", "quit", "Quit", show=True),
    ]

    current_plot_type = reactive("multidim")

    def __init__(
        self,
        files: list[Path],
        available_fields: list[str],
        ndim: int,
        coord_system: str,
        has_amr: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.files = files
        self.available_fields = available_fields
        self.ndim = ndim
        self.coord_system = coord_system
        self.has_amr = has_amr
        self.result: Optional[plot_params_t] = None
        self._default_plot_type = "line" if ndim == 1 else "multidim"
        self.current_plot_type = self._default_plot_type

    def compose(self) -> ComposeResult:
        yield Header()

        with ScrollableContainer(id="main-container"):
            yield Static(
                f"Files: {len(self.files)}  |  "
                f"Dimensions: {self.ndim}D  |  "
                f"Coordinates: {self.coord_system}"
                f"{'  |  AMR' if self.has_amr else ''}",
                classes="data-summary",
            )

            with TabbedContent():
                # === data tab ===
                with TabPane("Data", id="tab-data"):
                    yield Static("Fields", classes="section-title")
                    yield SelectionList[str](
                        *[(f, f, f == "rho") for f in self.available_fields],
                        id="field-select",
                    )

                    yield Static("Plot Type", classes="section-title")
                    yield Select(
                        [
                            ("Line (1D)", "line"),
                            ("Multidimensional (2D/3D)", "multidim"),
                            ("Coordinate Profile", "coordinate_bin"),
                            ("Time Series", "time_series"),
                        ],
                        value=self._default_plot_type,
                        id="plot-type",
                        prompt="Select plot type",
                    )

                    yield Checkbox(
                        "Overlay (multiple files, same axes)",
                        id="overlay",
                    )

                    # slicing (for 3D data)
                    if self.ndim >= 3:
                        yield Static(
                            "Slice (required for 3D)",
                            classes="section-title",
                        )
                        with Horizontal(classes="param-row"):
                            yield Static("Slice axis:", classes="param-label")
                            yield Select(
                                [
                                    ("x3 (outermost)", "x3"),
                                    ("x2 (middle)", "x2"),
                                    ("x1 (innermost)", "x1"),
                                ],
                                value="x3",
                                id="slice-axis",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Slice value:", classes="param-label")
                            yield Input(
                                value="0.0",
                                id="slice-value",
                                classes="param-input",
                            )

                    # file filtering
                    yield Static("File Filtering", classes="section-title")
                    with Horizontal(classes="param-row"):
                        yield Static("Min timestep:", classes="param-label")
                        yield Input(
                            value="",
                            placeholder="no minimum",
                            id="tmin",
                            classes="param-input",
                        )
                    with Horizontal(classes="param-row"):
                        yield Static("Max timestep:", classes="param-label")
                        yield Input(
                            value="",
                            placeholder="no maximum",
                            id="tmax",
                            classes="param-input",
                        )
                    with Horizontal(classes="param-row"):
                        yield Static("Stride:", classes="param-label")
                        yield Input(
                            value="1",
                            id="stride",
                            classes="param-input",
                        )
                        yield Static(
                            "[use every Nth file]", classes="param-hint"
                        )

                # === figure tab ===
                with TabPane("Figure", id="tab-figure"):
                    yield Static("Layout", classes="section-title")
                    with Horizontal(classes="param-row"):
                        yield Static("Width:", classes="param-label")
                        yield Input(
                            value="8.0",
                            id="fig-width",
                            classes="param-input",
                        )
                    with Horizontal(classes="param-row"):
                        yield Static("Height:", classes="param-label")
                        yield Input(
                            value="6.0",
                            id="fig-height",
                            classes="param-input",
                        )
                    with Horizontal(classes="param-row"):
                        yield Static("DPI:", classes="param-label")
                        yield Input(
                            value="300", id="dpi", classes="param-input"
                        )
                    with Horizontal(classes="param-row"):
                        yield Static("Title:", classes="param-label")
                        yield Input(
                            value="",
                            placeholder="auto",
                            id="title",
                            classes="param-input",
                        )

                    yield Static("Axes", classes="section-title")
                    with Horizontal(classes="param-row"):
                        yield Static("X label:", classes="param-label")
                        yield Input(
                            value="",
                            placeholder="auto",
                            id="xlabel",
                            classes="param-input",
                        )
                    with Horizontal(classes="param-row"):
                        yield Static("Y label:", classes="param-label")
                        yield Input(
                            value="",
                            placeholder="auto",
                            id="ylabel",
                            classes="param-input",
                        )
                    with Horizontal(classes="param-row"):
                        yield Static("X limits:", classes="param-label")
                        yield Input(
                            value="",
                            placeholder="min",
                            id="xlim-min",
                            classes="param-input",
                        )
                        yield Input(
                            value="",
                            placeholder="max",
                            id="xlim-max",
                            classes="param-input",
                        )
                    with Horizontal(classes="param-row"):
                        yield Static("Y limits:", classes="param-label")
                        yield Input(
                            value="",
                            placeholder="min",
                            id="ylim-min",
                            classes="param-input",
                        )
                        yield Input(
                            value="",
                            placeholder="max",
                            id="ylim-max",
                            classes="param-input",
                        )

                    yield Static("Scales", classes="section-title")
                    with Horizontal(classes="param-row"):
                        yield Static("X scale:", classes="param-label")
                        yield Select(
                            [
                                ("Linear", "linear"),
                                ("Log", "log"),
                                ("Symlog", "symlog"),
                                ("Asinh", "asinh"),
                            ],
                            value="linear",
                            id="xscale",
                        )
                    with Horizontal(classes="param-row"):
                        yield Static("Y scale:", classes="param-label")
                        yield Select(
                            [
                                ("Linear", "linear"),
                                ("Log", "log"),
                                ("Symlog", "symlog"),
                                ("Asinh", "asinh"),
                            ],
                            value="linear",
                            id="yscale",
                        )

                    yield Static("Style", classes="section-title")
                    yield Select(
                        [
                            ("Default", "default"),
                            ("Dark", "dark"),
                            ("Scientific", "scientific"),
                        ],
                        value="default",
                        id="theme",
                        prompt="Select theme",
                    )
                    yield Checkbox("Transparent background", id="transparent")
                    yield Checkbox("Draw immersed bodies", id="draw-bodies")
                    yield Checkbox("Use LaTeX rendering", id="use-tex")

                # === component tab (reactive) ===
                with TabPane("Component", id="tab-component"):
                    # -- multidim (2d) props --
                    with Vertical(
                        id="multidim-props",
                        classes=""
                        if self._default_plot_type == "multidim"
                        else "hidden",
                    ):
                        yield Static(
                            "2D Plot Properties"
                            + (
                                " (polygons only, AMR detected)"
                                if self.has_amr
                                else ""
                            ),
                            classes="section-title",
                        )
                        with Horizontal(classes="param-row"):
                            yield Static("Colormap:", classes="param-label")
                            yield Input(
                                value="viridis",
                                id="cmap",
                                classes="param-input",
                            )
                        yield Checkbox("Log scale", id="log-scale")
                        with Horizontal(classes="param-row"):
                            yield Static("Color min:", classes="param-label")
                            yield Input(
                                value="",
                                placeholder="auto",
                                id="color-min",
                                classes="param-input",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Color max:", classes="param-label")
                            yield Input(
                                value="",
                                placeholder="auto",
                                id="color-max",
                                classes="param-input",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Power:", classes="param-label")
                            yield Input(
                                value="1.0",
                                id="power",
                                classes="param-input",
                            )
                            yield Static(
                                "[data^power scaling]",
                                classes="param-hint",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Alpha:", classes="param-label")
                            yield Input(
                                value="1.0",
                                id="alpha-2d",
                                classes="param-input",
                            )

                        if not self.has_amr:
                            yield Static("Render Mode", classes="section-title")
                            yield Select(
                                [
                                    ("Pcolormesh", "pcolormesh"),
                                    ("Polygons", "polygons"),
                                ],
                                value="pcolormesh",
                                id="render-mode",
                            )
                            with Horizontal(classes="param-row"):
                                yield Static("Shading:", classes="param-label")
                                yield Select(
                                    [
                                        ("Auto", "auto"),
                                        ("Nearest", "nearest"),
                                        ("Gouraud", "gouraud"),
                                        ("Flat", "flat"),
                                    ],
                                    value="auto",
                                    id="shading",
                                )
                            yield Checkbox(
                                "Show mesh grid", id="show-mesh-grid"
                            )
                        else:
                            yield Checkbox(
                                "Show level bounds",
                                id="show-level-bounds",
                            )
                            yield Checkbox(
                                "Composite view", id="composite-view"
                            )
                            with Horizontal(classes="param-row"):
                                yield Static(
                                    "Active levels:",
                                    classes="param-label",
                                )
                                yield Input(
                                    value="",
                                    placeholder="all (e.g. 0 1 2)",
                                    id="active-levels",
                                    classes="param-input",
                                )

                        yield Static("Vector Overlay", classes="section-title")
                        with Horizontal(classes="param-row"):
                            yield Static("Components:", classes="param-label")
                            yield Input(
                                value="",
                                placeholder="e.g. v1 v2",
                                id="vector-fields",
                                classes="param-input",
                            )
                            yield Static(
                                "[space-separated pair]",
                                classes="param-hint",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Vector type:", classes="param-label")
                            yield Select(
                                [
                                    ("Quiver", "quiver"),
                                    ("Streamlines", "stream"),
                                ],
                                value="quiver",
                                id="vector-type",
                            )

                        # quiver props
                        with Vertical(id="quiver-props"):
                            with Horizontal(classes="param-row"):
                                yield Static(
                                    "Arrow color:",
                                    classes="param-label",
                                )
                                yield Input(
                                    value="white",
                                    id="quiver-color",
                                    classes="param-input",
                                )
                            with Horizontal(classes="param-row"):
                                yield Static("Skip:", classes="param-label")
                                yield Input(
                                    value="5",
                                    id="quiver-skip",
                                    classes="param-input",
                                )
                                yield Static(
                                    "[plot every Nth vector]",
                                    classes="param-hint",
                                )
                            with Horizontal(classes="param-row"):
                                yield Static(
                                    "Arrow alpha:",
                                    classes="param-label",
                                )
                                yield Input(
                                    value="1.0",
                                    id="quiver-alpha",
                                    classes="param-input",
                                )

                        # stream props
                        with Vertical(id="stream-props", classes="hidden"):
                            with Horizontal(classes="param-row"):
                                yield Static(
                                    "Line color:",
                                    classes="param-label",
                                )
                                yield Input(
                                    value="white",
                                    id="stream-color",
                                    classes="param-input",
                                )
                            with Horizontal(classes="param-row"):
                                yield Static(
                                    "Line width:",
                                    classes="param-label",
                                )
                                yield Input(
                                    value="0.5",
                                    id="stream-linewidth",
                                    classes="param-input",
                                )
                            with Horizontal(classes="param-row"):
                                yield Static("Density:", classes="param-label")
                                yield Input(
                                    value="1.0",
                                    id="stream-density",
                                    classes="param-input",
                                )
                            with Horizontal(classes="param-row"):
                                yield Static("Alpha:", classes="param-label")
                                yield Input(
                                    value="0.6",
                                    id="stream-alpha",
                                    classes="param-input",
                                )

                    # -- line props --
                    with Vertical(
                        id="line-props",
                        classes=""
                        if self._default_plot_type == "line"
                        else "hidden",
                    ):
                        yield Static(
                            "Line Plot Properties", classes="section-title"
                        )
                        with Horizontal(classes="param-row"):
                            yield Static("Line width:", classes="param-label")
                            yield Input(
                                value="2.0",
                                id="line-linewidth",
                                classes="param-input",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Marker:", classes="param-label")
                            yield Input(
                                value="",
                                placeholder="none (e.g. o, s, ^)",
                                id="line-marker",
                                classes="param-input",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Alpha:", classes="param-label")
                            yield Input(
                                value="1.0",
                                id="line-alpha",
                                classes="param-input",
                            )

                    # -- coordinate profile props --
                    with Vertical(id="coord-props", classes="hidden"):
                        yield Static(
                            "Coordinate Profile Properties",
                            classes="section-title",
                        )
                        with Horizontal(classes="param-row"):
                            yield Static("Bins:", classes="param-label")
                            yield Input(
                                value="64",
                                id="n-bins",
                                classes="param-input",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Line style:", classes="param-label")
                            yield Select(
                                [
                                    ("Solid", "-"),
                                    ("Dashed", "--"),
                                    ("Dotted", ":"),
                                    ("Dash-dot", "-."),
                                ],
                                value="-",
                                id="coord-linestyle",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Line width:", classes="param-label")
                            yield Input(
                                value="2.0",
                                id="coord-linewidth",
                                classes="param-input",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static(
                                "Normalization:", classes="param-label"
                            )
                            yield Input(
                                value="1.0",
                                id="coord-norm",
                                classes="param-input",
                            )
                        yield Checkbox(
                            "Show reference lines",
                            id="coord-show-ref",
                            value=True,
                        )
                        with Horizontal(classes="param-row"):
                            yield Static("Ref line end:", classes="param-label")
                            yield Input(
                                value="0.5",
                                id="coord-rend",
                                classes="param-input",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("X scale:", classes="param-label")
                            yield Select(
                                [
                                    ("Linear", "linear"),
                                    ("Log", "log"),
                                ],
                                value="linear",
                                id="coord-xscale",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Y scale:", classes="param-label")
                            yield Select(
                                [
                                    ("Linear", "linear"),
                                    ("Log", "log"),
                                ],
                                value="linear",
                                id="coord-yscale",
                            )

                    # -- time series props --
                    with Vertical(id="ts-props", classes="hidden"):
                        yield Static(
                            "Time Series Properties",
                            classes="section-title",
                        )
                        with Horizontal(classes="param-row"):
                            yield Static("Weight field:", classes="param-label")
                            yield Input(
                                value="",
                                placeholder="none",
                                id="weight",
                                classes="param-input",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Line style:", classes="param-label")
                            yield Select(
                                [
                                    ("Solid", "-"),
                                    ("Dashed", "--"),
                                    ("Dotted", ":"),
                                    ("Dash-dot", "-."),
                                ],
                                value="-",
                                id="ts-linestyle",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Line width:", classes="param-label")
                            yield Input(
                                value="2.0",
                                id="ts-linewidth",
                                classes="param-input",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Marker:", classes="param-label")
                            yield Input(
                                value="",
                                placeholder="none",
                                id="ts-marker",
                                classes="param-input",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static("Alpha:", classes="param-label")
                            yield Input(
                                value="0.6",
                                id="ts-alpha",
                                classes="param-input",
                            )
                        with Horizontal(classes="param-row"):
                            yield Static(
                                "Normalization:", classes="param-label"
                            )
                            yield Input(
                                value="",
                                placeholder="none",
                                id="ts-norm",
                                classes="param-input",
                            )
                        yield Checkbox("Show moving average", id="ts-show-ma")
                        with Horizontal(classes="param-row"):
                            yield Static("MA window:", classes="param-label")
                            yield Input(
                                value="5",
                                id="ts-ma-window",
                                classes="param-input",
                            )
                        yield Checkbox("Show trend line", id="ts-show-trend")

                # === output tab ===
                with TabPane("Output", id="tab-output"):
                    yield Static("Save", classes="section-title")
                    with Horizontal(classes="param-row"):
                        yield Static("Save as:", classes="param-label")
                        yield Input(
                            value="",
                            placeholder="leave empty to display",
                            id="save-as",
                            classes="param-input",
                        )
                        yield Static("[.png/.pdf/.mp4]", classes="param-hint")

                    yield Static("Animation", classes="section-title")
                    yield Checkbox("Animate", id="animate")
                    with Horizontal(classes="param-row"):
                        yield Static("Frame rate:", classes="param-label")
                        yield Input(
                            value="10",
                            id="frame-rate",
                            classes="param-input",
                        )

                    yield Static("Time Display", classes="section-title")
                    with Horizontal(classes="param-row"):
                        yield Static("Time scale:", classes="param-label")
                        yield Input(
                            value="",
                            placeholder="e.g. 4pi, 1e6",
                            id="time-scale",
                            classes="param-input",
                        )
                    with Horizontal(classes="param-row"):
                        yield Static("Time units:", classes="param-label")
                        yield Input(
                            value="",
                            placeholder="e.g. yr, s",
                            id="time-units",
                            classes="param-input",
                        )

            # buttons (outside tabs)
            with Horizontal(id="buttons"):
                yield Button("Plot", id="plot-btn", variant="primary")
                yield Button("Cancel", id="cancel", variant="error")

            yield Static("", id="status")

        yield Footer()

    # -- reactivity --

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "plot-type" and event.value is not Select.BLANK:
            self.current_plot_type = event.value
        elif (
            event.select.id == "vector-type" and event.value is not Select.BLANK
        ):
            self._toggle_vector_props(event.value)

    def watch_current_plot_type(self, plot_type: str) -> None:
        sections = {
            "multidim": "multidim-props",
            "line": "line-props",
            "coordinate_bin": "coord-props",
            "time_series": "ts-props",
        }
        for key, widget_id in sections.items():
            try:
                widget = self.query_one(f"#{widget_id}")
                if key == plot_type:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

    def _toggle_vector_props(self, vector_type: str) -> None:
        try:
            quiver = self.query_one("#quiver-props")
            stream = self.query_one("#stream-props")
            if vector_type == "quiver":
                quiver.remove_class("hidden")
                stream.add_class("hidden")
            else:
                quiver.add_class("hidden")
                stream.remove_class("hidden")
        except Exception:
            pass

    # -- event handlers --

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "plot-btn":
            self.action_plot()
        elif event.button.id == "cancel":
            self.action_quit()

    # -- helpers --

    def _get_str(self, field_id: str) -> Optional[str]:
        try:
            val = self.query_one(f"#{field_id}", Input).value.strip()
            return val if val else None
        except Exception:
            return None

    def _get_float(
        self, field_id: str, default: Optional[float] = None
    ) -> Optional[float]:
        try:
            val = self.query_one(f"#{field_id}", Input).value.strip()
            if not val:
                return default
            return float(val)
        except (ValueError, Exception):
            return default

    def _get_int(self, field_id: str, default: int = 0) -> int:
        try:
            val = self.query_one(f"#{field_id}", Input).value.strip()
            if not val:
                return default
            return int(val)
        except (ValueError, Exception):
            return default

    def _get_select(self, field_id: str, default: str = "") -> str:
        try:
            val = self.query_one(f"#{field_id}", Select).value
            return val if val is not Select.BLANK else default
        except Exception:
            return default

    def _get_checkbox(self, field_id: str) -> bool:
        try:
            return self.query_one(f"#{field_id}", Checkbox).value
        except Exception:
            return False

    def _parse_limits(
        self, min_id: str, max_id: str
    ) -> Optional[tuple[Optional[float], Optional[float]]]:
        lo = self._get_float(min_id)
        hi = self._get_float(max_id)
        if lo is None and hi is None:
            return None
        return (lo, hi)

    def _parse_time_scale(self) -> Optional[float]:
        ts_str = self._get_str("time-scale")
        if not ts_str:
            return None
        import math
        import re

        try:
            parts = re.findall(r"[\d\.]+|[^\d\.]+", ts_str)
            for ii, part in enumerate(parts):
                if part == "pi":
                    parts[ii] = math.pi
                elif part == "e":
                    parts[ii] = math.e
                else:
                    parts[ii] = float(part)
            return math.prod(parts)
        except Exception:
            return None

    # -- actions --

    def action_plot(self) -> None:
        status = self.query_one("#status", Static)

        field_select = self.query_one("#field-select", SelectionList)
        selected = list(field_select.selected)
        if not selected:
            status.update("[bold red]Error: select at least one field[/]")
            return

        plot_type = self._get_select("plot-type", self._default_plot_type)

        if plot_type == "multidim" and len(selected) > 1:
            status.update(
                "[bold red]Error: 2D plots support only one field at a time[/]"
            )
            return

        # parse slice for 3D data
        slice_spec = None
        if self.ndim >= 3:
            axis = self._get_select("slice-axis", "x3")
            val = self._get_float("slice-value", 0.0)
            slice_spec = {axis: val}

        # parse vector fields
        vector_str = self._get_str("vector-fields")
        vector_fields = None
        if vector_str:
            parts = vector_str.split()
            if len(parts) != 2:
                status.update(
                    "[bold red]Error: vector fields needs exactly "
                    "2 components[/]"
                )
                return
            vector_fields = parts

        # parse active levels
        active_levels = None
        levels_str = self._get_str("active-levels")
        if levels_str:
            try:
                active_levels = set(int(x) for x in levels_str.split())
            except ValueError:
                status.update(
                    "[bold red]Error: active levels must be integers[/]"
                )
                return

        # parse time scale
        time_scale = self._parse_time_scale()
        if self._get_str("time-scale") and time_scale is None:
            status.update("[bold red]Error: invalid time scale value[/]")
            return

        # determine render mode
        if self.has_amr:
            render_mode = "polygons"
        else:
            render_mode = self._get_select("render-mode", "pcolormesh")

        self.result = plot_params_t(
            fields=selected,
            plot_type=plot_type,
            slice=slice_spec,
            overlay=self._get_checkbox("overlay"),
            tmin=self._get_float("tmin"),
            tmax=self._get_float("tmax"),
            stride=self._get_int("stride", 1),
            fig_size=(
                self._get_float("fig-width", 8.0),
                self._get_float("fig-height", 6.0),
            ),
            dpi=self._get_int("dpi", 300),
            xlims=self._parse_limits("xlim-min", "xlim-max"),
            ylims=self._parse_limits("ylim-min", "ylim-max"),
            xlabel=self._get_str("xlabel"),
            ylabel=self._get_str("ylabel"),
            xscale=self._get_select("xscale", "linear"),
            yscale=self._get_select("yscale", "linear"),
            title=self._get_str("title"),
            transparent=self._get_checkbox("transparent"),
            draw_bodies=self._get_checkbox("draw-bodies"),
            theme=self._get_select("theme", "default"),
            use_tex=self._get_checkbox("use-tex"),
            save_as=self._get_str("save-as"),
            animate=self._get_checkbox("animate"),
            frame_rate=self._get_int("frame-rate", 10),
            # 2d props
            cmap=self._get_str("cmap") or "viridis",
            log_scale=self._get_checkbox("log-scale"),
            color_min=self._get_float("color-min"),
            color_max=self._get_float("color-max"),
            power=self._get_float("power", 1.0),
            alpha=self._get_float("alpha-2d", 1.0),
            shading=self._get_select("shading", "auto"),
            show_mesh_grid=self._get_checkbox("show-mesh-grid"),
            show_level_bounds=self._get_checkbox("show-level-bounds"),
            # line props
            linewidth=self._get_float("line-linewidth", 1.0),
            marker=self._get_str("line-marker"),
            # coord profile props
            coord_linestyle=self._get_select("coord-linestyle", "-"),
            coord_linewidth=self._get_float("coord-linewidth", 1.0),
            coord_normalization=self._get_float("coord-norm", 1.0),
            coord_rend=self._get_float("coord-rend", 0.5),
            coord_show_ref_lines=self._get_checkbox("coord-show-ref"),
            coord_x_scale=self._get_select("coord-xscale", "linear"),
            coord_y_scale=self._get_select("coord-yscale", "linear"),
            # time series props
            ts_linestyle=self._get_select("ts-linestyle", "-"),
            ts_linewidth=self._get_float("ts-linewidth", 1.0),
            ts_marker=self._get_str("ts-marker"),
            ts_alpha=self._get_float("ts-alpha", 0.6),
            ts_normalization=self._get_float("ts-norm"),
            ts_show_moving_avg=self._get_checkbox("ts-show-ma"),
            ts_moving_avg_window=self._get_int("ts-ma-window", 5),
            ts_show_trend=self._get_checkbox("ts-show-trend"),
            # vector props
            vector_fields=vector_fields,
            vector_type=self._get_select("vector-type", "quiver"),
            quiver_color=self._get_str("quiver-color") or "white",
            quiver_skip=self._get_int("quiver-skip", 5),
            quiver_alpha=self._get_float("quiver-alpha", 1.0),
            stream_color=self._get_str("stream-color") or "white",
            stream_linewidth=self._get_float("stream-linewidth", 0.5),
            stream_density=self._get_float("stream-density", 1.0),
            stream_alpha=self._get_float("stream-alpha", 0.6),
            # refinement
            render_mode=render_mode,
            composite_view=self._get_checkbox("composite-view"),
            active_levels=active_levels,
            # binning
            n_bins=self._get_int("n-bins", 64),
            # time series weight
            weight=self._get_str("weight"),
            # time display
            time_scale=time_scale,
            time_units=self._get_str("time-units") or "",
        )
        self.exit()

    def action_quit(self) -> None:
        self.result = None
        self.exit()


def config_from_plot_params(params: plot_params_t, files: list[Path]) -> tuple:
    """
    convert plot_params_t to (VisualizationConfig, component_props dict).
    """
    from simbi.viz.config import (
        AnimationConfig,
        CoordinateConfig,
        FigureConfig,
        PlotConfig,
        RefinementConfig,
        TimeSeriesConfig,
        VisualizationConfig,
    )
    from simbi.viz.styling import ThemeManager
    from simbi.viz.types import Bounds, ColorRange
    from simbi.viz.utility import get_dimensionality

    ndim = get_dimensionality(files)

    def to_bounds(pair):
        if pair is None:
            return None
        return Bounds(min=pair[0], max=pair[1])

    config = VisualizationConfig(
        plot=PlotConfig(
            plot_type=params.plot_type,
            fields=params.fields,
            ndim=ndim,
            slice=params.slice,
        ),
        figure=FigureConfig(
            fig_size=params.fig_size,
            dpi=params.dpi,
            xlims=to_bounds(params.xlims),
            ylims=to_bounds(params.ylims),
            xlabel=params.xlabel,
            ylabel=params.ylabel,
            xscale=params.xscale,
            yscale=params.yscale,
            title=params.title,
            draw_bodies=params.draw_bodies,
            transparent=params.transparent,
            time_scale=params.time_scale,
            time_units=params.time_units,
        ),
        refinement=RefinementConfig(
            render_mode=params.render_mode,
            composite_view=params.composite_view,
            active_levels=params.active_levels,
        ),
        coordinate=CoordinateConfig(n_bins=params.n_bins),
        time_series=TimeSeriesConfig(weight=params.weight),
        animation=AnimationConfig(
            total_frames=len(files),
            frame_rate=params.frame_rate,
        ),
        theme=ThemeManager.get_theme(params.theme),
    )

    # build component props based on plot type
    component_props = {}
    color_range = ColorRange(min=params.color_min, max=params.color_max)

    if params.plot_type in ("multidim", "line"):
        if params.render_mode == "polygons":
            from simbi.viz.components.polygons import PolygonPlotProps

            component_props["polygon"] = PolygonPlotProps(
                cmap=params.cmap,
                log_scale=params.log_scale,
                color_range=color_range,
                power=params.power,
                alpha=params.alpha,
                show_level_bounds=params.show_level_bounds,
            )
        else:
            from simbi.viz.components.quad import QuadPlotProps

            component_props["quad"] = QuadPlotProps(
                cmap=params.cmap,
                log_scale=params.log_scale,
                color_range=color_range,
                power=params.power,
                alpha=params.alpha,
                shading=params.shading,
                show_mesh_grid=params.show_mesh_grid,
            )

        if params.plot_type == "line":
            from simbi.viz.components.line import LinePlotProps

            component_props["line"] = LinePlotProps(
                linewidth=params.linewidth,
                marker=params.marker,
                alpha=params.alpha,
            )

        # vector overlay props
        if params.vector_fields:
            if params.vector_type == "quiver":
                from simbi.viz.components.quiver import QuiverPlotProps

                component_props["quiver"] = QuiverPlotProps(
                    color=params.quiver_color,
                    skip=params.quiver_skip,
                    alpha=params.quiver_alpha,
                )
            else:
                from simbi.viz.components.stream import StreamPlotProps

                component_props["stream"] = StreamPlotProps(
                    color=params.stream_color,
                    linewidth=params.stream_linewidth,
                    density=params.stream_density,
                    alpha=params.stream_alpha,
                )

    elif params.plot_type == "coordinate_bin":
        from simbi.viz.components.coord_binning import (
            CoordinateProfileProps,
        )

        component_props["coordinate_profile"] = CoordinateProfileProps(
            linestyle=params.coord_linestyle,
            linewidth=params.coord_linewidth,
            normalization=params.coord_normalization,
            rend=params.coord_rend,
            show_reference_lines=params.coord_show_ref_lines,
        )

    elif params.plot_type == "time_series":
        from simbi.viz.components.time_series import TimeSeriesPlotProps

        component_props["time_series"] = TimeSeriesPlotProps(
            linestyle=params.ts_linestyle,
            linewidth=params.ts_linewidth,
            marker=params.ts_marker,
            alpha=params.ts_alpha,
            normalization=params.ts_normalization,
            show_moving_average=params.ts_show_moving_avg,
            moving_average_window=params.ts_moving_avg_window,
            show_trend=params.ts_show_trend,
        )

    return config, component_props


def filter_files_from_params(
    files: list[Path], params: plot_params_t
) -> list[Path]:
    """apply tmin/tmax/stride filtering from TUI params."""
    from simbi.viz.cli import filter_files

    return filter_files(
        files,
        tmin=params.tmin,
        tmax=params.tmax,
        stride=params.stride,
    )


def run_plot_tui(files: list[Path]) -> Optional[plot_params_t]:
    """
    run the plot TUI and return selected parameters.

    reads the first checkpoint to detect available fields, dimensionality,
    coordinate system, and AMR status, then launches the interactive TUI.

    returns None if user cancels.
    """
    from simbi.reader import read_simulation

    data = read_simulation(str(files[0]))
    available_fields = sorted(data.available_fields(level=0))
    ndim = data.metadata.dimensions
    coord_system = data.metadata.coord_system
    has_amr = data.has_refinement()

    app = PlotTUI(
        files=files,
        available_fields=available_fields,
        ndim=ndim,
        coord_system=coord_system,
        has_amr=has_amr,
    )
    app.run()
    return app.result
