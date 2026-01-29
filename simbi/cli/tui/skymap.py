# =============================================================================
# simbi/cli/tui/skymap.py
#
# interactive terminal ui for skymap generation.
# allows users to explore event data and select parameters before rendering.
# =============================================================================

from dataclasses import dataclass
from typing import Optional

import numpy as np
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Select,
    Static,
)


@dataclass
class skymap_params_t:
    """parameters collected from TUI for skymap generation"""

    time: float
    time_window: float
    energy_min: float
    energy_max: float
    observer_angle: float
    n_theta: int
    n_phi: int
    beam: Optional[float]
    distance: Optional[float]  # Mpc
    output: Optional[str]
    save_fig: Optional[str]


class SkymapTUI(App):
    """interactive TUI for skymap parameter selection"""

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
        width: 20;
        color: $text-muted;
        padding-left: 1;
    }

    #time-select {
        width: 40;
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

    Input:focus {
        border: tall $accent;
    }
    """

    BINDINGS = [
        Binding("ctrl+g", "generate", "Generate", show=True),
        Binding("escape", "quit", "Quit", show=True),
    ]

    def __init__(self, events_file: str, events, meta, **kwargs):
        super().__init__(**kwargs)
        self.events_file = events_file
        self.events = events
        self.meta = meta
        self.result: Optional[skymap_params_t] = None

        # compute arrival times for on-axis observer
        c_cgs = 2.998e10
        day_s = 86400.0
        one_plus_z = 1.0 + meta.z
        observer_dir = np.array([0.0, 0.0, 1.0])
        r_dot_n = (
            events.x * observer_dir[0]
            + events.y * observer_dir[1]
            + events.z * observer_dir[2]
        )
        self.t_arrival = (
            one_plus_z * (events.t_emission - r_dot_n / c_cgs) / day_s
        )

        # compute defaults from data
        mask = ~events.absorbed
        t_filtered = self.t_arrival[mask]
        self.t_10 = np.percentile(t_filtered, 10)
        self.t_50 = np.percentile(t_filtered, 50)
        self.t_90 = np.percentile(t_filtered, 90)
        self.t_min = t_filtered.min()
        self.t_max = t_filtered.max()
        self.e_min = events.energy[mask].min()
        self.e_max = events.energy[mask].max()
        self.n_total = len(events.energy)
        self.n_available = mask.sum()

    def compose(self) -> ComposeResult:
        yield Header()

        with ScrollableContainer(id="main-container"):
            # data summary section
            yield Static("Event Data Summary", classes="section-title")
            yield Static(
                f"File: {self.events_file}\n"
                f"Total events: {self.n_total:,}  |  Available: {self.n_available:,}\n"
                f"Time range: [{self.t_min:.1f}, {self.t_max:.1f}] day\n"
                f"Energy range: [{self.e_min:.2e}, {self.e_max:.2e}] erg",
                classes="data-summary",
            )

            # time selection
            yield Static("Time Selection", classes="section-title")

            yield Select(
                [
                    (f"10th percentile: {self.t_10:.1f} day", self.t_10),
                    (
                        f"50th percentile (median): {self.t_50:.1f} day",
                        self.t_50,
                    ),
                    (f"90th percentile: {self.t_90:.1f} day", self.t_90),
                    ("Custom (enter below)", None),
                ],
                value=self.t_50,
                id="time-select",
                prompt="Select observer time",
            )

            with Horizontal(classes="param-row"):
                yield Static("Custom time:", classes="param-label")
                yield Input(
                    value=f"{self.t_50:.1f}",
                    id="time-input",
                    classes="param-input",
                )
                yield Static("[day]", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Static("Time window:", classes="param-label")
                yield Input(
                    value="10.0", id="time-window", classes="param-input"
                )
                yield Static("[day]", classes="param-hint")

            # energy filter
            yield Static("Energy Filter", classes="section-title")

            with Horizontal(classes="param-row"):
                yield Static("Energy min:", classes="param-label")
                yield Input(
                    value=f"{self.e_min:.2e}",
                    id="energy-min",
                    classes="param-input",
                )
                yield Static("[erg]", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Static("Energy max:", classes="param-label")
                yield Input(
                    value=f"{self.e_max:.2e}",
                    id="energy-max",
                    classes="param-input",
                )
                yield Static("[erg]", classes="param-hint")

            # observer settings
            yield Static("Observer Settings", classes="section-title")

            with Horizontal(classes="param-row"):
                yield Static("Observer angle:", classes="param-label")
                yield Input(
                    value="0.0", id="observer-angle", classes="param-input"
                )
                yield Static("[degrees]", classes="param-hint")

            # resolution
            yield Static("Resolution", classes="section-title")

            with Horizontal(classes="param-row"):
                yield Static("n_theta:", classes="param-label")
                yield Input(value="128", id="n-theta", classes="param-input")
                yield Static("[pixels]", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Static("n_phi:", classes="param-label")
                yield Input(value="256", id="n-phi", classes="param-input")
                yield Static("[pixels]", classes="param-hint")

            # optional settings
            yield Static("Optional", classes="section-title")

            with Horizontal(classes="param-row"):
                yield Static("Distance:", classes="param-label")
                yield Input(
                    value="",
                    placeholder="leave empty to use metadata",
                    id="distance",
                    classes="param-input",
                )
                yield Static("[Mpc] override", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Static("Beam FWHM:", classes="param-label")
                yield Input(
                    value="",
                    placeholder="leave empty for none",
                    id="beam",
                    classes="param-input",
                )
                yield Static("[arcsec]", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Static("Save data:", classes="param-label")
                yield Input(
                    value="",
                    placeholder="leave empty for none",
                    id="output",
                    classes="param-input",
                )
                yield Static("[.h5 file]", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Static("Save figure:", classes="param-label")
                yield Input(
                    value="",
                    placeholder="leave empty to show",
                    id="save-fig",
                    classes="param-input",
                )
                yield Static("[.pdf/.png]", classes="param-hint")

            # buttons
            with Horizontal(id="buttons"):
                yield Button(
                    "Generate Skymap", id="generate", variant="primary"
                )
                yield Button("Cancel", id="cancel", variant="error")

            yield Static("", id="status")

        yield Footer()

    def on_select_changed(self, event: Select.Changed) -> None:
        """update time input when dropdown selection changes"""
        if event.select.id == "time-select" and event.value is not None:
            time_input = self.query_one("#time-input", Input)
            time_input.value = f"{event.value:.1f}"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """handle button clicks"""
        if event.button.id == "generate":
            self.action_generate()
        elif event.button.id == "cancel":
            self.action_quit()

    def _get_float(
        self, field_id: str, default: float = 0.0
    ) -> Optional[float]:
        """safely get float from input field"""
        try:
            inp = self.query_one(f"#{field_id}", Input)
            val = inp.value.strip()
            if not val:
                return None
            return float(val)
        except (ValueError, Exception):
            return None

    def _get_int(self, field_id: str, default: int = 0) -> int:
        """safely get int from input field"""
        try:
            inp = self.query_one(f"#{field_id}", Input)
            return int(inp.value.strip())
        except (ValueError, Exception):
            return default

    def _get_str(self, field_id: str) -> Optional[str]:
        """get string from input field, None if empty"""
        try:
            inp = self.query_one(f"#{field_id}", Input)
            val = inp.value.strip()
            return val if val else None
        except Exception:
            return None

    def action_generate(self) -> None:
        """validate and generate skymap"""
        status = self.query_one("#status", Static)

        # get time
        time = self._get_float("time-input")
        if time is None:
            status.update("[bold red]Error: invalid time value[/]")
            return

        time_window = self._get_float("time-window")
        if time_window is None or time_window <= 0:
            status.update("[bold red]Error: invalid time window[/]")
            return

        energy_min = self._get_float("energy-min")
        if energy_min is None:
            energy_min = self.e_min

        energy_max = self._get_float("energy-max")
        if energy_max is None:
            energy_max = self.e_max

        observer_angle = self._get_float("observer-angle")
        if observer_angle is None:
            observer_angle = 0.0

        n_theta = self._get_int("n-theta", 128)
        n_phi = self._get_int("n-phi", 256)

        distance = self._get_float("distance")
        beam = self._get_float("beam")
        output = self._get_str("output")
        save_fig = self._get_str("save-fig")

        # validate time is in range
        if time < self.t_min or time > self.t_max:
            status.update(
                f"[bold yellow]Warning: time {time:.1f} outside [{self.t_min:.1f}, {self.t_max:.1f}][/]"
            )

        self.result = skymap_params_t(
            time=time,
            time_window=time_window,
            energy_min=energy_min,
            energy_max=energy_max,
            observer_angle=observer_angle,
            n_theta=n_theta,
            n_phi=n_phi,
            beam=beam,
            distance=distance,
            output=output,
            save_fig=save_fig,
        )
        self.exit()

    def action_quit(self) -> None:
        """quit without generating"""
        self.result = None
        self.exit()


def run_skymap_tui(events_file: str, events, meta) -> Optional[skymap_params_t]:
    """
    run the skymap TUI and return selected parameters.

    returns None if user cancels.
    """
    app = SkymapTUI(events_file, events, meta)
    app.run()
    return app.result
