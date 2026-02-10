# =============================================================================
# simbi/viz/tui/widgets/plot_queue.py
#
# list of queued plot configurations.
# each entry stores files, field, component props, figure config, and
# optional slice config. "plot all" iterates the queue and calls the
# appropriate api function per entry.
# =============================================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Label, ListItem, ListView, Static


@dataclass
class plot_request_t:
    """a single plot configuration in the queue."""

    files: list[Path]
    fields: list[str]
    component_type: str
    component_props: dict
    figure_props: dict
    effective_ndim: int = 1
    slice_config: Optional[dict[str, float]] = None
    render_mode: str = "pcolormesh"
    is_animation: bool = False
    is_overlay: bool = False


class PlotQueue(Static):
    """manages a list of plot requests."""

    class PlotAllRequested(Message):
        """emitted when user clicks plot all."""

        def __init__(self, requests: list[plot_request_t]) -> None:
            super().__init__()
            self.requests = requests

    DEFAULT_CSS = """
    PlotQueue {
        height: auto;
        max-height: 12;
        width: 100%;
    }

    PlotQueue #queue-list {
        height: auto;
        max-height: 6;
        border: solid $primary;
    }

    PlotQueue #queue-buttons {
        height: 3;
        margin-top: 1;
    }

    PlotQueue #queue-buttons Button {
        min-width: 10;
        margin-right: 1;
    }

    PlotQueue .queue-item {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._requests: list[plot_request_t] = []

    def compose(self) -> ComposeResult:
        yield Label("Plot Queue", classes="section-title")
        yield ListView(id="queue-list")
        with Horizontal(id="queue-buttons"):
            yield Button("Remove", id="queue-remove", variant="error")
            yield Button("Clear", id="queue-clear", variant="error")
            yield Button("Plot All", id="queue-plot-all", variant="success")

    def add_request(self, request: plot_request_t) -> None:
        self._requests.append(request)
        self._refresh_list()

    def _refresh_list(self) -> None:
        list_view = self.query_one("#queue-list", ListView)
        list_view.clear()
        for ii, req in enumerate(self._requests):
            fields_str = ", ".join(req.fields)
            mode = (
                "animate"
                if req.is_animation
                else "overlay"
                if req.is_overlay
                else "plot"
            )
            n_files = len(req.files)
            summary = (
                f"{ii + 1}. [{mode}] {fields_str} "
                f"({req.component_type}, {n_files} file{'s' if n_files != 1 else ''})"
            )
            list_view.append(ListItem(Label(summary), classes="queue-item"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "queue-remove":
            list_view = self.query_one("#queue-list", ListView)
            idx = list_view.index
            if idx is not None and 0 <= idx < len(self._requests):
                self._requests.pop(idx)
                self._refresh_list()
        elif event.button.id == "queue-clear":
            self._requests.clear()
            self._refresh_list()
        elif event.button.id == "queue-plot-all":
            if self._requests:
                self.post_message(self.PlotAllRequested(list(self._requests)))

    @property
    def requests(self) -> list[plot_request_t]:
        return list(self._requests)
