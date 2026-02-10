# =============================================================================
# simbi/viz/tui/widgets/slice_selector.py
#
# 3d slice axis/position picker.
# when data is effectively 3d, the user must pick a slice axis and position
# before adding a plot to the queue. coordinate ranges are shown from the
# checkpoint data.
# =============================================================================
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Label, Select, Static


class SliceSelector(Static):
    """slice configuration for 3d data."""

    DEFAULT_CSS = """
    SliceSelector {
        height: auto;
        width: 100%;
        padding: 0 1;
    }

    SliceSelector .field-row {
        height: auto;
        margin-bottom: 1;
    }

    SliceSelector .field-label {
        width: 22;
        padding-right: 1;
    }

    SliceSelector .field-input {
        width: 1fr;
    }

    SliceSelector .field-select {
        width: 1fr;
    }

    SliceSelector #slice-range-info {
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ranges: dict[str, tuple[float, float]] = {}
        self._visible = False

    def compose(self) -> ComposeResult:
        yield Label("Slice (required for 3D)", classes="section-title")
        with Vertical():
            with Horizontal(classes="field-row"):
                yield Label("axis", classes="field-label")
                yield Select(
                    [("x1", "x1"), ("x2", "x2"), ("x3", "x3")],
                    value="x3",
                    id="slice-axis",
                    classes="field-select",
                )
            yield Static("", id="slice-range-info")
            with Horizontal(classes="field-row"):
                yield Label("position", classes="field-label")
                yield Input(
                    value="0.0",
                    placeholder="slice position",
                    id="slice-position",
                    classes="field-input",
                )

    def set_ranges(self, ranges: dict[str, tuple[float, float]]) -> None:
        """update coordinate ranges from checkpoint data."""
        self._ranges = ranges
        self._update_range_info()

    def _update_range_info(self) -> None:
        axis_select = self.query_one("#slice-axis", Select)
        axis = axis_select.value
        if axis in self._ranges:
            lo, hi = self._ranges[axis]
            info = self.query_one("#slice-range-info", Static)
            info.update(f"range: [{lo:.4g}, {hi:.4g}]")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "slice-axis":
            self._update_range_info()

    def get_slice_config(self) -> Optional[dict[str, float]]:
        """return slice config dict, e.g. {'x3': 0.5}."""
        axis_select = self.query_one("#slice-axis", Select)
        pos_input = self.query_one("#slice-position", Input)
        axis = axis_select.value
        try:
            pos = float(pos_input.value.strip())
        except (ValueError, AttributeError):
            return None
        return {str(axis): pos}
