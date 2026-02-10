# =============================================================================
# simbi/viz/tui/widgets/config_panel.py
#
# props-driven parameter form for plot configuration.
# introspects pydantic ComponentProps models to auto-generate widgets.
# bool -> Switch, float/int -> Input, Literal[...] -> Select, etc.
# adding a new prop to any component automatically shows up in the tui.
# =============================================================================
from typing import Any, Literal, Optional, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Label, Select, Static, Switch


def _is_literal(annotation) -> bool:
    return get_origin(annotation) is Literal


def _is_optional(annotation) -> bool:
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        return type(None) in args
    return False


def _unwrap_optional(annotation):
    """strip Optional wrapper to get inner type."""
    args = get_args(annotation)
    return next((a for a in args if a is not type(None)), annotation)


def _is_bool(annotation) -> bool:
    if annotation is bool:
        return True
    if _is_optional(annotation):
        return _unwrap_optional(annotation) is bool
    return False


def _is_numeric(annotation) -> bool:
    base = (
        _unwrap_optional(annotation) if _is_optional(annotation) else annotation
    )
    return base in (int, float)


class ConfigPanel(Static):
    """auto-generated config form from a pydantic model."""

    DEFAULT_CSS = """
    ConfigPanel {
        height: auto;
        width: 100%;
        padding: 0 1;
    }

    ConfigPanel .field-row {
        height: auto;
        margin-bottom: 1;
    }

    ConfigPanel .field-label {
        width: 22;
        padding-right: 1;
    }

    ConfigPanel .field-input {
        width: 1fr;
    }

    ConfigPanel .field-switch {
        width: auto;
    }

    ConfigPanel .field-select {
        width: 1fr;
    }
    """

    def __init__(
        self,
        title: str,
        model_class: type[BaseModel],
        initial_values: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._model_class = model_class
        self._initial = initial_values or {}
        self._widget_ids: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Label(self._title, classes="section-title")
        with Vertical():
            for name, field_info in self._model_class.model_fields.items():
                yield from self._make_field_widget(name, field_info)

    def _make_field_widget(
        self, name: str, field_info: FieldInfo
    ) -> ComposeResult:
        annotation = field_info.annotation
        default = self._initial.get(name, field_info.default)
        widget_id = f"cfg-{self._title.lower().replace(' ', '-')}-{name}"
        self._widget_ids[name] = widget_id

        if _is_literal(annotation):
            choices = get_args(annotation)
            with Horizontal(classes="field-row"):
                yield Label(name, classes="field-label")
                yield Select(
                    [(str(c), c) for c in choices],
                    value=default if default is not None else choices[0],
                    id=widget_id,
                    classes="field-select",
                )

        elif _is_bool(annotation):
            with Horizontal(classes="field-row"):
                yield Label(name, classes="field-label")
                yield Switch(
                    value=bool(default) if default is not None else False,
                    id=widget_id,
                    classes="field-switch",
                )

        elif _is_numeric(annotation):
            with Horizontal(classes="field-row"):
                yield Label(name, classes="field-label")
                yield Input(
                    value=str(default) if default is not None else "",
                    placeholder=name,
                    id=widget_id,
                    classes="field-input",
                )

        elif annotation is str or (
            _is_optional(annotation) and _unwrap_optional(annotation) is str
        ):
            with Horizontal(classes="field-row"):
                yield Label(name, classes="field-label")
                yield Input(
                    value=str(default) if default is not None else "",
                    placeholder=name,
                    id=widget_id,
                    classes="field-input",
                )

        else:
            # fallback: render as text input for complex types
            with Horizontal(classes="field-row"):
                yield Label(name, classes="field-label")
                yield Input(
                    value=str(default) if default is not None else "",
                    placeholder=f"{name} ({annotation})",
                    id=widget_id,
                    classes="field-input",
                )

    def collect_values(self) -> dict[str, Any]:
        """collect current widget values as a dict."""
        result = {}
        for name, field_info in self._model_class.model_fields.items():
            widget_id = self._widget_ids.get(name)
            if not widget_id:
                continue
            annotation = field_info.annotation

            try:
                if _is_bool(annotation):
                    widget = self.query_one(f"#{widget_id}", Switch)
                    result[name] = widget.value
                elif _is_literal(annotation):
                    widget = self.query_one(f"#{widget_id}", Select)
                    result[name] = widget.value
                elif _is_numeric(annotation):
                    widget = self.query_one(f"#{widget_id}", Input)
                    val = widget.value.strip()
                    if not val:
                        result[name] = field_info.default
                    else:
                        base = (
                            _unwrap_optional(annotation)
                            if _is_optional(annotation)
                            else annotation
                        )
                        result[name] = base(val)
                else:
                    widget = self.query_one(f"#{widget_id}", Input)
                    val = widget.value.strip()
                    result[name] = val if val else field_info.default
            except Exception:
                result[name] = field_info.default

        return result

    def build_props(self) -> BaseModel:
        """build a props instance from current widget values."""
        values = self.collect_values()
        return self._model_class(**values)
