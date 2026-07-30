# =============================================================================
# config_loader.py
#
# loads component props from yaml/json config files with cli overrides.
# supports dot-notation for quick property changes without editing files.
#
# usage:
#   # load from file
#   props = load_component_props("viz_config.yaml")
#
#   # load with cli overrides
#   props = load_component_props(
#       "viz_config.yaml",
#       overrides=["polygon.show_level_bounds=true", "polygon.level_color=blue"]
#   )
#
#   # cli-only (no file)
#   props = load_component_props(
#       overrides=["quad.log_scale=true", "quad.cmap=inferno"]
#   )
#
# config file format (yaml):
#   polygon:
#     show_level_bounds: true
#     level_color: "red"
#     level_linewidth: 2.0
#
#   quad:
#     cmap: "viridis"
#     log_scale: false
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from pydantic import ValidationError

from .components.interface import ComponentProps
from .props_registry import PROPS_REGISTRY, get_props_class

# type alias for raw config dict
ConfigDict = dict[str, dict[str, Any]]


def _coerce_value(value: str) -> Union[bool, int, float, str]:
    """
    Coerce a string value to its appropriate python type.
    Handles booleans, integers, floats, and strings.
    """
    lower = value.lower()

    # booleans
    if lower in ("true", "yes", "on", "1"):
        return True
    if lower in ("false", "no", "off", "0"):
        return False

    # none/null
    if lower in ("none", "null", ""):
        return None  # type: ignore

    # try numeric types
    try:
        if "." in value or "e" in lower:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # strip quotes if present
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    return value


def _parse_override(override: str) -> tuple[str, str, Any]:
    """
    Parse a dot-notation override string into (component, field, value).

    Examples:
        "polygon.show_level_bounds=true" -> ("polygon", "show_level_bounds", True)
        "quad.color_range.min=0.1" -> ("quad", "color_range.min", 0.1)
    """
    if "=" not in override:
        raise ValueError(
            f"invalid override format: '{override}'. expected 'component.field=value'"
        )

    key, _, raw_value = override.partition("=")
    parts = key.split(".", 1)

    if len(parts) < 2:
        raise ValueError(
            f"invalid override key: '{key}'. expected 'component.field' format"
        )

    component = parts[0].lower().replace("-", "_")
    field = parts[1]
    value = _coerce_value(raw_value)

    return component, field, value


def _set_nested(d: dict, key: str, value: Any) -> None:
    """
    Set a potentially nested key in a dict.

    Examples:
        _set_nested(d, "foo", 1) -> d["foo"] = 1
        _set_nested(d, "foo.bar", 1) -> d["foo"]["bar"] = 1
    """
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def parse_overrides(overrides: Sequence[str]) -> ConfigDict:
    """
    Parse a sequence of override strings into a config dict.

    Args:
        overrides: list of "component.field=value" strings

    Returns:
        dict mapping component names to field dicts
    """
    config: ConfigDict = {}

    for override in overrides:
        component, field, value = _parse_override(override)
        if component not in config:
            config[component] = {}
        _set_nested(config[component], field, value)

    return config


def load_config_file(path: Union[str, Path]) -> ConfigDict:
    """
    Load a config file (yaml or json) and return raw dict.

    Args:
        path: path to config file (.yaml, .yml, or .json)

    Returns:
        dict mapping component names to their config dicts

    Raises:
        FileNotFoundError: if file doesn't exist
        ValueError: if file format is unsupported or invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    suffix = path.suffix.lower()
    content = path.read_text()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(content)
        except ImportError:
            raise ImportError(
                "pyyaml is required for yaml config files. "
                "install with: pip install pyyaml"
            )
        except yaml.YAMLError as e:
            raise ValueError(f"invalid yaml in {path}: {e}")
    elif suffix == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid json in {path}: {e}")
    else:
        raise ValueError(
            f"unsupported config file format: {suffix}. use .yaml, .yml, or .json"
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"config file must contain a dict, got {type(data).__name__}"
        )

    # normalize keys
    return {k.lower().replace("-", "_"): v for k, v in data.items()}


def merge_configs(base: ConfigDict, overrides: ConfigDict) -> ConfigDict:
    """
    Deep merge two config dicts, with overrides taking precedence.
    """
    result = {}

    # copy base
    for key, value in base.items():
        if isinstance(value, dict):
            result[key] = dict(value)
        else:
            result[key] = value

    # apply overrides
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            # merge nested dicts
            result[key].update(value)
        else:
            result[key] = value

    return result


def validate_props(component: str, config: dict[str, Any]) -> ComponentProps:
    """
    Validate and instantiate a props object from config dict.

    Args:
        component: component name (e.g., "polygon", "quad")
        config: dict of field values

    Returns:
        validated props instance

    Raises:
        KeyError: if component is unknown
        ValidationError: if config values are invalid
    """
    props_cls = get_props_class(component)

    # name unknown keys up front with a close-match suggestion; pydantic's
    # extra=forbid rejects them without one.
    known = set(props_cls.model_fields)
    unknown = [k for k in config if k not in known]
    if unknown:
        import difflib

        parts = []
        for k in unknown:
            close = difflib.get_close_matches(k, known, n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            parts.append(f"'{k}'{hint}")
        raise ValueError(
            f"unknown {component} prop(s): {', '.join(parts)};"
            f" known props: {sorted(known)}"
        )

    try:
        return props_cls(**config)
    except ValidationError as e:
        # re-raise with component context
        raise ValidationError.from_exception_data(
            title=f"{component} props validation failed",
            line_errors=e.errors(),
        ) from e


def load_component_props(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Sequence[str]] = None,
) -> dict[str, ComponentProps]:
    """
    Load and validate component props from config file and/or cli overrides.

    Args:
        config_path: optional path to yaml/json config file
        overrides: optional list of "component.field=value" cli overrides

    Returns:
        dict mapping component names to validated props instances

    Raises:
        FileNotFoundError: if config file doesn't exist
        ValueError: if config format is invalid
        ValidationError: if props validation fails

    Examples:
        # file only
        props = load_component_props("viz.yaml")

        # overrides only
        props = load_component_props(overrides=["polygon.cmap=inferno"])

        # file + overrides
        props = load_component_props("viz.yaml", ["polygon.cmap=inferno"])
    """
    file_config: ConfigDict = {}
    cli_config: ConfigDict = {}

    # load file config
    if config_path:
        file_config = load_config_file(config_path)

    # parse cli overrides
    if overrides:
        cli_config = parse_overrides(overrides)

    # merge configs (cli wins)
    merged = merge_configs(file_config, cli_config)

    # validate and instantiate
    props: dict[str, ComponentProps] = {}
    errors: list[str] = []

    for component, config in merged.items():
        if component not in PROPS_REGISTRY:
            # accepting an unknown component name would silently drop EVERY override
            # under it (qaud.cmap=inferno), so it raises with a close-match suggestion.
            import difflib

            close = difflib.get_close_matches(component, PROPS_REGISTRY, n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            raise ValueError(
                f"unknown props component '{component}'{hint}; valid components: "
                + ", ".join(sorted(PROPS_REGISTRY))
            )

        try:
            props[component] = validate_props(component, config)
        except (ValidationError, TypeError, ValueError) as e:
            errors.append(f"{component}: {e}")

    if errors:
        raise ValueError(
            "config validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return props


def load_theme_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """
    Load and return validated theme props (if present) using load_component_props.

    This convenience wrapper focuses on the `theme` key and returns the validated
    ThemeProps instance (pydantic model) when found, otherwise returns an empty dict.

    It delegates parsing, coercion and validation to load_component_props so CLI
    overrides and config files are handled consistently.
    """
    # nothing to do if no source provided
    if not config_path and not overrides:
        return {}

    # reuse existing loader which merges file + overrides and validates props
    loaded = load_component_props(config_path, overrides)
    return loaded.get("theme", {})


def generate_example_config() -> str:
    """
    Generate an example config file showing all available options.

    Returns:
        yaml-formatted string with all components and their defaults
    """
    lines = [
        "# simbi visualization config",
        "# auto-generated example showing all available options",
        "",
    ]

    for name, props_cls in sorted(PROPS_REGISTRY.items()):
        lines.append(f"{name}:")

        # get field info from pydantic model
        for field_name, field_info in props_cls.model_fields.items():
            default = field_info.default
            annotation = field_info.annotation

            # format default value
            if default is None:
                default_str = "null"
            elif isinstance(default, bool):
                default_str = str(default).lower()
            elif isinstance(default, str):
                default_str = f'"{default}"'
            else:
                default_str = str(default)

            # add comment with type hint
            type_hint = getattr(annotation, "__name__", str(annotation))
            lines.append(f"  # {field_name}: {type_hint}")
            lines.append(f"  {field_name}: {default_str}")

        lines.append("")

    return "\n".join(lines)
