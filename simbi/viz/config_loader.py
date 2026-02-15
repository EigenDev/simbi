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
from .registry import get_props_class, get_props_registry

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


def _split_file_prefix(token: str) -> tuple[int | None, str]:
    """
    detect an optional N: file-index prefix.

    returns (file_index, remainder) where file_index is None for global
    overrides. component names never start with a digit so the split is
    unambiguous.
    """
    colon = token.find(":")
    if colon < 1:
        return None, token
    prefix = token[:colon]
    if prefix.isdigit():
        return int(prefix), token[colon + 1 :]
    return None, token


def parse_overrides(
    overrides: Sequence[str],
) -> tuple[ConfigDict, dict[int, ConfigDict]]:
    """
    parse a sequence of override strings into global and per-file config dicts.

    supports two forms:
        "component.field=value"        — global (applies to all files)
        "N:component.field=value"      — per-file (applies to file N only)

    returns:
        (global_config, per_file_config)
    """
    global_config: ConfigDict = {}
    per_file: dict[int, ConfigDict] = {}

    for override in overrides:
        file_idx, remainder = _split_file_prefix(override)
        component, field, value = _parse_override(remainder)

        if file_idx is None:
            target = global_config
        else:
            target = per_file.setdefault(file_idx, {})

        if component not in target:
            target[component] = {}
        _set_nested(target[component], field, value)

    return global_config, per_file


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
) -> tuple[dict[str, ComponentProps], dict[int, ConfigDict]]:
    """
    load and validate component props from config file and/or cli overrides.

    returns:
        (validated_global_props, per_file_raw_dicts)

        per-file dicts are left raw — validated at render time via
        resolve_per_file_props, matching the grid panel_overrides pattern.
    """
    file_config: ConfigDict = {}
    global_cli: ConfigDict = {}
    per_file: dict[int, ConfigDict] = {}

    # load file config
    if config_path:
        file_config = load_config_file(config_path)

    # parse cli overrides
    if overrides:
        global_cli, per_file = parse_overrides(overrides)

    # merge configs (cli wins)
    merged = merge_configs(file_config, global_cli)

    # extract grid section (not a component — handled separately)
    merged.pop("grid", None)

    # validate and instantiate global props
    props: dict[str, ComponentProps] = {}
    errors: list[str] = []

    for component, config in merged.items():
        if component not in get_props_registry():
            continue

        try:
            props[component] = validate_props(component, config)
        except (ValidationError, TypeError) as e:
            errors.append(f"{component}: {e}")

    if errors:
        raise ValueError(
            "config validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return props, per_file


def load_grid_config(
    config_path: Optional[Union[str, Path]] = None,
) -> Optional[dict[str, Any]]:
    """
    extract the grid section from a config file.

    returns the raw grid dict (with keys like shared_colorbar, auto_label,
    panels) or None if no grid section exists.
    """
    if not config_path:
        return None

    file_config = load_config_file(config_path)
    return file_config.get("grid", None)


def get_props_for_component(
    component_name: str,
    loaded_props: dict[str, ComponentProps],
    defaults: Optional[dict[str, Any]] = None,
) -> ComponentProps:
    """
    Get props for a specific component, with fallback to defaults.

    Args:
        component_name: name of the component (e.g., "polygon")
        loaded_props: dict from load_component_props()
        defaults: optional default values to use if not in loaded_props

    Returns:
        props instance for the component
    """
    key = component_name.lower().replace("-", "_")

    if key in loaded_props:
        return loaded_props[key]

    # fall back to defaults or empty props
    props_cls = get_props_class(key)
    return props_cls(**(defaults or {}))


def resolve_per_file_props(
    base_props: Optional[dict[str, ComponentProps]],
    per_file_overrides: Optional[dict[int, dict]],
    file_idx: int,
) -> dict[str, ComponentProps]:
    """merge base props with per-file overrides for a single file index."""
    result = dict(base_props) if base_props else {}

    if not per_file_overrides or file_idx not in per_file_overrides:
        return result

    overrides = per_file_overrides[file_idx]
    for comp_name, comp_overrides in overrides.items():
        if comp_name == "label":
            continue
        if comp_name in result:
            existing = result[comp_name]
            merged = {**existing.model_dump(), **comp_overrides}
            result[comp_name] = type(existing)(**merged)
        else:
            try:
                props_cls = get_props_class(comp_name)
                result[comp_name] = props_cls(**comp_overrides)
            except KeyError:
                pass

    return result


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

    for name, props_cls in sorted(get_props_registry().items()):
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
