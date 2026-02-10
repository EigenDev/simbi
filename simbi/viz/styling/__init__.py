# =============================================================================
# styling/__init__.py
#
# theme lookup for visualization.
# =============================================================================
from dataclasses import asdict
from typing import Any, Optional

from .theme import THEMES, ThemeConfig


def get_theme(
    name: str = "default", overrides: Optional[dict[str, Any]] = None
) -> ThemeConfig:
    """get a named theme, optionally with field overrides."""
    base = THEMES.get(name, THEMES["default"])
    if not overrides:
        return base
    theme_dict = asdict(base)
    theme_dict.update({k: v for k, v in overrides.items() if k in theme_dict})
    return ThemeConfig(**theme_dict)
