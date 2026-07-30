from dataclasses import asdict
from typing import Any, Optional

from .theme import ThemeConfig
from .themes.dark import dark_theme
from .themes.default import default_theme
from .themes.scientific import scientific_theme


class ThemeManager:
    """Manages themes for visualization components"""

    _themes = {
        "default": default_theme,
        "scientific": scientific_theme,
        "dark": dark_theme,
    }

    _current_theme = "default"

    @classmethod
    def get_theme(
        cls, theme_name=None, theme_props: Optional[dict[str, Any]] = {}
    ) -> ThemeConfig:
        """Get a theme by name or the current theme"""
        if theme_name is None:
            theme_name = cls._current_theme

        if theme_name in cls._themes:
            theme = cls._themes[theme_name]
            theme_dict = asdict(theme)
            shared_keys = set(theme_dict).intersection(set(theme_props))
            theme_dict.update({k: theme_props[k] for k in shared_keys})
            return ThemeConfig(**theme_dict)
        else:
            return cls._themes["default"]
