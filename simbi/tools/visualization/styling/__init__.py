from typing import Optional

from simbi.tools.visualization.styling.theme import ThemeConfig

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
    def get_theme(cls, theme_name=None) -> ThemeConfig:
        """Get a theme by name or the current theme"""
        if theme_name is None:
            theme_name = cls._current_theme

        if theme_name in cls._themes:
            return cls._themes[theme_name]
        else:
            return cls._themes["default"]

    @classmethod
    def set_theme(
        cls,
        theme_name: str,
        nfiles: int = 1,
        nfields: int = 1,
        user_fig_size: Optional[tuple[float, float]] = None,
    ) -> bool:
        """Set the current theme"""
        if theme_name in cls._themes:
            cls._current_theme = theme_name
            cls._themes[theme_name].apply(nfiles, nfields, user_fig_size)
            return True
        return False

    @classmethod
    def register_theme(cls, name, theme):
        """Register a new theme"""
        cls._themes[name] = theme

    @classmethod
    def apply_current_theme(cls):
        """Apply the current theme"""
        cls._themes[cls._current_theme].apply()
