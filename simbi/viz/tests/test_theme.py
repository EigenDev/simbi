# =============================================================================
# test_theme.py
#
# unit tests for theme system.
# =============================================================================
import pytest

from simbi.viz.styling import get_theme
from simbi.viz.styling.theme import THEMES, ThemeConfig


class TestThemeConfig:
    """test ThemeConfig dataclass."""

    def test_default_values(self):
        config = ThemeConfig()

        assert config.font_family == "serif"
        assert config.font_size == 12
        assert config.color_map == "viridis"
        assert config.use_tex is False

    def test_from_name_default(self):
        config = ThemeConfig.from_name("default")

        assert config.font_family == "serif"
        assert config.color_map == "viridis"

    def test_from_name_dark(self):
        config = ThemeConfig.from_name("dark")

        assert config.font_family == "sans-serif"
        assert config.color_map == "plasma"
        assert config.text_color == "white"

    def test_from_name_scientific(self):
        config = ThemeConfig.from_name("scientific")

        assert config.font_family == "Times New Roman"
        assert config.use_tex is True
        assert config.line_width == 1.2

    def test_from_name_invalid_raises(self):
        with pytest.raises(ValueError, match="unknown theme"):
            ThemeConfig.from_name("nonexistent")

    def test_from_mapping(self):
        data = {
            "font_family": "monospace",
            "font_size": 14,
            "color_map": "inferno",
            "unknown_field": "ignored",
        }
        config = ThemeConfig.from_mapping(data)

        assert config.font_family == "monospace"
        assert config.font_size == 14
        assert config.color_map == "inferno"

    def test_from_mapping_invalid_type(self):
        with pytest.raises(TypeError, match="expected dict"):
            ThemeConfig.from_mapping("not a dict")


class TestThemeRegistry:
    """test theme registry."""

    def test_contains_default(self):
        assert "default" in THEMES

    def test_contains_dark(self):
        assert "dark" in THEMES

    def test_contains_scientific(self):
        assert "scientific" in THEMES

    def test_all_are_theme_config(self):
        for name, theme in THEMES.items():
            assert isinstance(theme, ThemeConfig), f"{name} is not ThemeConfig"


class TestGetTheme:
    """test get_theme function."""

    def test_get_default(self):
        theme = get_theme("default")
        assert theme.font_family == "serif"

    def test_get_with_overrides(self):
        theme = get_theme("default", {"font_size": 16, "color_map": "inferno"})

        assert theme.font_size == 16
        assert theme.color_map == "inferno"
        # other defaults preserved
        assert theme.font_family == "serif"

    def test_get_unknown_falls_back_to_default(self):
        theme = get_theme("nonexistent")
        assert theme.font_family == "serif"  # default theme


class TestThemeApply:
    """test theme application to matplotlib."""

    def test_apply_does_not_raise(self):
        config = ThemeConfig()
        # should not raise
        config.apply(nfiles=2, nfields=3)

    def test_apply_overlay_mode(self):
        config = ThemeConfig()
        # should not raise
        config.apply(nfiles=2, nfields=3, overlay_mode=True)
