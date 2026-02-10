# =============================================================================
# test_config.py
#
# unit tests for config parsing and validation.
# =============================================================================
import pytest

from simbi.viz.config import OverlayConfig, PlotConfig, VisualizationConfig
from simbi.viz.pipeline.conversion import overlays_from_args, parse_overlay_spec


class TestOverlayConfig:
    """test OverlayConfig validation."""

    def test_defaults(self):
        config = OverlayConfig(field="mach")

        assert config.field == "mach"
        assert config.component == "contour"
        assert config.levels == [1.0]
        assert config.color == "white"
        assert config.linewidth == 1.5

    def test_custom_levels(self):
        config = OverlayConfig(field="mach", levels=[0.5, 1.0, 2.0])

        assert config.levels == [0.5, 1.0, 2.0]

    def test_alpha_validation(self):
        with pytest.raises(ValueError, match="alpha"):
            OverlayConfig(field="mach", alpha=1.5)

        with pytest.raises(ValueError, match="alpha"):
            OverlayConfig(field="mach", alpha=-0.1)


class TestParseOverlaySpec:
    """test overlay specification parsing."""

    def test_basic_spec(self):
        config = parse_overlay_spec("mach:contour:1.0")

        assert config.field == "mach"
        assert config.component == "contour"
        assert config.levels == [1.0]

    def test_multiple_levels(self):
        config = parse_overlay_spec("mach:contour:0.5,1.0,1.5")

        assert config.levels == [0.5, 1.0, 1.5]

    def test_field_only(self):
        config = parse_overlay_spec("mach")

        assert config.field == "mach"
        assert config.component == "contour"
        assert config.levels == [1.0]

    def test_field_and_component(self):
        config = parse_overlay_spec("v:contour")

        assert config.field == "v"
        assert config.component == "contour"
        assert config.levels == [1.0]

    def test_default_color_and_linewidth(self):
        config = parse_overlay_spec(
            "mach:contour:1.0", default_color="red", default_linewidth=2.0
        )

        assert config.color == "red"
        assert config.linewidth == 2.0

    def test_invalid_levels_raises(self):
        with pytest.raises(ValueError, match="invalid levels"):
            parse_overlay_spec("mach:contour:not_a_number")


class TestOverlaysFromArgs:
    """test parsing overlays from argparse namespace."""

    def test_no_overlays(self):
        class MockArgs:
            field_overlays = None

        result = overlays_from_args(MockArgs())
        assert result == []

    def test_single_overlay(self):
        class MockArgs:
            field_overlays = [["mach:contour:1.0"]]
            overlay_color = "white"
            overlay_linewidth = 1.5

        result = overlays_from_args(MockArgs())

        assert len(result) == 1
        assert result[0].field == "mach"

    def test_multiple_overlays(self):
        class MockArgs:
            field_overlays = [["mach:contour:1.0"], ["v:contour:0.5,1.0"]]
            overlay_color = "white"
            overlay_linewidth = 1.5

        result = overlays_from_args(MockArgs())

        assert len(result) == 2
        assert result[0].field == "mach"
        assert result[1].field == "v"
        assert result[1].levels == [0.5, 1.0]

    def test_multiple_specs_same_flag(self):
        class MockArgs:
            field_overlays = [["mach:contour:1.0", "v:contour:0.5"]]
            overlay_color = "red"
            overlay_linewidth = 2.0

        result = overlays_from_args(MockArgs())

        assert len(result) == 2
        assert result[0].color == "red"
        assert result[0].linewidth == 2.0


class TestVisualizationConfigWithOverlays:
    """test VisualizationConfig with overlays."""

    def test_empty_overlays_default(self):
        config = VisualizationConfig(
            plot=PlotConfig(plot_type="multidim", fields=["rho"])
        )

        assert config.overlays == []

    def test_with_overlays(self):
        overlays = [
            OverlayConfig(field="mach", levels=[1.0]),
            OverlayConfig(field="v", levels=[0.5]),
        ]
        config = VisualizationConfig(
            plot=PlotConfig(plot_type="multidim", fields=["rho"]),
            overlays=overlays,
        )

        assert len(config.overlays) == 2
        assert config.overlays[0].field == "mach"
