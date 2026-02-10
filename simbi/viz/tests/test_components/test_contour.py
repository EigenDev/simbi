# =============================================================================
# test_contour.py
#
# unit tests for ContourPlotComponent.
# =============================================================================
import pytest

from simbi.viz.components.contour import ContourPlotComponent, ContourPlotProps
from simbi.viz.config import FigureConfig


class TestContourPlotProps:
    """test ContourPlotProps validation and defaults."""

    def test_default_values(self):
        props = ContourPlotProps()
        assert props.levels == (1.0,)
        assert props.color == "white"
        assert props.linewidths == 1.5
        assert props.alpha == 1.0
        assert props.filled is False

    def test_custom_levels(self):
        props = ContourPlotProps(levels=[0.5, 1.0, 1.5])
        assert props.levels == [0.5, 1.0, 1.5]

    def test_filled_contours(self):
        props = ContourPlotProps(filled=True)
        assert props.filled is True


class TestContourPlotComponent:
    """test ContourPlotComponent rendering."""

    def test_initialize(self, fig_and_ax):
        fig, ax = fig_and_ax
        props = ContourPlotProps()
        component = ContourPlotComponent(props)

        assert not component.initialized
        component.initialize(fig, ax)
        assert component.initialized

    def test_render_single_level(self, fig_and_ax, mock_2d_field):
        fig, ax = fig_and_ax
        props = ContourPlotProps(levels=[0.5])
        component = ContourPlotComponent(props)
        component.initialize(fig, ax)

        result = component.render(mock_2d_field, FigureConfig())

        assert "contour" in result.artists
        assert result.artists["contour"] is not None
        assert result.metadata["is_contour"] is True
        assert result.metadata["is_overlay"] is True

    def test_render_multiple_levels(self, fig_and_ax, mock_mach_field):
        fig, ax = fig_and_ax
        props = ContourPlotProps(levels=[0.5, 1.0, 1.5])
        component = ContourPlotComponent(props)
        component.initialize(fig, ax)

        result = component.render(mock_mach_field, FigureConfig())

        assert result.metadata["levels"] == [0.5, 1.0, 1.5]

    def test_render_filled_contours(self, fig_and_ax, mock_2d_field):
        fig, ax = fig_and_ax
        props = ContourPlotProps(levels=[0.3, 0.6, 0.9], filled=True)
        component = ContourPlotComponent(props)
        component.initialize(fig, ax)

        result = component.render(mock_2d_field, FigureConfig())

        assert "contour" in result.artists

    def test_render_with_edge_coordinates(
        self, fig_and_ax, mock_2d_field_with_edges
    ):
        fig, ax = fig_and_ax
        props = ContourPlotProps(levels=[0.0])
        component = ContourPlotComponent(props)
        component.initialize(fig, ax)

        # should handle edge-to-center conversion
        result = component.render(mock_2d_field_with_edges, FigureConfig())

        assert "contour" in result.artists

    def test_rejects_1d_data(self, fig_and_ax, mock_1d_field):
        fig, ax = fig_and_ax
        props = ContourPlotProps()
        component = ContourPlotComponent(props)
        component.initialize(fig, ax)

        with pytest.raises(ValueError, match="2d"):
            component.render(mock_1d_field, FigureConfig())

    def test_cleanup(self, fig_and_ax, mock_2d_field):
        fig, ax = fig_and_ax
        props = ContourPlotProps(levels=[0.5])
        component = ContourPlotComponent(props)
        component.initialize(fig, ax)

        component.render(mock_2d_field, FigureConfig())
        assert component._contour_set is not None

        component.cleanup()
        assert component._contour_set is None

    def test_animation_cleans_up_previous(self, fig_and_ax, mock_2d_field):
        """test that re-rendering cleans up previous contours."""
        fig, ax = fig_and_ax
        props = ContourPlotProps(levels=[0.5])
        component = ContourPlotComponent(props)
        component.initialize(fig, ax)

        # first render
        component.render(mock_2d_field, FigureConfig())
        first_contour = component._contour_set

        # second render should clean up first
        component.render(mock_2d_field, FigureConfig())
        second_contour = component._contour_set

        # contour set should be different (new instance)
        assert second_contour is not first_contour

    def test_not_initialized_raises(self, mock_2d_field):
        props = ContourPlotProps()
        component = ContourPlotComponent(props)

        with pytest.raises(RuntimeError, match="not initialized"):
            component.render(mock_2d_field, FigureConfig())

    def test_update_props(self, fig_and_ax):
        fig, ax = fig_and_ax
        props = ContourPlotProps(levels=[1.0])
        component = ContourPlotComponent(props)
        component.initialize(fig, ax)

        new_props = ContourPlotProps(levels=[2.0], color="red")
        component.update(new_props)

        assert list(component.props.levels) == [2.0]
        assert component.props.color == "red"
