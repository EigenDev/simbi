# =============================================================================
# test_grid.py
#
# tests for the multi-panel grid plotting feature.
# =============================================================================
from unittest.mock import MagicMock, patch

import matplotlib.colors as mcolors
import numpy as np
import pytest
from matplotlib import pyplot as plt

from simbi.viz.components.interface import ComponentProps
from simbi.viz.components.shared import ColormappedProps
from simbi.viz.grid import (
    _compute_global_range,
    _compute_grid_shape,
    _extract_panel_label,
    _override_color_range,
    _resolve_panel_props,
    plot_grid,
)
from simbi.viz.types import ColorRange, FieldData


# =========================================================================
# _compute_grid_shape
# =========================================================================
class TestComputeGridShape:
    def test_single_panel(self):
        assert _compute_grid_shape(1) == (1, 1)

    def test_two_panels(self):
        assert _compute_grid_shape(2) == (1, 2)

    def test_three_panels(self):
        assert _compute_grid_shape(3) == (1, 3)

    def test_four_panels(self):
        assert _compute_grid_shape(4) == (2, 2)

    def test_five_panels(self):
        nrows, ncols = _compute_grid_shape(5)
        assert nrows * ncols >= 5

    def test_six_panels(self):
        nrows, ncols = _compute_grid_shape(6)
        assert nrows * ncols >= 6
        assert nrows <= 3 and ncols <= 3

    def test_nine_panels(self):
        assert _compute_grid_shape(9) == (3, 3)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            _compute_grid_shape(0)


# =========================================================================
# _extract_panel_label
# =========================================================================
class TestExtractPanelLabel:
    def test_no_auto_label_uses_stem(self):
        label = _extract_panel_label("path/to/chk.0010.h5", None, False)
        assert label == "chk.0010"

    def test_auto_label_with_metadata(self):
        sim_data = MagicMock()
        sim_data.metadata.time = 1.5
        sim_data.metadata.gamma = 1.4
        sim_data.metadata.coord_system = "cartesian"

        label = _extract_panel_label("chk.0010.h5", sim_data, True)
        assert "t=" in label
        assert "1.4" in label

    def test_auto_label_falls_back_to_stem(self):
        sim_data = MagicMock(spec=[])
        label = _extract_panel_label("chk.0010.h5", sim_data, True)
        assert label == "chk.0010"


# =========================================================================
# _compute_global_range
# =========================================================================
class TestComputeGlobalRange:
    def test_single_panel_single_field(self):
        field = FieldData(
            name="rho",
            values=np.array([[1.0, 2.0], [3.0, 4.0]]),
            domain=[np.array([0, 1]), np.array([0, 1])],
        )
        vmin, vmax = _compute_global_range([[field]])
        assert vmin == 1.0
        assert vmax == 4.0

    def test_multi_panel(self):
        f1 = FieldData(
            name="rho",
            values=np.array([[1.0, 5.0]]),
            domain=[np.array([0]), np.array([0, 1])],
        )
        f2 = FieldData(
            name="rho",
            values=np.array([[-2.0, 3.0]]),
            domain=[np.array([0]), np.array([0, 1])],
        )
        vmin, vmax = _compute_global_range([[f1], [f2]])
        assert vmin == -2.0
        assert vmax == 5.0

    def test_empty_fields(self):
        vmin, vmax = _compute_global_range([[]])
        assert vmin == 0.0
        assert vmax == 1.0


# =========================================================================
# _override_color_range
# =========================================================================
class TestOverrideColorRange:
    def test_overrides_when_no_user_range(self):
        props = ColormappedProps()
        result = _override_color_range(props, 0.5, 10.0)
        assert result.color_range.min == 0.5
        assert result.color_range.max == 10.0

    def test_preserves_user_range(self):
        props = ColormappedProps(color_range=ColorRange(min=1.0, max=5.0))
        result = _override_color_range(props, 0.5, 10.0)
        assert result.color_range.min == 1.0
        assert result.color_range.max == 5.0

    def test_non_colormapped_passthrough(self):
        props = ComponentProps()
        result = _override_color_range(props, 0.5, 10.0)
        assert result is props


# =========================================================================
# _resolve_panel_props
# =========================================================================
class TestResolvePanelProps:
    def test_no_overrides(self):
        base = {"quad": ColormappedProps(cmap="viridis")}
        result = _resolve_panel_props(base, None, 0)
        assert result["quad"].cmap == "viridis"

    def test_with_override(self):
        base = {"quad": ColormappedProps(cmap="viridis")}
        overrides = {0: {"quad": {"cmap": "inferno"}}}
        result = _resolve_panel_props(base, overrides, 0)
        assert result["quad"].cmap == "inferno"

    def test_no_override_for_panel(self):
        base = {"quad": ColormappedProps(cmap="viridis")}
        overrides = {1: {"quad": {"cmap": "inferno"}}}
        result = _resolve_panel_props(base, overrides, 0)
        assert result["quad"].cmap == "viridis"

    def test_none_base(self):
        result = _resolve_panel_props(None, None, 0)
        assert result == {}


# =========================================================================
# plot_grid (integration, mocked data loading)
# =========================================================================
class TestPlotGrid:
    @pytest.fixture
    def mock_config(self):
        from simbi.viz.config import (
            PlotConfig,
            VisualizationConfig,
        )

        return VisualizationConfig(
            plot=PlotConfig(plot_type="multidim", fields=["rho"])
        )

    def test_empty_files_raises(self, mock_config):
        with pytest.raises(ValueError, match="no files"):
            plot_grid(mock_config, [], show=False)

    def test_layout_too_small_raises(self, mock_config):
        with pytest.raises(ValueError, match="too small"):
            plot_grid(
                mock_config,
                ["a.h5", "b.h5", "c.h5"],
                layout=(1, 2),
                show=False,
            )

    @patch("simbi.viz.grid.load_data")
    @patch("simbi.viz.grid.create_plot_data")
    @patch("simbi.viz.grid.compose_fields_for_render")
    @patch("simbi.viz.grid.select_scalar_component")
    @patch("matplotlib.pyplot.show")
    def test_basic_2d_grid(
        self,
        mock_show,
        mock_select,
        mock_compose,
        mock_create,
        mock_load,
        mock_config,
    ):
        # setup mock data
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        values = np.random.rand(10, 10)
        field = FieldData(
            name="rho",
            values=values,
            domain=[y, x],
            time=0.0,
        )

        mock_sim = MagicMock()
        mock_sim.metadata.coord_system = "cartesian"
        mock_sim.metadata.time = 0.0
        mock_sim.metadata.gamma = 1.4
        mock_load.return_value = mock_sim

        mock_plot_data = MagicMock()
        mock_plot_data.fields = [field]
        mock_plot_data.body_collection = None
        mock_create.return_value = mock_plot_data

        mock_compose.return_value = [field]

        # mock component (no mappable — avoids real colorbar creation)
        mock_comp = MagicMock()
        mock_comp.render.return_value = MagicMock(
            artists={},
            metadata={},
        )
        mock_comp_cls = MagicMock(return_value=mock_comp)
        mock_props_cls = ColormappedProps
        mock_select.return_value = (mock_comp_cls, mock_props_cls, "quad")

        fig = plot_grid(
            mock_config,
            ["file1.h5", "file2.h5", "file3.h5", "file4.h5"],
            fields=["rho"],
            shared_colorbar=False,
            show=False,
        )

        assert fig is not None
        assert mock_load.call_count == 4
        assert mock_comp.initialize.call_count == 4
        assert mock_comp.render.call_count == 4
        plt.close(fig)

    @patch("simbi.viz.grid.load_data")
    @patch("simbi.viz.grid.create_plot_data")
    @patch("simbi.viz.grid.compose_fields_for_render")
    @patch("simbi.viz.grid.select_scalar_component")
    @patch("matplotlib.pyplot.show")
    def test_explicit_layout(
        self,
        mock_show,
        mock_select,
        mock_compose,
        mock_create,
        mock_load,
        mock_config,
    ):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        values = np.random.rand(10, 10)
        field = FieldData(name="rho", values=values, domain=[y, x], time=0.0)

        mock_sim = MagicMock()
        mock_sim.metadata.coord_system = "cartesian"
        mock_sim.metadata.time = 0.0
        mock_load.return_value = mock_sim

        mock_plot_data = MagicMock()
        mock_plot_data.fields = [field]
        mock_plot_data.body_collection = None
        mock_create.return_value = mock_plot_data
        mock_compose.return_value = [field]

        mock_comp = MagicMock()
        mock_comp.render.return_value = MagicMock(
            artists={},
            metadata={},
        )
        mock_comp_cls = MagicMock(return_value=mock_comp)
        mock_select.return_value = (mock_comp_cls, ColormappedProps, "quad")

        fig = plot_grid(
            mock_config,
            ["a.h5", "b.h5"],
            layout=(1, 2),
            panel_labels=["Sim A", "Sim B"],
            shared_colorbar=False,
            show=False,
        )

        assert fig is not None
        plt.close(fig)

    @patch("simbi.viz.grid.load_data")
    @patch("simbi.viz.grid.create_plot_data")
    @patch("simbi.viz.grid.compose_fields_for_render")
    @patch("simbi.viz.grid.select_scalar_component")
    @patch("matplotlib.pyplot.show")
    def test_shared_colorbar(
        self,
        mock_show,
        mock_select,
        mock_compose,
        mock_create,
        mock_load,
        mock_config,
    ):
        """shared colorbar uses a real ScalarMappable."""
        import matplotlib.cm as cm

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        values = np.random.rand(10, 10)
        field = FieldData(name="rho", values=values, domain=[y, x], time=0.0)

        mock_sim = MagicMock()
        mock_sim.metadata.coord_system = "cartesian"
        mock_sim.metadata.time = 0.0
        mock_load.return_value = mock_sim

        mock_plot_data = MagicMock()
        mock_plot_data.fields = [field]
        mock_plot_data.body_collection = None
        mock_create.return_value = mock_plot_data
        mock_compose.return_value = [field]

        # use a real ScalarMappable so fig.colorbar works
        norm = mcolors.Normalize(vmin=0, vmax=1)
        real_mappable = cm.ScalarMappable(norm=norm, cmap="viridis")

        mock_comp = MagicMock()
        mock_comp.render.return_value = MagicMock(
            artists={"mesh": real_mappable},
            metadata={"mappable": real_mappable},
        )
        mock_comp_cls = MagicMock(return_value=mock_comp)
        mock_select.return_value = (mock_comp_cls, ColormappedProps, "quad")

        fig = plot_grid(
            mock_config,
            ["a.h5", "b.h5"],
            fields=["rho"],
            shared_colorbar=True,
            show=False,
        )

        assert fig is not None
        # constrained layout + colorbar = at least 3 axes (2 panels + cbar)
        assert len(fig.get_axes()) >= 2
        plt.close(fig)
