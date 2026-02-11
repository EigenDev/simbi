# =============================================================================
# test_dispatch.py
#
# unit tests for component dispatch registry.
# =============================================================================
import numpy as np
import pytest

from simbi.viz.components import (
    ContourPlotComponent,
    LinePlotComponent,
    PolygonPlotComponent,
    QuadPlotComponent,
    QuiverPlotComponent,
    StreamPlotComponent,
)
from simbi.viz.registry import (
    list_overlay_types,
    select_overlay_component,
    select_scalar_component,
    select_vector_component,
)
from simbi.viz.types import FieldData


class TestSelectScalarComponent:
    """test scalar component dispatch."""

    def test_1d_field_to_line(self):
        field = FieldData(
            name="rho",
            values=np.array([1, 2, 3]),
            domain=[np.array([0, 1, 2])],
        )
        comp_cls, _, key = select_scalar_component(field)

        assert comp_cls == LinePlotComponent
        assert key == "line"

    def test_2d_field_to_quad(self):
        field = FieldData(
            name="rho",
            values=np.random.rand(10, 10),
            domain=[np.linspace(0, 1, 10), np.linspace(0, 1, 10)],
        )
        comp_cls, _, key = select_scalar_component(field, use_polygons=False)

        assert comp_cls == QuadPlotComponent
        assert key == "quad"

    def test_2d_field_to_polygon_when_forced(self):
        field = FieldData(
            name="rho",
            values=np.random.rand(10, 10),
            domain=[np.linspace(0, 1, 10), np.linspace(0, 1, 10)],
        )
        comp_cls, _, key = select_scalar_component(field, use_polygons=True)

        assert comp_cls == PolygonPlotComponent
        assert key == "polygon"

    def test_polygon_suffix_detected(self):
        # polygon contract: 1d values, domain is array of patches
        field = FieldData(
            name="rho_polygons",
            values=np.array([1, 2, 3]),
            domain=np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]]),
        )
        comp_cls, _, key = select_scalar_component(field)

        assert comp_cls == PolygonPlotComponent
        assert key == "polygon"

    def test_3d_field_raises(self):
        field = FieldData(
            name="rho",
            values=np.random.rand(5, 5, 5),
            domain=[np.linspace(0, 1, 5)] * 3,
        )

        with pytest.raises(ValueError, match="3D"):
            select_scalar_component(field)


class TestSelectVectorComponent:
    """test vector component dispatch."""

    def test_quiver(self):
        comp_cls, _, key = select_vector_component("quiver")

        assert comp_cls == QuiverPlotComponent
        assert key == "quiver"

    def test_stream(self):
        comp_cls, _, key = select_vector_component("stream")

        assert comp_cls == StreamPlotComponent
        assert key == "stream"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="unknown vector_type"):
            select_vector_component("invalid")


class TestSelectOverlayComponent:
    """test overlay component dispatch."""

    def test_contour(self):
        comp_cls, _, key = select_overlay_component("contour")

        assert comp_cls == ContourPlotComponent
        assert key == "contour"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="unknown overlay_type"):
            select_overlay_component("invalid")


class TestListOverlayTypes:
    """test overlay type listing."""

    def test_contains_contour(self):
        types = list_overlay_types()
        assert "contour" in types
