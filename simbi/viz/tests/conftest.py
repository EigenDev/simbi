# =============================================================================
# conftest.py
#
# pytest fixtures for visualization tests.
# =============================================================================
import numpy as np
import pytest
from matplotlib import pyplot as plt

from simbi.viz.config import (
    FigureConfig,
    OverlayConfig,
    PlotConfig,
    VisualizationConfig,
)
from simbi.viz.types import FieldData


@pytest.fixture
def mock_1d_field():
    """create a simple 1d field for testing."""
    x = np.linspace(0, 10, 100)
    values = np.sin(x)
    return FieldData(
        name="test_1d",
        values=values,
        domain=[x],
    )


@pytest.fixture
def mock_2d_field():
    """create a simple 2d field for testing."""
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    values = np.sqrt(X**2 + Y**2)
    return FieldData(
        name="test_2d",
        values=values,
        domain=[y, x],
    )


@pytest.fixture
def mock_2d_field_with_edges():
    """create a 2d field with edge-based coordinates (n+1 points)."""
    x_edges = np.linspace(0, 1, 51)
    y_edges = np.linspace(0, 1, 51)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers)
    values = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    return FieldData(
        name="test_2d_edges",
        values=values,
        domain=[y_edges, x_edges],
    )


@pytest.fixture
def mock_mach_field():
    """create a mach number-like field for contour testing."""
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    # radial mach number: 0 at origin, increases outward
    values = 2.0 * np.sqrt(X**2 + Y**2)
    return FieldData(
        name="mach",
        values=values,
        domain=[y, x],
    )


@pytest.fixture
def fig_and_ax():
    """create a matplotlib figure and axes for testing."""
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


@pytest.fixture
def default_figure_config():
    """create a default FigureConfig for testing."""
    return FigureConfig()


@pytest.fixture
def default_viz_config():
    """create a default VisualizationConfig for testing."""
    return VisualizationConfig(
        plot=PlotConfig(plot_type="multidim", fields=["rho"])
    )


@pytest.fixture
def overlay_config():
    """create a sample OverlayConfig for testing."""
    return OverlayConfig(
        field="mach",
        component="contour",
        levels=[1.0],
        color="white",
        linewidth=1.5,
    )
