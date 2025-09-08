"""
Testing utilities for the visualization system.

This module provides helper functions and utilities for testing visualization components
and functionality.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.types import CoordSystem, FieldData, PlotData


def create_test_field_data(
    name: str, values: np.ndarray, domain: Optional[Sequence[np.ndarray]] = None
) -> FieldData:
    """
    Create a FieldData object for testing.

    Args:
        name: Field name
        values: Field values
        domain: Optional domain arrays

    Returns:
        FieldData object
    """
    # Create default domain if not provided
    if domain is None:
        if values.ndim == 1:
            domain = [np.arange(values.shape[0], dtype=np.float64)]
        elif values.ndim == 2:
            domain = [
                np.arange(values.shape[0], dtype=np.float64),
                np.arange(values.shape[1], dtype=np.float64),
            ]
        elif values.ndim == 3:
            domain = [
                np.arange(values.shape[0], dtype=np.float64),
                np.arange(values.shape[1], dtype=np.float64),
                np.arange(values.shape[2], dtype=np.float64),
            ]
        else:
            raise ValueError(f"Unsupported array dimension: {values.ndim}")

    return FieldData(name=name, values=values, domain=domain)


def create_test_plot_data(
    fields: Sequence[FieldData],
    time: float = 0.0,
    dimensions: int = 1,
    coord_system: Union[str, CoordSystem] = "cartesian",
) -> PlotData:
    """
    Create a PlotData object for testing.

    Args:
        fields: Sequence of field data objects
        time: Simulation time
        dimensions: Number of spatial dimensions
        coord_system: Coordinate system

    Returns:
        PlotData object
    """
    # Convert string to CoordSystem enum if needed
    if isinstance(coord_system, str):
        coord_system = CoordSystem(coord_system)

    return PlotData(
        fields=fields,
        time=time,
        dimensions=dimensions,
        coord_system=coord_system,
    )


def create_1d_test_data(
    size: int = 100, field_names: List[str] = ["rho", "p", "v1"]
) -> PlotData:
    """
    Create 1D test data for visualization testing.

    Args:
        size: Size of the domain
        field_names: List of field names to create

    Returns:
        PlotData object with 1D fields
    """
    # Create domain
    x = np.linspace(0, 1, size)

    # Create fields
    fields = []
    for name in field_names:
        if name == "rho":
            # Density with gaussian profile
            values = 1.0 + 2.0 * np.exp(-(((x - 0.5) / 0.1) ** 2))
        elif name == "p":
            # Pressure with step function
            values = np.ones_like(x)
            values[x > 0.5] = 2.0
        elif name == "v1":
            # Velocity with sine wave
            values = 0.1 * np.sin(2 * np.pi * x)
        else:
            # Default to random data for other fields
            values = np.random.random(size)

        fields.append(create_test_field_data(name, values, [x]))

    return create_test_plot_data(fields, time=0.0, dimensions=1)


def create_2d_test_data(
    size: Tuple[int, int] = (50, 50),
    field_names: List[str] = ["rho", "p", "v1", "v2"],
) -> PlotData:
    """
    Create 2D test data for visualization testing.

    Args:
        size: Size of the domain (nx, ny)
        field_names: List of field names to create

    Returns:
        PlotData object with 2D fields
    """
    nx, ny = size

    # Create domain
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Create fields
    fields = []
    for name in field_names:
        if name == "rho":
            # Density with gaussian profile
            R = np.sqrt(X**2 + Y**2)
            values = 1.0 + 2.0 * np.exp(-((R / 0.5) ** 2))
        elif name == "p":
            # Pressure with radial profile
            R = np.sqrt(X**2 + Y**2)
            values = 1.0 + R
        elif name == "v1":
            # x-velocity component
            values = -Y
        elif name == "v2":
            # y-velocity component
            values = X
        else:
            # Default to random data for other fields
            values = np.random.random(size)

        fields.append(create_test_field_data(name, values, [x, y]))

    return create_test_plot_data(fields, time=0.0, dimensions=2)


def create_time_series_data(
    num_frames: int = 10,
    field_names: List[str] = ["rho"],
    domain_size: int = 100,
) -> List[PlotData]:
    """
    Create a time series of plot data for animation testing.

    Args:
        num_frames: Number of frames to create
        field_names: List of field names to create
        domain_size: Size of the spatial domain

    Returns:
        List of PlotData objects representing a time series
    """
    time_series = []

    for t in range(num_frames):
        time = t * 0.1  # Time step

        # Create domain
        x = np.linspace(0, 1, domain_size)

        # Create fields with time dependence
        fields = []
        for name in field_names:
            if name == "rho":
                # Moving gaussian pulse
                center = 0.5 + 0.3 * np.sin(time)
                values = 1.0 + 2.0 * np.exp(-(((x - center) / 0.1) ** 2))
            elif name == "p":
                # Oscillating pressure
                values = 1.0 + 0.5 * np.sin(time) * np.sin(2 * np.pi * x)
            elif name == "v1":
                # Traveling wave
                values = 0.1 * np.sin(2 * np.pi * (x - 0.1 * time))
            else:
                # Default to random data for other fields
                values = np.random.random(domain_size)

            fields.append(create_test_field_data(name, values, [x]))

        time_series.append(
            create_test_plot_data(fields, time=time, dimensions=1)
        )

    return time_series


def mock_figure():
    """
    Create a mock figure and axes for testing components without rendering.

    Returns:
        Tuple of (fig, ax) with mock objects that have the necessary attributes
    """
    import matplotlib.pyplot as plt

    # Create a minimal figure/axes for testing
    # Using a very small size that won't render
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_subplot(111)

    # Set non-interactive backend
    plt.switch_backend("Agg")

    return fig, ax
