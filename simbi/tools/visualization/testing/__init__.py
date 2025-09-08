"""
Testing utilities for the visualization system.

This module provides tools and utilities for testing the visualization components
and functionality, including sample data generation and mock objects.
"""

from .helpers import (
    create_1d_test_data,
    create_2d_test_data,
    create_test_field_data,
    create_test_plot_data,
    create_time_series_data,
    mock_figure,
)

__all__ = [
    "create_1d_test_data",
    "create_2d_test_data",
    "create_test_field_data",
    "create_test_plot_data",
    "create_time_series_data",
    "mock_figure",
]
