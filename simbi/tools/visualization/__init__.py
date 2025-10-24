"""
Viz system

This package provides a more type-safe, component-based approach to visualization
compared to the original visualization tools.
"""

from .api import (
    animate,
    plot_histogram,
    plot_line,
    plot_multidim,
    plot_time_series,
)

__all__ = [
    "plot_line",
    "plot_histogram",
    "plot_multidim",
    "plot_time_series",
    "animate",
]
