"""
Core module for the visualization system.

This module provides the fundamental types, state management, and configuration
classes that form the foundation of the visualization system.
"""

from .config import (
    AnimationConfig,
    HistogramConfig,
    MultidimConfig,
    PlotConfig,
    StyleConfig,
    TemporalConfig,
    VisualizationConfig,
)
from .figure import Figure
from .types import (
    Bounds,
    ColorRange,
    CoordSystem,
    FieldData,
    PlotData,
)

__all__ = [
    # Configuration classes
    "StyleConfig",
    "MultidimConfig",
    "PlotConfig",
    "HistogramConfig",
    "TemporalConfig",
    "AnimationConfig",
    "VisualizationConfig",
    # Type definitions
    "PlotData",
    "FieldData",
    "Bounds",
    "ColorRange",
    "CoordSystem",
    # Figure management
    "Figure",
]
