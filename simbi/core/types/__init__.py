"""
Core type definitions for simbi.

This module provides type definitions and constants used throughout the framework.
"""

from .bodies import (
    AccretionProperties,
    BaseBody,
    Body,
    BodyCapability,
    BodyData,
    BodyDiagnostics,
    DeformableProperties,
    ElasticProperties,
    GravitationalProperties,
    RigidProperties,
)
from .input import (
    Array,
    CellSpacing,
    HierarchyData,
    IArray,
    LevelData,
    MeshConfig,
    Metadata,
    ProcessedData,
    RawHDF5,
    Reconstruction,
    Solver,
    TimeStepping,
    UArray,
)
from .typing import (
    GasStateFunction,
    GasStateGenerator,
    InitialStateType,
    MHDStateGenerators,
    StaggeredBFieldGenerator,
)

__all__ = [
    "InitialStateType",
    "GasStateGenerator",
    "Solver",
    "CellSpacing",
    "TimeStepping",
    "Reconstruction",
    "ProcessedData",
    "Metadata",
    "BodyData",
    "RawHDF5",
    "MeshConfig",
    "Array",
    "BodyCapability",
    "Body",
    "GravitationalProperties",
    "AccretionProperties",
    "RigidProperties",
    "ElasticProperties",
    "DeformableProperties",
    "BodyData",
    "BodyDiagnostics",
    "BaseBody",
    "IArray",
    "UArray",
    "LevelData",
    "HierarchyData",
    "GasStateFunction",
    "MHDStateGenerators",
    "StaggeredBFieldGenerator",
]
