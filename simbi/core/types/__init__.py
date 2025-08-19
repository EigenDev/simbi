"""
Core type definitions for simbi.

This module provides type definitions and constants used throughout the framework.
"""

from .input import (
    Solver,
    CellSpacing,
    TimeStepping,
    Reconstruction,
    ProcessedData,
    Metadata,
    FieldData,
    RawHDF5,
    MeshConfig,
    Array,
    IArray,
    UArray,
)

from .bodies import (
    BodyCapability,
    Body,
    GravitationalBody,
    AccretionBody,
    RigidBody,
    DeformableBody,
    ElasticBody,
    BodyDiagnostics,
    BodyData,
    BaseBody,
)

from .typing import (
    InitialStateType,
    GasStateGenerator,
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
    "FieldData",
    "BodyData",
    "RawHDF5",
    "MeshConfig",
    "Array",
    "BodyCapability",
    "Body",
    "GravitationalBody",
    "AccretionBody",
    "RigidBody",
    "DeformableBody",
    "ElasticBody",
    "BodyData",
    "BodyDiagnostics",
    "BaseBody",
    "IArray",
    "UArray",
]
