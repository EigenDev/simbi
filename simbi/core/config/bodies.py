from dataclasses import dataclass
from .constants import BodyType
from typing import Any
from numpy.typing import NDArray
import numpy as np


@dataclass(frozen=True)
class BodyConfig:
    """Base configuration for all body types"""

    body_type: BodyType
    position: NDArray[np.floating[Any]]
    velocity: NDArray[np.floating[Any]]
    mass: float
    radius: float


@dataclass(frozen=True)
class GravitationalBodyConfig(BodyConfig):
    """Configuration for gravitational bodies"""
    softening_length: float = 0.01


@dataclass(frozen=True)
class ElasticBodyConfig(BodyConfig):
    """Configuration for elastic bodies"""

    stiffness: float
    damping: float


@dataclass(frozen=True)
class ViscousBodyConfig(BodyConfig):
    """Configuration for viscous bodies"""

    viscosity: float
    bulk_viscosity: float
    shear_viscosity: float


@dataclass(frozen=True)
class GravitationalSinkConfig(BodyConfig):
    """Configuration for gravitational sinks"""

    accretion_efficiency: float
    softening_length: float
