from typing import Optional, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ImmersedBodyConfig:
    body_type: str
    mass: float
    velocity: Sequence[float]
    position: Sequence[float]
    radius: float
    specifics: Optional[dict[str, float | int | bool]] = None


@dataclass(frozen=True)
class BinaryComponentConfig:
    mass: float
    radius: float
    is_an_accretor: bool
    softening_length: float
    two_way_coupling: bool
    accretion_efficiency: float
    accretion_radius: float


@dataclass(frozen=True)
class BinaryConfig:
    semi_major: float
    eccentricity: float
    mass_ratio: float
    total_mass: float
    components: Sequence[BinaryComponentConfig]


@dataclass(frozen=True)
class BodySystemConfig:
    """Configuration for generic body system."""

    pass


@dataclass(frozen=True)
class GravitationalSystemConfig(BodySystemConfig):
    """Configuration for gravitational system."""

    # General gravitational config
    prescribed_motion: bool
    reference_frame: str
    system_type: str
    # Only used if system_type="binary"
    binary_config: BinaryConfig


__all__ = [
    "ImmersedBodyConfig",
    "GravitationalSystemConfig",
    "BinaryConfig",
    "BinaryComponentConfig",
]
