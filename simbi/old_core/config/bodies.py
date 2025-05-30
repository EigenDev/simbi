from typing import Optional, Sequence
from dataclasses import dataclass, field
from ..types.constants import BodyCapability


@dataclass(frozen=True)
class ImmersedBodyConfig:
    capability: BodyCapability
    mass: float
    velocity: Sequence[float]
    position: Sequence[float]
    radius: float
    two_way_coupling: bool = field(default=False)
    force: Sequence[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
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
    position: Sequence[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: Sequence[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    force: Sequence[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    total_accreted_mass: float = 0.0


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
