# =============================================================================
# bodies.py
#
# type definitions for immersed bodies and gravitational systems.
# includes Body, ImmersedBodyConfig, BinaryConfig, and property dataclasses.
# =============================================================================
from dataclasses import asdict, dataclass, field
from enum import IntFlag
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


class BodyCapability(IntFlag):
    NONE = 0
    GRAVITATIONAL = 1 << 0
    ACCRETION = 1 << 1
    ELASTIC = 1 << 2
    DEFORMABLE = 1 << 3
    RIGID = 1 << 4


def has_capability(
    body_capability: BodyCapability, capability: BodyCapability
) -> bool:
    return bool(body_capability & capability)


@dataclass(frozen=True)
class BinaryComponentConfig:
    mass: float
    radius: float
    is_an_accretor: bool
    softening_length: float
    two_way_coupling: bool
    accretion_radius: float
    sink_rate: float = 0.0
    sink_delta: float = 1.0  # 1 is standard sink, 0 is torque-free
    position: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    velocity: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    force: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    total_accreted_mass: float = 0.0

    def to_body_config(self) -> dict:
        """convert to format expected by c++ factory with nested property dicts."""
        config = {
            "mass": self.mass,
            "radius": self.radius,
            "position": self.position,
            "velocity": self.velocity,
            "force": self.force,
            "two_way_coupling": self.two_way_coupling,
        }

        # all binary components have gravitational properties
        config["gravitational"] = {"softening_length": self.softening_length}

        # add accretion properties if this component is an accretor
        if self.is_an_accretor:
            config["accretion"] = {
                "accretion_radius": self.accretion_radius,
                "sink_rate": self.sink_rate,
                "sink_delta": self.sink_delta,
                "total_accreted_mass": self.total_accreted_mass,
            }

        return config


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

    def to_dict(self) -> dict:
        """custom serialization for c++ factory with proper nested structure."""
        result = {
            "prescribed_motion": self.prescribed_motion,
            "reference_frame": self.reference_frame,
            "system_type": self.system_type,
        }

        if self.binary_config:
            result["binary_config"] = {
                "semi_major": self.binary_config.semi_major,
                "eccentricity": self.binary_config.eccentricity,
                "mass_ratio": self.binary_config.mass_ratio,
                "total_mass": self.binary_config.total_mass,
                "components": [
                    comp.to_body_config()
                    for comp in self.binary_config.components
                ],
            }

        return result


@dataclass(frozen=True)
class BodyData:
    mass: float
    radius: float
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    # Type-specific fields would need handling


@dataclass(frozen=True)
class BodyDiagnostics:
    force_components: dict[str, Array]  # force_1, force_2, force_3
    torque_components: dict[str, Array]  # torque_1, torque_2, torque_3
    cumulative_mass_delta: Array
    accretion_rate: Array


@dataclass(frozen=True)
class GravitationalProperties:
    softening_length: float


@dataclass(frozen=True)
class AccretionProperties:
    accretion_radius: float
    sink_rate: float = 0.0
    total_accreted_mass: float = 0.0
    accretion_rate: float = 0.0
    sink_delta: float = 1.0  # 1 is standard sink, 0 is torque-free


@dataclass(frozen=True)
class RigidProperties:
    inertia: float
    apply_no_slip: bool


@dataclass(frozen=True)
class DeformableProperties:
    yield_stress: float
    plastic_strain: float


@dataclass(frozen=True)
class ElasticProperties:
    elastic_modulus: float
    poisson_ratio: float


@dataclass(frozen=True)
class Unionable:
    def __or__(self, other: Any) -> Any:
        return self.__class__(**asdict(self) | asdict(other))


@dataclass(frozen=True)
class BaseBody(Unionable):
    mass: float
    radius: float
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    capabilities: BodyCapability


@dataclass(frozen=True)
class Body(BaseBody):
    force: tuple[float, ...] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    torque: tuple[float, ...] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    gravitational: Optional[GravitationalProperties] = None
    accretion: Optional[AccretionProperties] = None
    rigid: Optional[RigidProperties] = None
    deformable: Optional[DeformableProperties] = None
    elastic: Optional[ElasticProperties] = None


@dataclass(frozen=True)
class ImmersedBodyConfig:
    capability: BodyCapability
    mass: float
    velocity: Sequence[float]
    position: Sequence[float]
    radius: float
    two_way_coupling: bool = field(default=False)
    force: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    gravitational: Optional[GravitationalProperties] = None
    accretion: Optional[AccretionProperties] = None
    rigid: Optional[RigidProperties] = None
    deformable: Optional[DeformableProperties] = None
    elastic: Optional[ElasticProperties] = None


__all__ = [
    "ImmersedBodyConfig",
    "GravitationalSystemConfig",
    "BinaryConfig",
    "BinaryComponentConfig",
    "BodySystemConfig",
    "BodyCapability",
    "has_capability",
]
