from typing import ClassVar, Optional, Sequence, Literal, Any
from dataclasses import dataclass, field, asdict
from enum import IntFlag
from numpy.typing import NDArray
import numpy as np

Array = NDArray[np.floating]


class BodyCapability(IntFlag):
    NONE = 0
    GRAVITATIONAL = 1 << 0
    ACCRETION = 1 << 1
    ELASTIC = 1 << 2
    DEFORMABLE = 1 << 3
    RIGID = 1 << 4


def has_capability(body_capability: BodyCapability, capability: BodyCapability) -> bool:
    return bool(body_capability & capability)


@dataclass(frozen=True)
class ImmersedBodyConfig:
    capability: BodyCapability
    mass: float
    velocity: Sequence[float]
    position: Sequence[float]
    radius: float
    two_way_coupling: bool = field(default=False)
    force: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
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
    position: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    velocity: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    force: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
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
    total_mass: Array
    accreted_mass: Array
    accretion_rate: Array


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
class GravitationalBody(BaseBody):
    softening_length: float


@dataclass(frozen=True)
class AccretionBody(BaseBody):
    accretion_efficiency: float
    accretion_radius: float
    total_accreted_mass: float
    accretion_rate: float


@dataclass(frozen=True)
class RigidBody(BaseBody):
    inertia: float
    apply_no_slip: bool


@dataclass(frozen=True)
class DeformableBody(BaseBody):
    yield_stress: float
    plastic_strain: float


@dataclass(frozen=True)
class ElasticBody(BaseBody):
    elastic_modulus: float
    poisson_ratio: float


# Union type for all body types
Body = GravitationalBody | AccretionBody | RigidBody | DeformableBody | ElasticBody

__all__ = [
    "ImmersedBodyConfig",
    "GravitationalSystemConfig",
    "BinaryConfig",
    "BinaryComponentConfig",
    "BodySystemConfig",
    "BodyCapability",
    "has_capability",
]
