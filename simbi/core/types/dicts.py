from typing import TypedDict, Optional, NotRequired


class ImmersedBodyConfig(TypedDict):
    body_type: str
    mass: float
    velocity: list[float]
    position: list[float]
    radius: float
    specifics: Optional[
        dict[str, float | int | bool]
    ]  # Specific parameters for the body


class BinaryComponentConfig(TypedDict):
    mass: float
    radius: float
    is_an_accretor: bool
    softening_length: float
    two_way_coupling: bool
    accretion_efficiency: float
    accretion_radius: float


class BinaryConfig(TypedDict):
    semi_major: float
    eccentricity: float
    mass_ratio: float
    total_mass: float
    components: list[BinaryComponentConfig]


class BodySystemConfig(TypedDict):
    """Configuration for generic body system."""

    pass


class GravitationalSystemConfig(BodySystemConfig):
    """Configuration for gravitational system."""

    # General gravitational config
    prescribed_motion: NotRequired[bool]
    reference_frame: NotRequired[str]
    system_type: NotRequired[str]
    # Only used if system_type="binary"
    binary_config: NotRequired[BinaryConfig]


__all__ = [
    "ImmersedBodyConfig",
    "GravitationalSystemConfig",
    "BinaryConfig",
    "BinaryComponentConfig",
]
