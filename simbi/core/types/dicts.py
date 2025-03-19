from typing import TypedDict, Optional

class ImmersedBodyConfig(TypedDict):
    body_type: str
    mass: float
    velocity: list[float]
    position: list[float]
    radius: float
    specifics: Optional[dict[str, float | int | bool]]  # Specific parameters for the body


class BinaryConfig(TypedDict):
    semi_major: float
    eccentricity: float
    mass_ratio: float
    total_mass: float


class BodySystemConfig(TypedDict):
    """Configuration for generic body system."""
    pass

class GravitationalSystemConfig(BodySystemConfig):
    """Configuration for gravitational system."""

    # General gravitational config
    prescribed_motion: bool
    reference_frame: str
    system_type: str
    binary_config: Optional[BinaryConfig]  # Only used if system_type="binary"

__all__ = [
    "ImmersedBodyConfig",
    "GravitationalSystemConfig",
    "BinaryConfig",
]
