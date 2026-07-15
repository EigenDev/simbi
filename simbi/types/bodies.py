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

from .shape import Shape

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


# capability bits the rust binding actually honors: GRAVITATIONAL builds a
# fixed-potential mass, ACCRETION a black-hole sink, RIGID a drain-off wall
# (the porous surface with porosity 0). ELASTIC / DEFORMABLE have no backend
# path, so a config declaring them is rejected rather than run as a silent lie.
_WIRED_CAPABILITIES = (
    BodyCapability.GRAVITATIONAL | BodyCapability.ACCRETION | BodyCapability.RIGID
)


def _config_error(message: str) -> Exception:
    """the house user-facing config error. imported lazily: simbi.simulation.
    problem imports this module at load time, so a top-level import would cycle."""
    from simbi.simulation.problem import ConfigError

    return ConfigError(message)


@dataclass(frozen=True)
class BinaryComponentConfig:
    mass: float
    radius: float
    is_an_accretor: bool
    softening_length: float
    two_way_coupling: bool
    accretion_radius: float
    sink_rate: float = 0.0
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
    # the porous surface dial (docs/design/50): None keeps the pure drain.
    # porosity p scales the drain channel, (1 - p) the wall channels; the
    # wall rates are k_eta_* c_s / dx (multiplicative dials, zero = channel
    # off exactly, so k_eta_t = 0 is a free-slip surface). p = 0 is a sealed
    # wall (zero mass receipts, exactly); p = 1 reduces to the pure drain.
    porosity: float | None = None
    k_eta_n: float = 0.0
    k_eta_t: float = 0.0
    # the torque-free dial: None keeps the pure drain; a value
    # in [0, 1] selects the isothermal torque-free accretor (the Dittmann sink),
    # where xi = 1 removes mass but no angular momentum. mutually exclusive with
    # porosity (a different surface physics on the same tangential channel).
    torque_free_xi: float | None = None

    def __post_init__(self) -> None:
        if self.accretion_radius <= 0.0:
            raise _config_error(
                f"accretion_radius must be > 0: a sink of zero radius drains no "
                f"gas. got {self.accretion_radius}."
            )
        if self.sink_rate < 0.0:
            raise _config_error(
                f"sink_rate must be >= 0: a negative drain rate injects mass. "
                f"got {self.sink_rate}."
            )
        if self.porosity is not None and not (0.0 <= self.porosity <= 1.0):
            raise _config_error(
                f"porosity must be in [0, 1]: it weights the drain channel by p "
                f"and the wall channels by (1 - p), so outside [0, 1] a channel "
                f"weight is negative (anti-relaxation). got {self.porosity}."
            )
        if self.k_eta_n < 0.0 or self.k_eta_t < 0.0:
            raise _config_error(
                f"k_eta_n and k_eta_t must be >= 0: they are surface-friction "
                f"rate dials; negative is anti-friction. got k_eta_n="
                f"{self.k_eta_n}, k_eta_t={self.k_eta_t}."
            )
        if self.torque_free_xi is not None and not (
            0.0 <= self.torque_free_xi <= 1.0
        ):
            raise _config_error(
                f"torque_free_xi must be in [0, 1]: it is the torque-free "
                f"strength (0 = standard drain, 1 = fully torque-free). got "
                f"{self.torque_free_xi}."
            )
        if self.torque_free_xi is not None and self.porosity is not None:
            raise _config_error(
                "an accretor cannot be both porous and torque-free: porosity and "
                "torque_free_xi are different surface physics on the same "
                "tangential channel. declare one."
            )


@dataclass(frozen=True)
class RigidProperties:
    # a rigid immersed wall: the drain-off porous surface (porosity 0). the wall
    # relaxes the gas velocity toward the body velocity on two channels at rates
    # k_eta_* c_s / dx: k_eta_n (normal, no-penetration) always acts; k_eta_t
    # (tangential) acts only under no-slip (apply_no_slip False = free slip, so
    # the tangential channel is switched off exactly). inertia carries the body's
    # moment of inertia for the (future) two-way rotational coupling. `shape` is an
    # optional signed-distance CSG (body-local frame): None is the analytic sphere of
    # radius `body.radius`; a Shape gives an arbitrary rigid wall whose penalization
    # kernel is runtime-built + JIT-compiled per distinct geometry.
    inertia: float
    apply_no_slip: bool
    k_eta_n: float = 1.0
    k_eta_t: float = 1.0
    shape: Optional[Shape] = None
    # prescribed angular velocity about z (radians/time). nonzero makes a SHAPED wall spin: its
    # mask rotates as R(omega*t) and its no-slip surface drags the gas at omega x r.
    omega: float = 0.0

    def __post_init__(self) -> None:
        if self.shape is not None and not isinstance(self.shape, Shape):
            raise _config_error(
                f"rigid shape must be a Shape (Shape.sphere/box/union/...), got "
                f"{type(self.shape).__name__}."
            )
        if self.omega != 0.0 and self.shape is None:
            raise _config_error(
                "a spinning rigid wall (omega != 0) needs a `shape`: a rotationally symmetric "
                "sphere's mask does not change under rotation (spinning spheres are a follow-on)."
            )
        if self.k_eta_n <= 0.0:
            raise _config_error(
                f"rigid k_eta_n must be > 0: it is the normal (no-penetration) "
                f"wall-relaxation rate dial; zero leaves the wall permeable. got "
                f"{self.k_eta_n}."
            )
        if self.k_eta_t < 0.0:
            raise _config_error(
                f"rigid k_eta_t must be >= 0: it is the tangential (no-slip) "
                f"wall-relaxation rate dial; negative is anti-friction. got "
                f"{self.k_eta_t}."
            )


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

    def __post_init__(self) -> None:
        unsupported = self.capability & ~_WIRED_CAPABILITIES
        if unsupported:
            raise _config_error(
                f"immersed-body capability {unsupported!r} is not wired to the "
                f"backend; only GRAVITATIONAL, ACCRETION, and RIGID are honored. a "
                f"body declaring it would run as a passive gravitating mass."
            )
        if (
            has_capability(self.capability, BodyCapability.RIGID)
            and self.rigid is None
        ):
            raise _config_error(
                "capability RIGID requires a `rigid` property block (the wall "
                "no-slip flag and stiffness dials); without it the wall is undefined."
            )
        # a two-way rigid wall spins up from the gas reaction torque (I domega = torque). that
        # needs a shape (a sphere's mask is rotation-invariant, so it never reaches the spinning
        # kernel) and a positive moment of inertia; otherwise the coupling is a silent no-op.
        if (
            has_capability(self.capability, BodyCapability.RIGID)
            and self.two_way_coupling
            and self.rigid is not None
        ):
            if self.rigid.shape is None:
                raise _config_error(
                    "a two-way rigid wall spins from the gas reaction torque; it needs a `shape` "
                    "(a rotationally symmetric sphere never spins up)."
                )
            if self.rigid.inertia <= 0.0:
                raise _config_error(
                    f"a two-way rigid wall needs inertia > 0 (I domega = torque); got "
                    f"{self.rigid.inertia}."
                )
        if (
            has_capability(self.capability, BodyCapability.ACCRETION)
            and self.accretion is None
        ):
            raise _config_error(
                "capability ACCRETION requires an `accretion` property block; "
                "without it the sink has zero accretion radius and drains nothing."
            )
        if (
            has_capability(self.capability, BodyCapability.GRAVITATIONAL)
            and self.gravitational is None
        ):
            raise _config_error(
                "capability GRAVITATIONAL requires a `gravitational` property "
                "block (the softening length)."
            )


__all__ = [
    "ImmersedBodyConfig",
    "GravitationalSystemConfig",
    "BinaryConfig",
    "BinaryComponentConfig",
    "BodySystemConfig",
    "BodyCapability",
    "has_capability",
]
