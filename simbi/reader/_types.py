from dataclasses import dataclass, asdict
from numpy.typing import NDArray
from typing import Literal
from ..core.types.bodies import BodyCapability
import numpy as np

Array = NDArray[np.floating]


@dataclass(frozen=True)
class Unionable:
    def __or__(self, other):
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
    type: Literal["gravitational"] = "gravitational"


@dataclass(frozen=True)
class AccretionBody(BaseBody):
    accretion_efficiency: float
    accretion_radius: float
    total_accreted_mass: float
    accretion_rate: float
    type: Literal["accretion"] = "accretion"


@dataclass(frozen=True)
class RigidBody(BaseBody):
    inertia: float
    apply_no_slip: bool
    type: Literal["rigid"] = "rigid"


@dataclass(frozen=True)
class DeformableBody(BaseBody):
    yield_stress: float
    plastic_strain: float
    type: Literal["deformable"] = "deformable"


@dataclass(frozen=True)
class ElasticBody(BaseBody):
    elastic_modulus: float
    poisson_ratio: float
    type: Literal["elastic"] = "elastic"


# Union type for all body types
Body = GravitationalBody | AccretionBody | RigidBody | DeformableBody | ElasticBody


@dataclass(frozen=True)
class FieldData:
    name: str
    data: Array


@dataclass(frozen=True)
class Metadata:
    """Structured simulation metadata"""

    time: float
    dt: float
    iteration: int
    dimensions: int
    regime: str
    adiabatic_index: float
    is_mhd: bool
    coord_system: str
    boundary_conditions: tuple[str, ...]
    resolution: tuple[int, ...]
    cfl_number: float
    end_time: float
    reconstruction: str
    timestepping: str


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
class MeshConfig:
    """Structured mesh configuration"""

    shape: tuple[int, ...]
    bounds_min: tuple[float, ...]
    bounds_max: tuple[float, ...]
    halo_radius: int
    spacing_types: tuple[str, ...]


@dataclass(frozen=True)
class ProcessedData:
    """Structured data after parsing"""

    fields: dict[str, Array]
    metadata: Metadata
    mesh: MeshConfig
    bodies: dict[str, Body] | None = None


@dataclass(frozen=True)
class RawHDF5:
    """Pure data from file - no processing"""

    fields: dict[str, Array]
    attributes: dict[str, str | float | int | bool]
    groups: dict[str, dict[str, str | float | int | bool | Array]]
