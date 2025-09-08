from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .bodies import Body

Array = NDArray[np.floating]
IArray = NDArray[np.signedinteger]
UArray = NDArray[np.unsignedinteger]


class ExtendedEnum(Enum):
    @classmethod
    def list(cls: Any) -> list[Any]:
        return list(map(lambda c: c.value, cls))

    def encode(self) -> bytes:
        return bytes(self.value.encode("utf-8"))


class CoordSystem(str, ExtendedEnum):
    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    PLANAR_CYLINDRICAL = "planar_cylindrical"
    AXIS_CYLINDRICAL = "axis_cylindrical"


class Regime(str, ExtendedEnum):
    CLASSICAL = "newtonian"
    SRHD = "srhd"
    SRMHD = "srmhd"


class BoundaryCondition(str, ExtendedEnum):
    OUTFLOW = "outflow"
    REFLECTING = "reflecting"
    DYNAMIC = "dynamic"
    PERIODIC = "periodic"


class CellSpacing(str, ExtendedEnum):
    LINEAR = "linear"
    LOG = "log"


class TimeStepping(str, ExtendedEnum):
    RK1 = "rk1"
    RK2 = "rk2"


class Reconstruction(str, ExtendedEnum):
    PCM = "pcm"
    PLM = "plm"


class Solver(str, ExtendedEnum):
    HLLE = "hlle"
    HLLC = "hllc"
    HLLD = "hlld"


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
class MeshConfig:
    """Structured mesh configuration"""

    shape: tuple[int, ...]
    bounds_min: tuple[float, ...]
    bounds_max: tuple[float, ...]
    halo_radius: int
    spacing_types: tuple[str, ...]

    @property
    def effective_dimensions(self) -> int:
        """Calculate effective dimensions based on shape"""
        return sum(1 for dim in self.shape if dim > 1)

    @property
    def x1v(self) -> Array:
        """Get x1 coordinates"""
        if self.spacing_types[0] == CellSpacing.LINEAR:
            return np.linspace(
                self.bounds_min[0], self.bounds_max[0], self.shape[0]
            )
        else:
            return np.geomspace(
                self.bounds_min[0], self.bounds_max[0], self.shape[0]
            )

    @property
    def x2v(self) -> Array:
        """Get x2 coordinates"""
        if self.spacing_types[1] == CellSpacing.LINEAR:
            return np.linspace(
                self.bounds_min[1], self.bounds_max[1], self.shape[1]
            )
        else:
            return np.geomspace(
                self.bounds_min[1], self.bounds_max[1], self.shape[1]
            )

    @property
    def x3v(self) -> Array:
        """Get x3 coordinates"""
        if self.spacing_types[2] == CellSpacing.LINEAR:
            return np.linspace(
                self.bounds_min[2], self.bounds_max[2], self.shape[2]
            )
        else:
            return np.geomspace(
                self.bounds_min[2], self.bounds_max[2], self.shape[2]
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get coordinate array by key"""
        if key == "x1v":
            return self.x1v
        elif key == "x2v":
            return self.x2v
        elif key == "x3v":
            return self.x3v
        else:
            return default


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
