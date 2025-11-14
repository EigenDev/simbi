from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .bodies import Body, BodySystemConfig, ImmersedBodyConfig

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
    NEWTONIAN = "newtonian"
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


class SubCycleMode(str, ExtendedEnum):
    STANDARD = "standard"
    ADAPTIVE = "adaptive"
    NONE = "none"


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
    plm_theta: float
    solver: str
    checkpoint_index: int
    checkpoint_interval: float
    x1_spacing: str
    x2_spacing: str
    x3_spacing: str
    halo_radius: int
    system_info: dict[str, str | float | int | bool | Array] | None = None


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
                self.bounds_min[0], self.bounds_max[0], self.shape[0] + 1
            )
        else:
            return np.geomspace(
                self.bounds_min[0], self.bounds_max[0], self.shape[0] + 1
            )

    @property
    def x2v(self) -> Array:
        """Get x2 coordinates"""
        if self.spacing_types[1] == CellSpacing.LINEAR:
            return np.linspace(
                self.bounds_min[1], self.bounds_max[1], self.shape[1] + 1
            )
        else:
            return np.geomspace(
                self.bounds_min[1], self.bounds_max[1], self.shape[1] + 1
            )

    @property
    def x3v(self) -> Array:
        """Get x3 coordinates"""
        if self.spacing_types[2] == CellSpacing.LINEAR:
            return np.linspace(
                self.bounds_min[2], self.bounds_max[2], self.shape[2] + 1
            )
        else:
            return np.geomspace(
                self.bounds_min[2], self.bounds_max[2], self.shape[2] + 1
            )

    @property
    def x1c(self) -> Array:
        """Get x1 cell centers"""
        if self.spacing_types[0] == CellSpacing.LINEAR:
            return 0.5 * (self.x1v[:-1] + self.x1v[1:])
        else:
            return np.sqrt(self.x1v[:-1] * self.x1v[1:])

    @property
    def x2c(self) -> Array:
        """Get x2 cell centers"""
        if self.spacing_types[1] == CellSpacing.LINEAR:
            return 0.5 * (self.x2v[:-1] + self.x2v[1:])
        else:
            return np.sqrt(self.x2v[:-1] * self.x2v[1:])

    @property
    def x3c(self) -> Array:
        """Get x3 cell centers"""
        if self.spacing_types[2] == CellSpacing.LINEAR:
            return 0.5 * (self.x3v[:-1] + self.x3v[1:])
        else:
            return np.sqrt(self.x3v[:-1] * self.x3v[1:])

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
class LevelData:
    level_id: int
    mesh: MeshConfig
    fields: dict[str, Array]
    ref_ratio: int | None  # ratio to next finer level


@dataclass(frozen=True)
class HierarchyData:
    num_levels: int
    levels: list[LevelData]
    ref_ratios: list[int]  # between levels


@dataclass(frozen=True)
class ProcessedData:
    """Structured data after parsing"""

    fields: dict[str, Array]
    metadata: Metadata
    mesh: MeshConfig

    hierarchy: Optional[HierarchyData] = None
    levels: Optional[list[LevelData]] = None

    bodies: dict[str, Body] | None = None
    body_system: BodySystemConfig | list[ImmersedBodyConfig] | None = None

    @property
    def has_fmr(self) -> bool:
        return self.hierarchy is not None and self.levels is not None

    @property
    def num_levels(self) -> int:
        """get number of refinment levels"""
        if self.hierarchy is None:
            return 1
        return self.hierarchy.num_levels

    def get_level(self, level_id: int) -> tuple[dict[str, Array], MeshConfig]:
        """Get data for a specific level

        Args:
            level_id: Level ID to retrieve (0 is base level)

        Returns:
            Tuple of (fields, mesh) for the requested level

        Raises:
            ValueError: If level_id is invalid or data isn't FMR
        """
        if level_id == 0:
            return (self.fields, self.mesh)

        if not self.has_fmr:
            raise ValueError("Not an FMR dataset")

        if not self.levels or level_id >= len(self.levels):
            raise ValueError(f"Invalid level ID: {level_id}")

        level = self.levels[level_id]
        return (level.fields, level.mesh)

    def get_refinement_ratio(self, level_id: int) -> Optional[int]:
        """Get refinement ratio between this level and next finer level

        Returns None if this is the finest level.
        """
        if not self.has_fmr or not self.hierarchy:
            return None

        if level_id >= len(self.hierarchy.ref_ratios):
            return None

        return self.hierarchy.ref_ratios[level_id]


@dataclass(frozen=True)
class RawHDF5:
    """Pure data from file - no processing"""

    fields: dict[str, Array]
    attributes: dict[str, str | float | int | bool]
    groups: dict[str, dict[str, str | float | int | bool | Array]]
