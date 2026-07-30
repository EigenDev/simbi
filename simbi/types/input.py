# =============================================================================
# input.py
#
# core type definitions for simulation configuration and data.
# includes enums (CoordSystem, Regime, Solver, etc.) and data structures
# (Metadata, MeshConfig, ProcessedData).
# =============================================================================
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Optional

import numpy as np
from numpy.typing import NDArray

from .bodies import BodySystemConfig, ImmersedBodyConfig

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


class Spacetime(str, ExtendedEnum):
    # the background spacetime, orthogonal to the spatial coord_system and the regime.
    # minkowski is flat (every existing run); schwarzschild selects the GR metric
    # (lapse / densitization / GR-wavespeed kernels) on a spherical grid.
    MINKOWSKI = "minkowski"
    SCHWARZSCHILD = "schwarzschild"
    # ingoing kerr-schild: the same physical schwarzschild vacuum in a HORIZON-PENETRATING chart
    # (regular across r = 2M) — the shift-advection-flux + KS densitization/wavespeed kernels.
    SCHWARZSCHILD_KS = "schwarzschild_ks"
    # spinning kerr in ingoing kerr-schild coordinates: horizon-penetrating, non-diagonal
    # spatial metric (frame dragging), theta-dependent lapse. requires the azimuthal momentum
    # DOF (5-tuple gas rows) and the kerr_spin parameter.
    KERR_KS = "kerr_ks"


class Regime(str, ExtendedEnum):
    NEWTONIAN = "newtonian"
    # the fluid regime names the physics of the fluid alone. the `Spacetime` axis is orthogonal and
    # carries the special-vs-general relativity distinction: a relativistic regime on Minkowski is
    # special-relativistic, on a curved spacetime it is general-relativistic.
    RHD = "rhd"
    RMHD = "rmhd"
    NMHD = "nmhd"
    IMHD = "imhd"
    ISOTHERMAL = "isothermal"


# legacy regime slugs: names that carry "special" in the fluid regime itself. a
# relativity-agnostic slug replaces each. `normalize_regime` remaps them so checkpoints and
# config strings written under those names still load.
_LEGACY_REGIME_SLUGS = {"srhd": "rhd", "srmhd": "rmhd"}


def normalize_regime(regime: str) -> str:
    """map a legacy regime slug to its current name (srhd -> rhd, srmhd -> rmhd) so old checkpoints
    and configs load. a no-op for every current name."""
    return _LEGACY_REGIME_SLUGS.get(regime, regime)


class BoundaryCondition(str, ExtendedEnum):
    OUTFLOW = "outflow"
    REFLECTING = "reflecting"
    DYNAMIC = "dynamic"
    PERIODIC = "periodic"


@dataclass(frozen=True)
class Neumann:
    """a NEUMANN boundary: prescribe the OUTWARD normal derivative `dU/dn = q` per primitive
    variable. the ghost holds `u_edge + q*dist`. a convenience short-circuit for a prescribed-
    gradient wall (a custom/dynamic boundary is the general path). place it in a config's
    `boundary_conditions` list in the face's slot, alongside plain `BoundaryCondition` members.

      rho / pressure : scalar gradients.
      velocity       : per-component gradients (length = spatial dimension); missing -> 0.
    """

    rho: float = 0.0
    velocity: tuple[float, ...] = ()
    pressure: float = 0.0
    # discriminator the exec parser reads (mirrors `BoundaryCondition(...).value`).
    value: ClassVar[str] = "neumann"


@dataclass(frozen=True)
class Robin:
    """a ROBIN boundary: prescribe `a*U_face + b*dU/dn = c` per primitive variable, each coefficient
    a `(a, b, c)` triple. degenerates to Dirichlet (`b=0`) and Neumann (`a=0`). same placement as
    `Neumann`.

      rho / pressure : one `(a, b, c)` triple.
      velocity       : per-component triples (length = spatial dimension); missing -> `(1, 0, 0)`.
    """

    rho: tuple[float, float, float] = (1.0, 0.0, 0.0)
    velocity: tuple[tuple[float, float, float], ...] = ()
    pressure: tuple[float, float, float] = (1.0, 0.0, 0.0)
    value: ClassVar[str] = "robin"


class CellSpacing(str, ExtendedEnum):
    LINEAR = "linear"
    LOG = "log"
    GEOMETRIC = "geometric"


class TimeStepping(str, ExtendedEnum):
    RK1 = "rk1"
    RK2 = "rk2"
    RK3 = "rk3"


class TracerScheme(str, ExtendedEnum):
    DISCRETE = "discrete"
    ITO2 = "ito2"
    ITO3 = "ito3"


class Reconstruction(str, ExtendedEnum):
    PCM = "pcm"
    PLM = "plm"


class Limiter(str, ExtendedEnum):
    # the slope limiter for PLM reconstruction (mirrors the C++ LIMITER enum). MINMOD is the
    # theta-MC family parameterised by `plm_theta` (1 = pure minmod, 2 = MC); VAN_LEER is the smooth
    # harmonic limiter (ignores plm_theta).
    MINMOD = "minmod"
    VAN_LEER = "vanleer"


class Solver(str, ExtendedEnum):
    HLLE = "hlle"
    HLLC = "hllc"
    HLLC_LM = "hllc_lm"  # fleischmann (2020) low-mach / low-dissipation HLLC (newtonian)
    HLLD = "hlld"


class CtMethod(str, ExtendedEnum):
    """constrained-transport edge-EMF scheme (MHD only)."""

    CONTACT = "contact"  # Gardiner & Stone 2005 (default)
    UCT = "uct"  # Del Zanna 2007 / Mignone & Del Zanna 2021 (kills the checkerboard)


class SubCycleMode(str, ExtendedEnum):
    """refinement subcycling schedule.

    only the fixed-ratio schedule is implemented: level `l` advances `2^l` times per root step,
    and the root step is the minimum over levels of that level's own cfl limit times `2^l`, so
    every level lands inside its own cfl. `STANDARD` and `NONE` both name it and are equivalent.

    `ADAPTIVE` (a per-level substep count derived from each level's own cfl, freeing the root from
    the finest level's requirement) and `MANUAL` (a hand-specified count) are REFUSED at
    validation — neither reaches the backend, and accepting them would let a configuration reason
    about a schedule it is not getting.
    """

    STANDARD = "standard"
    ADAPTIVE = "adaptive"
    MANUAL = "manual"
    NONE = "none"


class RefinementMode(str, ExtendedEnum):
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class RefinementCriterion(str, ExtendedEnum):
    GRADIENT = "gradient"
    VALUE = "value"
    CUSTOM = "custom"


@dataclass(frozen=True)
class MeshConfig:
    """Structured mesh configuration"""

    shape: tuple[int, ...]
    bounds_min: tuple[float, ...]
    bounds_max: tuple[float, ...]
    halo_radius: int
    spacing_types: tuple[str, ...]
    spacing_ratios: tuple[float, ...] = (1.0, 1.0, 1.0)

    def _vertices(self, axis: int, count: int) -> Array:
        spacing = self.spacing_types[axis]
        lower = self.bounds_min[axis]
        upper = self.bounds_max[axis]
        if spacing == CellSpacing.LINEAR:
            return np.linspace(lower, upper, count + 1)
        if spacing == CellSpacing.LOG:
            return np.geomspace(lower, upper, count + 1)
        ratio = self.spacing_ratios[axis]
        indices = np.arange(count + 1, dtype=float)
        if abs(ratio - 1.0) < 1.0e-12:
            fractions = indices / count
        else:
            fractions = np.expm1(indices * np.log(ratio)) / np.expm1(
                count * np.log(ratio)
            )
        return lower + (upper - lower) * fractions

    @property
    def effective_dimensions(self) -> int:
        """Calculate effective dimensions based on shape"""
        return sum(1 for dim in self.shape if dim > 1)

    @property
    def x1v(self) -> Array:
        """Get x1 coordinates"""
        return self._vertices(0, self.shape[-1])

    @property
    def x2v(self) -> Array:
        """Get x2 coordinates"""
        return self._vertices(1, self.shape[-2])

    @property
    def x3v(self) -> Array:
        """Get x3 coordinates"""
        return self._vertices(2, self.shape[-3])

    @property
    def x1c(self) -> Array:
        """Get x1 cell centers"""
        if self.spacing_types[0] == CellSpacing.LOG:
            return np.sqrt(self.x1v[:-1] * self.x1v[1:])
        return 0.5 * (self.x1v[:-1] + self.x1v[1:])

    @property
    def x2c(self) -> Array:
        """Get x2 cell centers"""
        if self.spacing_types[1] == CellSpacing.LOG:
            return np.sqrt(self.x2v[:-1] * self.x2v[1:])
        return 0.5 * (self.x2v[:-1] + self.x2v[1:])

    @property
    def x3c(self) -> Array:
        """Get x3 cell centers"""
        if self.spacing_types[2] == CellSpacing.LOG:
            return np.sqrt(self.x3v[:-1] * self.x3v[1:])
        return 0.5 * (self.x3v[:-1] + self.x3v[1:])

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
class Metadata:
    """simulation metadata."""

    # time control
    time: float
    dt: float
    dlogt: float
    tend: float
    iteration: int
    checkpoint_index: int

    # physics
    gamma: float
    cfl: float
    plm_theta: float
    viscosity: float
    resistivity: float

    # domain
    dimensions: int
    coord_system: str
    halo_radius: int

    # flags
    is_mhd: bool
    is_relativistic: bool

    # enums
    regime: str
    solver: str
    reconstruction: str
    timestepping: str

    # optional fields from checkpoint
    checkpoint_interval: float = 0.0
    x1_spacing: str = "linear"
    x1_spacing_ratio: float = 1.0
    x2_spacing: str = "linear"
    x2_spacing_ratio: float = 1.0
    x3_spacing: str = "linear"
    x3_spacing_ratio: float = 1.0
    boundary_conditions: tuple[str, ...] = ()
    initial_time: float = 0.0  # start time from original initial conditions
    # the constant sound speed of an isothermal run; None on energy regimes
    # and on isothermal checkpoints written before the attr existed.
    sound_speed: float | None = None

    # the background spacetime chart
    # (minkowski/schwarzschild/schwarzschild_ks/kerr_ks),
    # orthogonal to coord_system; "minkowski" on flat runs and on pre-attr checkpoints.
    spacetime: str = "minkowski"
    # the schwarzschild geometric mass M (G=c=1); 0 on a flat background.
    schwarzschild_mass: float = 0.0

    # amr fields
    level_dts: tuple[float, ...] = ()
    level_substeps: tuple[int, ...] = ()
    subcycling_mode: str = "none"


@dataclass(frozen=True)
class ProcessedData:
    """Structured data after parsing"""

    fields: dict[str, Array]
    metadata: Metadata
    mesh: MeshConfig

    hierarchy: Optional[HierarchyData] = None
    levels: Optional[list[LevelData]] = None

    body_system: BodySystemConfig | list[ImmersedBodyConfig] | None = None

    @property
    def has_refinement(self) -> bool:
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
            ValueError: If level_id is invalid or data isn't refined
        """
        if level_id == 0:
            return (self.fields, self.mesh)

        if not self.has_refinement:
            raise ValueError("Not an refinement dataset")

        if not self.levels or level_id >= len(self.levels):
            raise ValueError(f"Invalid level ID: {level_id}")

        level = self.levels[level_id]
        return (level.fields, level.mesh)

    def get_refinement_ratio(self, level_id: int) -> Optional[int]:
        """Get refinement ratio between this level and next finer level

        Returns None if this is the finest level.
        """
        if not self.has_refinement or not self.hierarchy:
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
