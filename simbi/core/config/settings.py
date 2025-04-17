import dataclasses
from enum import Enum
from dataclasses import dataclass, asdict, field, fields
import textwrap
from typing import Optional, Sequence, Callable, Any, TypeVar, Self
from pathlib import Path
from simbi.core.config.bodies import ImmersedBodyConfig
from simbi.core.types.typing import ExpressionDict
from simbi.functional.helpers import to_tuple_of_tuples
from ...core.config.bodies import BodySystemConfig
from .constants import (
    CoordSystem,
    Regime,
    TimeStepping,
    SpatialOrder,
    CellSpacing,
    Solver,
)

T = TypeVar("T", bound="BaseSettings")


def get_first_existing_key(
    some_dict: dict[str, Any], keys: list[str], default: Any
) -> Any:
    for key in keys:
        if key in some_dict:
            return some_dict[key]
    return default


@dataclass(frozen=True)
class BaseSettings:
    @classmethod
    def update_from(cls, instance: Any, cli_args: dict[str, Any]) -> Any:
        # Filter out None values and process enum types
        processed_args = {}

        for key, value in cli_args.items():
            if value is None or not hasattr(instance, key):
                continue

            current_value = getattr(instance, key)

            # Handle enum conversions
            if isinstance(current_value, Enum) and isinstance(value, str):
                enum_class = type(current_value)
                try:
                    processed_args[key] = enum_class(value)
                except ValueError:
                    # Log warning or raise error
                    continue
            else:
                processed_args[key] = value

        # I didn't know dataclasses has the replace method. Nice!
        # My life is officially 0.1% easier
        return dataclasses.replace(instance, **processed_args)


@dataclass(frozen=True)
class GridSettings(BaseSettings):
    nx: int  # Number of cells in the x-direction including ghost cells
    ny: int  # Number of cells in the y-direction including ghost cells
    nz: int  # Number of cells in the z-direction including ghost cells
    nxv: int  # Number of vertices in the x-direction
    nyv: int  # Number of vertices in the y-direction
    nzv: int  # Number of vertices in the z-direction
    nghosts: int  # Number of ghost cells
    dimensionality: int

    @classmethod
    def from_dict(cls, setup: dict[str, Any], spatial_order: str) -> "GridSettings":
        # pad the resolution with ones up to 3D
        dim: int = len(setup["resolution"])
        resolution: tuple[int, ...] = setup["resolution"]
        if len(resolution) < 3:
            resolution += (1,) * (3 - len(resolution))
        nx_active, ny_active, nz_active = resolution
        nghosts = 2 * (1 + (spatial_order == "plm"))
        return cls(
            nx=nx_active + nghosts,
            ny=ny_active + nghosts * (dim > 1),
            nz=nz_active + nghosts * (dim > 2),
            nxv=nx_active + 1,
            nyv=ny_active + 1,
            nzv=nz_active + 1,
            nghosts=nghosts,
            dimensionality=dim,
        )

    @classmethod
    def update_from(cls, instance: "GridSettings", cli_args: dict[str, Any]) -> Self:
        self_params = asdict(instance)
        if cli_args["spatial_order"] is None:
            return cls(**self_params)

        nghosts = 2 * (1 + (cli_args["spatial_order"] == "plm"))
        nx_active, ny_active, nz_active = instance.resolution
        dimensionality = instance.dimensionality
        return cls(
            nx=nx_active + nghosts,
            ny=ny_active + nghosts * (dimensionality > 1),
            nz=nz_active + nghosts * (dimensionality > 2),
            nxv=nx_active + 1,
            nyv=ny_active + 1,
            nzv=nz_active + 1,
            nghosts=nghosts,
            dimensionality=dimensionality,
        )

    def to_execution_dict(self) -> dict[str, Any]:
        """convert the settings to execution format dict"""
        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "nxv": self.nxv,
            "nyv": self.nyv,
            "nzv": self.nzv,
            "nghosts": self.nghosts,
            "dimensionality": self.dimensionality,
        }

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.nx, self.ny, self.nz

    @property
    def vertex_shape(self) -> tuple[int, int, int]:
        return self.nxv, self.nyv, self.nzv

    @property
    def active_shape(self) -> tuple[int, int, int]:
        return (
            self.nx - self.nghosts,
            max(1, self.ny - self.nghosts),
            max(1, self.nz - self.nghosts),
        )

    @property
    def active_vertex_shape(self) -> tuple[int, int, int]:
        return (
            self.nxv - 1,
            self.nyv - 1,
            self.nzv - 1,
        )

    @property
    def resolution(self) -> tuple[int, int, int]:
        return self.active_shape


@dataclass(frozen=True)
class MeshSettings(BaseSettings):
    """Mesh configuration"""

    coord_system: CoordSystem
    bounds: Sequence[tuple[float, float]] | Sequence[float]
    boundary_conditions: Sequence[str]
    dimensionality: int
    mesh_motion: bool
    is_homologous: bool
    x1_spacing: CellSpacing = CellSpacing.LINEAR
    x2_spacing: CellSpacing = CellSpacing.LINEAR
    x3_spacing: CellSpacing = CellSpacing.LINEAR
    scale_factor: Optional[Callable[[float], float]] = None
    scale_factor_derivative: Optional[Callable[[float], float]] = None

    def effective_dim(self, resolution: Sequence[int]) -> int:
        return sum(1 for _ in filter(lambda r: r > 1, resolution))

    @classmethod
    def from_dict(cls, setup: dict[str, Any]) -> "MeshSettings":
        return cls(**setup)

    def to_execution_dict(self) -> dict[str, Any]:
        """convert the settings to execution format dict"""
        return {
            "coord_system": self.coord_system,
            "bounds": to_tuple_of_tuples(self.bounds),
            "boundary_conditions": [bc for bc in self.boundary_conditions],
            "dimensionality": self.dimensionality,
            "mesh_motion": self.mesh_motion,
            "is_homologous": self.is_homologous,
            "x1_spacing": self.x1_spacing,
            "x2_spacing": self.x2_spacing,
            "x3_spacing": self.x3_spacing,
            "scale_factor": self.scale_factor,
            "scale_factor_derivative": self.scale_factor_derivative,
        }


@dataclass(frozen=True)
class IOSettings(BaseSettings):
    data_directory: Path
    checkpoint_file: Optional[Path]
    checkpoint_interval: float
    checkpoint_index: int
    log_output: bool
    hydro_expressions: Optional[ExpressionDict] = None
    gravity_expressions: Optional[ExpressionDict] = None
    bx1_outer_expressions: Optional[ExpressionDict] = None
    bx1_inner_expressions: Optional[ExpressionDict] = None
    bx2_outer_expressions: Optional[ExpressionDict] = None
    bx2_inner_expressions: Optional[ExpressionDict] = None
    bx3_outer_expressions: Optional[ExpressionDict] = None
    bx3_inner_expressions: Optional[ExpressionDict] = None

    @staticmethod
    def try_get_path(path: str) -> Optional[Path]:
        return Path(path) if path else None

    @classmethod
    def from_dict(cls, setup: dict[str, Any]) -> "IOSettings":
        return cls(
            data_directory=Path(setup["data_directory"]),
            checkpoint_file=Path(setup["checkpoint_file"] or ""),
            checkpoint_interval=setup["checkpoint_interval"],
            checkpoint_index=setup["checkpoint_index"],
            log_output=setup["log_output"],
            hydro_expressions=setup.get("hydro_expressions"),
            gravity_expressions=setup.get("gravity_expressions"),
            bx1_outer_expressions=setup.get("bx1_outer_expressions"),
            bx1_inner_expressions=setup.get("bx1_inner_expressions"),
            bx2_outer_expressions=setup.get("bx2_outer_expressions"),
            bx2_inner_expressions=setup.get("bx2_inner_expressions"),
            bx3_outer_expressions=setup.get("bx3_outer_expressions"),
            bx3_inner_expressions=setup.get("bx3_inner_expressions"),
        )

    def to_execution_dict(self) -> dict[str, Any]:
        """convert the settings to execution format dict"""
        return {
            "data_directory": f"{Path(self.data_directory)}/",
            "checkpoint_file": (
                str(self.checkpoint_file) if self.checkpoint_file else ""
            ),
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_index": self.checkpoint_index,
            "log_output": self.log_output,
            "hydro_expressions": self.hydro_expressions,
            "gravity_expressions": self.gravity_expressions,
            "bx1_outer_expressions": self.bx1_outer_expressions,
            "bx1_inner_expressions": self.bx1_inner_expressions,
            "bx2_outer_expressions": self.bx2_outer_expressions,
            "bx2_inner_expressions": self.bx2_inner_expressions,
            "bx3_outer_expressions": self.bx3_outer_expressions,
            "bx3_inner_expressions": self.bx3_inner_expressions,
        }


@dataclass(frozen=True)
class SimulationSettings(BaseSettings):
    adiabatic_index: float
    tstart: float
    tend: float
    cfl: float
    regime: Regime = Regime.CLASSICAL
    temporal_order: TimeStepping = TimeStepping.RK1
    spatial_order: SpatialOrder = SpatialOrder.PLM
    plm_theta: float = 1.5
    quirk_smoothing: bool = False
    is_mhd: bool = False
    dlogt: float = 0.0
    solver: Solver = Solver.HLLC
    bodies: list[ImmersedBodyConfig] = field(default_factory=list)
    sound_speed: float = 0.0
    isothermal: bool = False
    body_system: Optional[BodySystemConfig] = None

    @classmethod
    def from_dict(cls, setup: dict[str, Any]) -> "SimulationSettings":
        return cls(
            adiabatic_index=setup["adiabatic_index"],
            tstart=setup["default_start_time"],
            tend=setup["default_end_time"],
            cfl=setup["cfl_number"],
            regime=Regime(setup["regime"]),
            temporal_order=TimeStepping(setup["temporal_order"]),
            spatial_order=SpatialOrder(setup["spatial_order"]),
            plm_theta=setup["plm_theta"],
            quirk_smoothing=setup["use_quirk_smoothing"],
            is_mhd=setup["is_mhd"],
            dlogt=setup["dlogt"],
            solver=Solver(setup["solver"]),
            bodies=setup["immersed_bodies"],
            sound_speed=setup["ambient_sound_speed"],
            isothermal=setup["isothermal"],
            body_system=get_first_existing_key(
                setup, ["gravitational_system", "elastic_system", "rigid_system"], None
            ),
        )

    def to_execution_dict(self) -> dict[str, Any]:
        """convert the settings to execution format dict"""
        return {
            "adiabatic_index": self.adiabatic_index,
            "tstart": self.tstart,
            "tend": self.tend,
            "cfl": self.cfl,
            "regime": self.regime.value,
            "temporal_order": self.temporal_order.value,
            "spatial_order": self.spatial_order.value,
            "plm_theta": self.plm_theta,
            "quirk_smoothing": self.quirk_smoothing,
            "is_mhd": self.is_mhd,
            "dlogt": self.dlogt,
            "solver": self.solver.value,
            "bodies": [asdict(x) for x in self.bodies],
            "sound_speed": self.sound_speed,
            "isothermal": self.isothermal,
            "body_system": asdict(self.body_system) if self.body_system else None,
        }
