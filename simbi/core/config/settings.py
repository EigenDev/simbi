from dataclasses import dataclass, asdict, field
from typing import Optional, Sequence, Callable, Any, Optional, TypeVar
from pathlib import Path

from simbi.functional.helpers import to_tuple_of_tuples

from ...core.types.dicts import BodySystemConfig
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
        self_params = asdict(instance)
        self_params.update(
            (k, cli_args[k] or self_params[k])
            for k in set(cli_args).intersection(self_params)
        )
        return cls(**self_params)


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
    def from_resolution(
        cls, resolution: Sequence[int], nghosts: int, dim: int
    ) -> "GridSettings":
        nx_active, ny_active, nz_active = resolution
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
    def update_from(cls, instance: "GridSettings", cli_args: dict[str, Any]) -> Any:
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
    hydro_source_lib: Optional[Path] = None
    gravity_source_lib: Optional[Path] = None
    boundary_source_lib: Optional[Path] = None

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
            hydro_source_lib=IOSettings.try_get_path(setup["hydro_source_lib"]),
            gravity_source_lib=IOSettings.try_get_path(setup["gravity_source_lib"]),
            boundary_source_lib=IOSettings.try_get_path(setup["boundary_source_lib"]),
        )

    def to_execution_dict(self) -> dict[str, Any]:
        """convert the settings to execution format dict"""
        return {
            "data_directory": f"{self.data_directory}/",
            "checkpoint_file": (
                str(self.checkpoint_file) if self.checkpoint_file else ""
            ),
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_index": self.checkpoint_index,
            "log_output": self.log_output,
            "hydro_source_lib": (
                str(self.hydro_source_lib) if self.hydro_source_lib else ""
            ),
            "gravity_source_lib": (
                str(self.gravity_source_lib) if self.gravity_source_lib else ""
            ),
            "boundary_source_lib": (
                str(self.boundary_source_lib) if self.boundary_source_lib else ""
            ),
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
    bodies: list[dict[str, Any]] = field(default_factory=list)
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
            sound_speed=setup["sound_speed"],
            isothermal=setup["isothermal"],
            body_system=get_first_existing_key(
                setup, ["gravitational_system", "elastic_system", "rigid_system"], None
            ),
        )

    @classmethod
    def update_from(cls, instance: Any, cli_args: dict[str, Any]) -> Any:
        self_params = asdict(instance)
        self_params.update(
            (k, cli_args[k] if cli_args[k] is not None else self_params[k])
            for k in set(cli_args).intersection(self_params)
        )
        return cls(
            adiabatic_index=self_params["adiabatic_index"],
            tstart=self_params["tstart"],
            tend=self_params["tend"],
            cfl=self_params["cfl"],
            regime=Regime(self_params["regime"]),
            temporal_order=TimeStepping(self_params["temporal_order"]),
            spatial_order=SpatialOrder(self_params["spatial_order"]),
            plm_theta=self_params["plm_theta"],
            quirk_smoothing=self_params["quirk_smoothing"],
            is_mhd=self_params["is_mhd"],
            dlogt=self_params["dlogt"],
            solver=Solver(self_params["solver"]),
            bodies=self_params["bodies"],
            sound_speed=self_params["sound_speed"],
            isothermal=self_params["isothermal"],
            body_system=self_params["body_system"],
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
            "solver": self.solver,
            "bodies": self.bodies,
            "sound_speed": self.sound_speed,
            "isothermal": self.isothermal,
            "body_system": dict(self.body_system) if self.body_system else None,
        }
