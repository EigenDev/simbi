import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Any
from pathlib import Path
from .constants import (
    CoordSystem,
    Regime,
    TimeStepping,
    SpatialOrder,
    CellSpacing,
    Solver,
)


@dataclass(frozen=True)
class GridSettings:
    nx: int  # Number of cells in the x-direction including ghost cells
    ny: int  # Number of cells in the y-direction including ghost cells
    nz: int  # Number of cells in the z-direction including ghost cells
    nxv: int  # Number of vertices in the x-direction
    nyv: int  # Number of vertices in the y-direction
    nzv: int  # Number of vertices in the z-direction
    nghosts: int  # Number of ghost cells

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
        )

    @classmethod
    def from_dict(
        cls, setup: dict[str, dict[str, Any]], spatial_order: str
    ) -> "GridSettings":
        # pad the resolution with ones up to 3D
        dim = len(setup["resolution"])
        resolution = setup["resolution"]
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
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.nx, self.ny, self.nz

    @property
    def vertex_shape(self) -> tuple[int, int, int]:
        return self.nxv, self.nyv, self.nzv

    @property
    def active_shape(self) -> tuple[int, int, int]:
        return (
            self.nx - 2 * self.nghosts,
            self.ny - 2 * self.nghosts,
            self.nz - 2 * self.nghosts,
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
class MeshSettings:
    """Mesh configuration"""

    coord_system: CoordSystem
    bounds: Sequence[Sequence[float]]
    boundary_conditions: Sequence[str]
    dimensionality: int
    mesh_motion: bool
    is_homologous: bool
    x1_spacing: CellSpacing = CellSpacing.LINEAR
    x2_spacing: CellSpacing = CellSpacing.LINEAR
    x3_spacing: CellSpacing = CellSpacing.LINEAR
    scale_factor: Optional[Callable[[float], float]] = None
    scale_factor_derivative: Optional[Callable[[float], float]] = None

    @classmethod
    def from_dict(cls, setup: dict) -> "MeshSettings":
        return cls(**setup)


@dataclass(frozen=True)
class IOSettings:
    data_directory: Path
    checkpoint_file: Optional[Path]
    checkpoint_interval: float
    checkpoint_index: int
    log_output: tuple[bool, int]
    hydro_source_lib: Optional[Path] = None
    gravity_source_lib: Optional[Path] = None
    boundary_source_lib: Optional[Path] = None

    @classmethod
    def from_dict(cls, setup: dict) -> "IOSettings":
        return cls(**setup)


@dataclass(frozen=True)
class SimulationSettings:
    adiabatic_index: float
    tstart: float
    tend: float
    cfl: float
    regime: Regime = Regime.CLASSICAL
    time_stepping: TimeStepping = TimeStepping.RK1
    spatial_order: SpatialOrder = SpatialOrder.PLM
    plm_theta: Optional[float] = None
    quirk_smoothing: bool = False
    is_mhd: bool = False
    dlogt: float = 0.0
    solver: Solver = Solver.HLLC

    @classmethod
    def from_dict(cls, setup: dict) -> "SimulationSettings":
        return cls(
            adiabatic_index=setup["adiabatic_index"],
            tstart=setup["default_start_time"],
            tend=setup["default_end_time"],
            cfl=setup["cfl_number"],
            regime=Regime(setup["regime"]),
            time_stepping=TimeStepping(setup["temporal_order"]),
            spatial_order=SpatialOrder(setup["spatial_order"]),
            plm_theta=setup["plm_theta"],
            quirk_smoothing=setup["use_quirk_smoothing"],
            is_mhd=setup["is_mhd"],
            dlogt=setup["dlogt"],
            solver=Solver(setup["solver"]),
        )
