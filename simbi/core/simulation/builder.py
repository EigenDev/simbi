import numpy as np
from dataclasses import dataclass
from ...functional import to_tuple_of_tuples, pipe
from typing import Any, Protocol, Sequence, TypedDict
from .state_init import SimulationBundle
from ..managers.boundary import BoundaryManager
from pathlib import Path
from ...io.logging import logger
from numpy.typing import NDArray


class SimStateDict(TypedDict):
    """TypedDict for simulation state"""

    gamma: float
    tstart: float
    tend: float
    cfl: float
    dlogt: float
    plm_theta: float
    checkpoint_interval: float
    checkpoint_idx: int
    regime: str | bytes
    dimensionality: int
    data_directory: str
    boundary_conditions: list[str]
    spatial_order: str | bytes
    temporal_order: str | bytes
    x1_spacing: str
    x2_spacing: str
    x3_spacing: str
    solver: str | bytes
    coord_system: str | bytes
    quirk_smoothing: bool
    nx: int
    ny: int
    nz: int
    x1bounds: tuple[float, float]
    x2bounds: tuple[float, float]
    x3bounds: tuple[float, float]
    bfield: list[Any]
    hydro_source_lib: str
    gravity_source_lib: str
    boundary_source_lib: str
    mesh_motion: bool
    homologous: bool


@dataclass
class SimStateBuilder:
    """Builder for simulation state configuration"""

    @staticmethod
    def prepare_data_directory(data_directory: str) -> None:
        """Ensure data directory exists"""
        data_path = Path(data_directory)
        if not data_path.is_dir():
            data_path.mkdir(parents=True)
            logger.info(f"Created data directory: {data_path}")

    @staticmethod
    def get_bounds(
        bounds: Sequence[tuple[float, float]] | Sequence[float], effective_dim: int
    ) -> Sequence[tuple[float, float]]:
        """Get bounds with defaults for missing dimensions"""
        bounds_copy = to_tuple_of_tuples(bounds)
        for bnd in bounds_copy:
            if len(bnd) != 2:
                raise ValueError(
                    f"Bounds must be a tuple of length 2, got {bounds_copy} instead"
                )
        defaults: list[tuple[float, float]] = [
            (0.0, 1.0) for _ in range(3 - effective_dim)
        ]

        # if the user for some reason inputs a 3-tuple for the bounds, and
        # the effective dimensions are not 3, then this is an error
        if len(bounds) == 3 and (ndef := len(defaults) != 0):
            raise ValueError(
                f"Detected a run wtih effective dimensions {effective_dim}, but the user is inserting undefined bounds. Please remove the bounds: {bounds[-ndef:]} from your configuration"
            )
        return (*bounds_copy, *tuple(defaults))

    @classmethod
    def to_dict(cls, state: "SimStateDict") -> dict[str, Any]:
        """Convert SimStateDict to raw dictionary with encoded strings"""
        raw_dict: dict[str, Any] = dict(state)

        # Handle data directory
        path = Path(raw_dict["data_directory"])
        raw_dict["data_directory"] = (
            f"{path}/".encode("utf-8")
            if str(path) and not str(path).endswith("/")
            else str(path).encode("utf-8")
        )

        # Encode string values
        string_fields = {
            "boundary_conditions": lambda x: [bc.encode("utf-8") for bc in x],
            "x1_spacing": lambda x: x.encode("utf-8"),
            "x2_spacing": lambda x: x.encode("utf-8"),
            "x3_spacing": lambda x: x.encode("utf-8"),
            "hydro_source_lib": lambda x: x.encode("utf-8"),
            "gravity_source_lib": lambda x: x.encode("utf-8"),
            "boundary_source_lib": lambda x: x.encode("utf-8"),
        }

        for field, encoder in string_fields.items():
            if field in raw_dict:
                raw_dict[field] = encoder(raw_dict[field])  # type: ignore

        return raw_dict

    @staticmethod
    def build(bundle: SimulationBundle) -> dict[str, Any]:
        """Build simulation state from bundle"""
        mesh = bundle.mesh_config
        grid = bundle.grid_config
        io = bundle.io_config
        sim = bundle.sim_config

        # Get bounds with defaults
        x1bounds, x2bounds, x3bounds = SimStateBuilder.get_bounds(
            mesh.bounds, mesh.effective_dim(grid.resolution)
        )

        bcs = pipe(
            mesh.boundary_conditions,
            lambda bcs: BoundaryManager.validate_conditions(bcs, mesh.dimensionality),
            lambda bcs: BoundaryManager.check_and_fix_curvlinear_conditions(
                conditions=bcs,
                coord_system=mesh.coord_system,
                contains_boundary_source_terms=io.boundary_source_lib is not None,
                dim=mesh.dimensionality,
            ),
        )

        # Build base state dict
        effective_bfields: list[Any] = [
            [0.0],
            [0.0],
            [0.0],
        ]
        if bundle.staggered_bfields is not None:
            effective_bfields = [b.flat for b in bundle.staggered_bfields]
        state_dict = SimStateDict(
            gamma=sim.adiabatic_index,
            tstart=sim.tstart,
            tend=sim.tend,
            cfl=sim.cfl,
            dlogt=sim.dlogt,
            plm_theta=sim.plm_theta,
            checkpoint_interval=io.checkpoint_interval,
            checkpoint_idx=io.checkpoint_index,
            data_directory=str(io.data_directory),
            boundary_conditions=[str(bc) for bc in bcs],
            spatial_order=sim.spatial_order.encode(),
            temporal_order=sim.temporal_order.encode(),
            x1_spacing=mesh.x1_spacing,
            x2_spacing=mesh.x2_spacing,
            x3_spacing=mesh.x3_spacing,
            solver=sim.solver.encode(),
            regime=sim.regime.encode(),
            dimensionality=mesh.dimensionality,
            coord_system=mesh.coord_system.encode(),
            quirk_smoothing=sim.quirk_smoothing,
            nx=grid.nx,
            ny=grid.ny,
            nz=grid.nz,
            x1bounds=x1bounds,
            x2bounds=x2bounds,
            x3bounds=x3bounds,
            bfield=effective_bfields,
            hydro_source_lib=str(io.hydro_source_lib or ""),
            gravity_source_lib=str(io.gravity_source_lib or ""),
            boundary_source_lib=str(io.boundary_source_lib or ""),
            mesh_motion=mesh.mesh_motion,
            homologous=mesh.is_homologous,
        )

        SimStateBuilder.prepare_data_directory(state_dict["data_directory"])
        x = SimStateBuilder.to_dict(state_dict)
        # Encode strings and return
        return SimStateBuilder.to_dict(state_dict)
