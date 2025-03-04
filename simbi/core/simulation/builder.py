from dataclasses import dataclass
from ...functional import get_iterable, pipe
from typing import Any, Protocol
from .state_init import SimulationBundle
from ..managers.boundary import BoundaryManager
from pathlib import Path


class SimulationState(Protocol):
    """Protocol defining simulation state interface"""

    gamma: float
    tstart: float
    tend: float
    cfl: float
    dlogt: float
    plm_theta: float
    checkpoint_interval: float
    checkpoint_idx: int
    data_directory: bytes
    boundary_conditions: list[bytes]
    spatial_order: bytes
    temporal_order: bytes
    x1_spacing: bytes
    x2_spacing: bytes
    x3_spacing: bytes
    solver: bytes
    coord_system: bytes
    quirk_smoothing: bool
    nx: int
    ny: int
    nz: int
    x1bounds: tuple[float, float]
    x2bounds: tuple[float, float]
    x3bounds: tuple[float, float]
    bfield: list[list[float]]
    hydro_source_lib: bytes
    gravity_source_lib: bytes
    boundary_source_lib: bytes
    mesh_motion: bool
    homologous: bool


@dataclass
class SimStateBuilder:
    """Builder for simulation state configuration"""

    @staticmethod
    def get_bounds(
        bounds: list[tuple[float, float]], dim: int
    ) -> tuple[tuple[float, float], ...]:
        """Get bounds with defaults for missing dimensions"""
        bounds_copy = get_iterable(bounds, tuple)
        defaults = [(0.0, 1.0) for _ in range(3 - dim)]
        return bounds_copy, *tuple(defaults)

    @staticmethod
    def encode_strings(state_dict: dict[str, Any]) -> dict[str, Any]:
        """Encode strings to bytes"""
        if "data_directory" in state_dict:
            path = Path(state_dict["data_directory"])
            if str(path) and not str(path).endswith("/"):
                state_dict["data_directory"] = f"{path}/".encode("utf-8")
            else:
                state_dict["data_directory"] = str(path).encode("utf-8")

        string_keys = [
            "coord_system",
            "solver",
            "spatial_order",
            "temporal_order",
            "boundary_conditions",
            "x1_spacing",
            "x2_spacing",
            "x3_spacing",
            "hydro_source_lib",
            "gravity_source_lib",
            "boundary_source_lib",
        ]
        for key in string_keys:
            if key == "boundary_conditions":
                state_dict[key] = [bc.encode("utf-8") for bc in state_dict[key]]
            elif key in state_dict:
                state_dict[key] = state_dict[key].encode("utf-8")
        return state_dict

    @staticmethod
    def build(bundle: SimulationBundle) -> SimulationState:
        """Build simulation state from bundle"""
        mesh = bundle.mesh_config
        grid = bundle.grid_config
        io = bundle.io_config
        sim = bundle.sim_config

        # Get bounds with defaults
        x1bounds, x2bounds, x3bounds = SimStateBuilder.get_bounds(
            mesh.bounds, mesh.dimensionality
        )

        bcs = pipe(
            mesh.boundary_conditions,
            lambda bcs: BoundaryManager.validate_conditions(bcs, mesh.dimensionality),
            lambda bcs: BoundaryManager.check_and_fix_curvlinear_conditions(
                conditions=bcs,
                coord_system=mesh.coord_system,
                boundary_source=io.boundary_source_lib,
                dim=mesh.dimensionality,
            ),
        )

        # Build base state dict
        state_dict = {
            "gamma": sim.adiabatic_index,
            "tstart": sim.tstart,
            "tend": sim.tend,
            "cfl": sim.cfl,
            "dlogt": sim.dlogt,
            "plm_theta": sim.plm_theta,
            "checkpoint_interval": io.checkpoint_interval,
            "checkpoint_idx": io.checkpoint_index,
            "data_directory": str(io.data_directory),
            "boundary_conditions": [str(bc) for bc in bcs],
            "spatial_order": sim.spatial_order,
            "temporal_order": sim.time_stepping,
            "x1_spacing": mesh.x1_spacing,
            "x2_spacing": mesh.x2_spacing,
            "x3_spacing": mesh.x3_spacing,
            "solver": sim.solver,
            "regime": sim.regime.encode(),
            "dimensionality": mesh.dimensionality,
            "coord_system": mesh.coord_system,
            "quirk_smoothing": sim.quirk_smoothing,
            "nx": grid.nx,
            "ny": grid.ny,
            "nz": grid.nz,
            "x1bounds": x1bounds,
            "x2bounds": x2bounds,
            "x3bounds": x3bounds,
            "bfield": bundle.staggered_bfields or [[0.0], [0.0], [0.0]],
            "hydro_source_lib": io.hydro_source_lib or "",
            "gravity_source_lib": io.gravity_source_lib or "",
            "boundary_source_lib": io.boundary_source_lib or "",
            "mesh_motion": mesh.mesh_motion,
            "homologous": mesh.is_homologous,
        }

        # Encode strings and return
        return SimStateBuilder.encode_strings(state_dict)
