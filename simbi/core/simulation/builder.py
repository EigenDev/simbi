from dataclasses import dataclass
from ...functional import to_tuple_of_tuples, pipe
from typing import Any, Sequence
from .state_init import SimulationBundle
from ..managers.boundary import BoundaryManager
from pathlib import Path
from ...io.logging import logger


@dataclass
class SimStateBuilder:
    """Builder for simulation state configuration"""

    @staticmethod
    def prepare_data_directory(data_directory: Path) -> None:
        """Ensure data directory exists"""
        if not data_directory.is_dir():
            data_directory.mkdir(parents=True)
            logger.info(f"Created data directory: {data_directory}")

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

    @staticmethod
    def build(bundle: SimulationBundle) -> dict[str, Any]:
        """Build simulation state from bundle"""
        mesh = bundle.mesh_config
        grid = bundle.grid_config
        io = bundle.io_config

        SimStateBuilder.prepare_data_directory(Path(io.data_directory))
        effective_dim = mesh.effective_dim(grid.resolution)
        # Get bounds with defaults
        x1bounds, x2bounds, x3bounds = SimStateBuilder.get_bounds(
            mesh.bounds, effective_dim
        )

        has_boundary_source_terms = (
            io.bx1_outer_expressions is not None or io.bx1_inner_expressions is not None
        )
        bcs = pipe(
            mesh.boundary_conditions,
            lambda bcs: BoundaryManager.validate_conditions(bcs, effective_dim),
            lambda bcs: BoundaryManager.extrapolate_conditions_if_needed(
                bcs, mesh.dimensionality, mesh.coord_system
            ),
            lambda bcs: BoundaryManager.check_and_fix_curvlinear_conditions(
                conditions=bcs,
                coord_system=mesh.coord_system,
                contains_boundary_source_terms=has_boundary_source_terms,
                dim=mesh.dimensionality,
                effective_dim=mesh.effective_dim(grid.resolution),
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

        x = [
            getattr(bundle, s).to_execution_dict()
            for s in ["mesh_config", "sim_config", "grid_config", "io_config"]
        ]
        state_dict = {**x[0], **x[1], **x[2], **x[3], "bfield": effective_bfields}
        state_dict.update(
            boundary_conditions=[bc for bc in bcs],
            x1bounds=x1bounds,
            x2bounds=x2bounds,
            x3bounds=x3bounds,
        )
        state_dict.pop("bounds")

        # Encode strings and return
        return state_dict
