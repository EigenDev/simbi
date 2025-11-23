from typing import Sequence

from simbi.core.types.input import (
    BoundaryCondition,  # Add this import
    CellSpacing,
    CoordSystem,
    Regime,
)
from simbi.core.types.typing import GasStateGenerator, InitialStateType
from simbi.simulation.base import BaseProblemConfig
from simbi.simulation.fields import ProblemField
from simbi.simulation.state import SimulationStateSpec


class SodProblem(BaseProblemConfig):
    """Sod's Shock Tube Problem in 1D Newtonian Fluid

    A classic test problem that involves a discontinuous initial condition
    with two constant states separated by a membrane that is removed at t=0.
    """

    # Core physics parameters
    adiabatic_index: float = ProblemField(
        5.0 / 3.0,
        description="Adiabatic gas index (gamma)",
        help_text="Ratio of specific heats (default: 5/3 for monatomic gas)",
    )

    # Domain configuration
    resolution: int = ProblemField(
        1000,
        description="Grid resolution",
        help_text="Number of cells in x-direction",
    )

    bounds: Sequence[Sequence[float]] = ProblemField(
        [(0.0, 1.0)],
        description="Domain boundaries",
        help_text="List of (min, max) for each dimension",
    )

    coord_system: CoordSystem = ProblemField(
        CoordSystem.CARTESIAN,
        description="Coordinate system",
        help_text="geometry of the problem",
    )

    regime: Regime = ProblemField(
        Regime.NEWTONIAN,
        description="Physics regime",
        help_text="Classical/Relativistic/etc",
    )

    x1_spacing: CellSpacing = ProblemField(
        CellSpacing.LINEAR,
        description="Grid spacing type",
        help_text="Cell distribution along x1 axis",
    )

    end_time: float = ProblemField(
        0.1,
        description="End time of the simulation",
        help_text="Time at which the simulation ends",
    )

    checkpoint_interval: float = ProblemField(
        0.1,
        description="Checkpoint interval",
        help_text="Time interval between checkpoints",
    )

    def build_state(self) -> SimulationStateSpec:
        """Build complete simulation state.

        For Sod shock tube:
        - 1D Cartesian grid
        - Classical hydrodynamics
        - Outflow boundaries
        """
        # Get base fields from parent class
        base_fields = self._get_state_fields_from_config()

        # Build complete state
        return SimulationStateSpec(
            **base_fields,
            # Grid configuration
            resolution=(self.resolution,),  # 1D problem
            bounds=self.bounds,
            coord_system=self.coord_system,
            # Physics configuration
            regime=self.regime,
            adiabatic_index=self.adiabatic_index,
            # Boundary conditions - outflow on both ends
            boundary_conditions=[
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.OUTFLOW,
            ],
            # Grid spacing
            x1_spacing=self.x1_spacing,
            # Required by StateInterface
            source_config=self,
            end_time=self.end_time,
            checkpoint_interval=self.checkpoint_interval,
        )

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for Sod shock tube.

        Left state (x < 0.5):  (ρ, v, P) = (1.0, 0.0, 1.0)
        Right state (x ≥ 0.5): (ρ, v, P) = (0.125, 0.0, 0.1)

        Returns:
            Generator yielding (density, velocity, pressure) tuples
        """

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            dx = (self.bounds[0][1] - self.bounds[0][0]) / nx

            for i in range(nx):
                x = i * dx
                if x < 0.5:
                    # Left state: (ρ, v, P) = (1.0, 0.0, 1.0)
                    yield (1.0, 0.0, 1.0)
                else:
                    # Right state: (ρ, v, P) = (0.125, 0.0, 0.1)
                    yield (0.125, 0.0, 0.1)

        return gas_state
