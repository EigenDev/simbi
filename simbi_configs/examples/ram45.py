from dataclasses import dataclass
from typing import Iterator

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import BoundaryCondition, CoordSystem, Regime, CellSpacing
from simbi.core.types.typing import GasStateGenerator, InitialStateType


@dataclass(frozen=True)
class ShockHeatingState:
    """State for shock heating problem"""

    rho: float
    v1: float
    p: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.v1
        yield self.p


class Ram45(SimbiBaseConfig):
    """
    1D shock-heating problem in planar geometry
    This setup was adapted from Zhang and MacFadyen (2006) section 4.5
    """

    # Required fields from SimbiBaseConfig
    resolution: int = SimbiField(100, description="Grid resolution")

    bounds: list[tuple[float, float]] = SimbiField(
        [(0.0, 1.0)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.SRHD, description="Physics regime")

    adiabatic_index: float = SimbiField(4.0 / 3.0, description="Adiabatic index")

    # Optional customizations
    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    boundary_conditions: list[BoundaryCondition] = SimbiField(
        [BoundaryCondition.OUTFLOW, BoundaryCondition.REFLECTING],
        description="Boundary conditions",
    )

    end_time: float = SimbiField(2.0, description="Simulation end time")

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for shock heating problem.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            xmin, xmax = self.bounds[0]
            dx = (xmax - xmin) / nx

            # Initialize with a uniform state - just below light speed velocity
            state = ShockHeatingState(1.0, (1.0 - 1.0e-8), 1e-6)

            for i in range(nx):
                x = xmin + (i + 0.5) * dx  # Cell center
                yield tuple(state)

        return gas_state
