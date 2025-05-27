from typing import Sequence
from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CellSpacing, CoordSystem, Regime
from simbi.core.types.typing import GasStateGenerator, InitialStateType


class SodProblem(SimbiBaseConfig):
    """
    Sod's Shock Tube Problem in 1D Newtonian Fluid
    """

    # Define basic configuration parameters
    adiabatic_index: float = SimbiField(5.0 / 3.0, description="Adiabatic gas index")

    # Required fields that we need to implement
    resolution: int = SimbiField(1000, description="Grid resolution")
    bounds: Sequence[Sequence[float]] = SimbiField(
        [(0.0, 1.0)], description="Domain boundaries"
    )
    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )
    regime: Regime = SimbiField(Regime.CLASSICAL, description="Physics regime")

    # Optional customizations (with defaults from SimbiBaseConfig)
    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for Sod shock tube.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            dx = (self.bounds[0][1] - self.bounds[0][0]) / nx
            for i in range(nx):
                if i * dx < 0.5:
                    yield (1.0, 0.0, 1.0)
                else:
                    yield (0.125, 0.0, 0.1)

        return gas_state
