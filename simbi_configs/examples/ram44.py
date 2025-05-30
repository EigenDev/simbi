from dataclasses import dataclass
from typing import Iterator

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CoordSystem, Regime, CellSpacing
from simbi.core.types.typing import GasStateGenerator, InitialStateType


@dataclass
class SplitState:
    """State for split shock problem"""

    rho: float
    v1: float
    v2: float
    pre: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.v1
        yield self.v2
        yield self.pre

    def __len__(self) -> int:
        return 4


class Ram44(SimbiBaseConfig):
    """
    Shock with non-zero transverse velocity on one side in 2D with 1 Partition
    This setup was adapted from Zhang and MacFadyen (2006) section 4.4
    """

    # Required fields from SimbiBaseConfig
    resolution: tuple[int, int] = SimbiField((400, 400), description="Grid resolution")

    bounds: list[tuple[float, float]] = SimbiField(
        [(0.0, 1.0), (0.0, 1.0)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.SRHD, description="Physics regime")

    adiabatic_index: float = SimbiField(5.0 / 3.0, description="Adiabatic index")

    # Optional customizations
    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    end_time: float = SimbiField(0.4, description="Simulation end time")

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for RAM44 shock.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / nx

            # Define left and right states
            left_state = SplitState(1.0, 0.0, 0.0, 1e3)  # High pressure, no velocity
            right_state = SplitState(
                1.0, 0.0, 0.99, 1e-2
            )  # Low pressure, high transverse velocity

            for j in range(ny):
                for i in range(nx):
                    x = xmin + (i + 0.5) * dx  # Cell center

                    if x < 0.5 * xextent:
                        yield tuple(left_state)  # Left state
                    else:
                        yield tuple(right_state)  # Right state

        return gas_state
