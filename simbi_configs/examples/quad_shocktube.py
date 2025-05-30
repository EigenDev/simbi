from dataclasses import dataclass

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CoordSystem, Regime, CellSpacing
from simbi.core.types.typing import GasStateGenerator, InitialStateType
from typing import Iterator


@dataclass
class ShockTubeState:
    """Dataclass for Shock Tube State"""

    rho: float
    v1: float
    v2: float
    p: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.v1
        yield self.v2
        yield self.p


class SodProblemQuad(SimbiBaseConfig):
    """
    Sod's Shock Tube Problem in 2D Newtonian Fluid with 4 Partitions
    This setup was adapted from Zhang and MacFadyen (2006) section 4.8 pg. 11
    """

    # Required fields from SimbiBaseConfig
    resolution: tuple[int, int] = SimbiField((256, 256), description="Grid resolution")

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
        """Generate initial primitive state for quadrant shock tube.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            ni, nj = self.resolution
            xmin, xmax = self.bounds[0]
            ymin, ymax = self.bounds[1]
            xextent = xmax - xmin
            yextent = ymax - ymin

            dx = xextent / ni
            dy = yextent / nj

            # Define the four quadrant states
            bottom_left = ShockTubeState(
                0.5, 0.0, 0.0, 1.0
            )  # Bottom-left: High density
            top_left = ShockTubeState(
                0.1, 0.9, 0.0, 1.0
            )  # Top-left: Rightward velocity
            bottom_right = ShockTubeState(
                0.1, 0.0, 0.9, 1.0
            )  # Bottom-right: Upward velocity
            top_right = ShockTubeState(0.1, 0.0, 0.0, 0.01)  # Top-right: Low pressure

            for j in range(nj):
                y = ymin + (j + 0.5) * dy  # Cell center
                for i in range(ni):
                    x = xmin + (i + 0.5) * dx  # Cell center

                    # Determine which quadrant the cell is in
                    if x < 0.5 * xextent:
                        if y < 0.5 * yextent:
                            yield tuple(bottom_left)  # Bottom-left
                        else:
                            yield tuple(top_left)  # Top-left
                    else:
                        if y < 0.5 * yextent:
                            yield tuple(bottom_right)  # Bottom-right
                        else:
                            yield tuple(top_right)  # Top-right

        return gas_state
