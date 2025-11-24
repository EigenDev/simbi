# =============================================================================
# quad_shocktube.py
#
# sod's shock tube problem in 2d with 4 partitions.
# adapted from zhang and macfadyen (2006) section 4.8.
# =============================================================================
from dataclasses import dataclass
from typing import Iterator

from simbi import ProblemParam, SimbiProblem
from simbi.types import CellSpacing, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


@dataclass
class ShockTubeState:
    """dataclass for shock tube state."""

    rho: float
    v1: float
    v2: float
    p: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.v1
        yield self.v2
        yield self.p


class SodProblemQuad(SimbiProblem):
    """sod's shock tube in 2d with 4 partitions, zhang & macfadyen (2006) 4.8."""

    # physics
    adiabatic_index: float = ProblemParam(
        5.0 / 3.0, description="adiabatic index"
    )

    # domain
    resolution: tuple[int, int] = ProblemParam(
        (256, 256), cli=True, description="grid resolution"
    )
    bounds: list[tuple[float, float]] = ProblemParam(
        [(0.0, 1.0), (0.0, 1.0)], description="domain boundaries"
    )
    coord_system: CoordSystem = ProblemParam(
        CoordSystem.CARTESIAN, description="coordinate system"
    )
    regime: Regime = ProblemParam(Regime.SRHD, description="physics regime")
    x1_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="grid spacing in x1 direction"
    )

    # simulation control
    end_time: float = ProblemParam(
        0.4, cli=True, checkpoint_safe=True, description="simulation end time"
    )

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for quadrant shock tube."""

        def gas_state() -> GasStateGenerator:
            ni, nj = self.resolution
            xmin, xmax = self.bounds[0]
            ymin, ymax = self.bounds[1]
            xextent = xmax - xmin
            yextent = ymax - ymin

            dx = xextent / ni
            dy = yextent / nj

            bottom_left = ShockTubeState(0.5, 0.0, 0.0, 1.0)
            top_left = ShockTubeState(0.1, 0.9, 0.0, 1.0)
            bottom_right = ShockTubeState(0.1, 0.0, 0.9, 1.0)
            top_right = ShockTubeState(0.1, 0.0, 0.0, 0.01)

            for jj in range(nj):
                y = ymin + (jj + 0.5) * dy
                for ii in range(ni):
                    x = xmin + (ii + 0.5) * dx

                    if x < 0.5 * xextent:
                        if y < 0.5 * yextent:
                            yield tuple(bottom_left)
                        else:
                            yield tuple(top_left)
                    else:
                        if y < 0.5 * yextent:
                            yield tuple(bottom_right)
                        else:
                            yield tuple(top_right)

        return gas_state
