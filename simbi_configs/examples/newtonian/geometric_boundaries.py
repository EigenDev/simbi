# =============================================================================
# geometric_boundaries.py
#
# a geometric x mesh can concentrate cells at either boundary by changing the
# adjacent-width ratio.
#
# validate:
#  simbi run simbi_configs/examples/newtonian/geometric_boundaries.py --validate
# expected diagnostic:
#  validation passed
# =============================================================================
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import CellSpacing, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class GeometricBoundaries(SimbiProblem):
    resolution: Annotated[
        int,
        ProblemParam(32, cli=True, description="grid resolution"),
    ]
    cluster_upper: Annotated[
        bool,
        ProblemParam(
            False,
            cli=True,
            description="concentrate cells at the upper rather than lower boundary",
        ),
    ]
    bounds: list[tuple[float, float]] = [(0.0, 1.0)]
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = 5.0 / 3.0
    x1_spacing: CellSpacing = CellSpacing.GEOMETRIC
    x1_spacing_ratio: float = 1.04
    end_time: float = 0.05
    checkpoint_interval: float = 0.05

    def setup(self) -> None:
        super().setup()
        self.x1_spacing_ratio = 1.0 / 1.04 if self.cluster_upper else 1.04

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> GasStateGenerator:
            for _ii in range(self.resolution):
                yield (1.0, 0.0, 1.0)

        return gas_state
