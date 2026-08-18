# =============================================================================
# tabulated_source_1d.py
#
# a piecewise-linear table prescribes a bounded spatial energy source.
#
# validate:
#  simbi run simbi_configs/examples/newtonian/tabulated_source_1d.py --validate
# expected diagnostic:
#  validation passed
# =============================================================================
from typing import Annotated

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import CoordSystem, Regime
from simbi.types.typing import (
    ExpressionDict,
    GasStateGenerator,
    InitialStateType,
)


class TabulatedSource1D(SimbiProblem):
    resolution: Annotated[
        int,
        ProblemParam(32, cli=True, description="grid resolution"),
    ]
    bounds: list[tuple[float, float]] = [(0.0, 1.0)]
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = 5.0 / 3.0
    end_time: float = 0.05
    checkpoint_interval: float = 0.05

    @property
    def source_expressions(self) -> list[ExpressionDict]:
        (x,) = expr.coords(1)
        heating = expr.tabulated_1d(
            x,
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.02, 0.04, 0.02, 0.0],
            bounds=expr.TableBounds.ZERO,
        )
        return [
            expr.raw([heating], dim=1, target=expr.ConservedField.ENERGY)
        ]

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> GasStateGenerator:
            for _ii in range(self.resolution):
                yield (1.0, 0.0, 1.0)

        return gas_state
