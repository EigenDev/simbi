# =============================================================================
# tabulated_source_2d.py
#
# a bilinear table prescribes a bounded two-dimensional energy source.
#
# validate:
#  simbi run simbi_configs/examples/newtonian/tabulated_source_2d.py --validate
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


class TabulatedSource2D(SimbiProblem):
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((24, 20), cli=True, description="grid resolution"),
    ]
    bounds: list[tuple[float, float]] = [(0.0, 1.0), (0.0, 1.0)]
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = 5.0 / 3.0
    end_time: float = 0.05
    checkpoint_interval: float = 0.05

    @property
    def source_expressions(self) -> list[ExpressionDict]:
        graph = expr.ExprGraph()
        x = expr.variable("x1", graph)
        y = expr.variable("x2", graph)
        heating = expr.tabulated_2d(
            x,
            y,
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [
                [0.0, 0.01, 0.0],
                [0.01, 0.04, 0.01],
                [0.0, 0.01, 0.0],
            ],
            bounds=expr.TableBounds.ZERO,
        )
        return [
            graph.compile([heating]).serialize_source(
                expr.SourceKind.RAW,
                dim=2,
                target=expr.ConservedField.ENERGY,
            )
        ]

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> GasStateGenerator:
            for _jj in range(self.resolution[1]):
                for _ii in range(self.resolution[0]):
                    yield (1.0, 0.0, 0.0, 1.0)

        return gas_state
