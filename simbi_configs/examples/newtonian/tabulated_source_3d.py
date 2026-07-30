# =============================================================================
# tabulated_source_3d.py
#
# a trilinear table prescribes a bounded three-dimensional energy source.
#
# validate:
#  simbi run simbi_configs/examples/newtonian/tabulated_source_3d.py --validate
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


class TabulatedSource3D(SimbiProblem):
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((12, 10, 8), cli=True, description="grid resolution"),
    ]
    bounds: list[tuple[float, float]] = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]
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
        z = expr.variable("x3", graph)
        samples = [0.0, 0.5, 1.0]
        heating = expr.tabulated_3d(
            x,
            y,
            z,
            samples,
            samples,
            samples,
            [
                [
                    [
                        0.04
                        * max(0.0, 1.0 - 2.0 * abs(x_value - 0.5))
                        * max(0.0, 1.0 - 2.0 * abs(y_value - 0.5))
                        * max(0.0, 1.0 - 2.0 * abs(z_value - 0.5))
                        for x_value in samples
                    ]
                    for y_value in samples
                ]
                for z_value in samples
            ],
            bounds=expr.TableBounds.ZERO,
        )
        return [
            graph.compile([heating]).serialize_source(
                expr.SourceKind.RAW,
                dim=3,
                target=expr.ConservedField.ENERGY,
            )
        ]

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> GasStateGenerator:
            for _kk in range(self.resolution[2]):
                for _jj in range(self.resolution[1]):
                    for _ii in range(self.resolution[0]):
                        yield (1.0, 0.0, 0.0, 0.0, 1.0)

        return gas_state
