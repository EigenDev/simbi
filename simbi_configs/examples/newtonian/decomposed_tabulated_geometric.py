# =============================================================================
# decomposed_tabulated_geometric.py
#
# a geometric mesh and tabulated source run through the same configuration on
# one device or two decomposed devices.
#
# validate:
#  simbi run simbi_configs/examples/newtonian/decomposed_tabulated_geometric.py --validate
# expected diagnostic:
#  validation passed
# bounded execution:
#  simbi run simbi_configs/examples/newtonian/decomposed_tabulated_geometric.py --end-time 0.001 --gpus 1
#  simbi run simbi_configs/examples/newtonian/decomposed_tabulated_geometric.py --end-time 0.001 --gpus 2 --gpu
# =============================================================================
from typing import Annotated

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import CellSpacing, CoordSystem, Regime
from simbi.types.typing import ExpressionDict, GasStateGenerator, InitialStateType


class DecomposedTabulatedGeometric(SimbiProblem):
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((32, 16), cli=True, description="grid resolution"),
    ]
    gpus: Annotated[
        int,
        ProblemParam(1, ge=1, cli=True, description="decomposition device count"),
    ]
    bounds: list[tuple[float, float]] = [(0.0, 1.0), (0.0, 1.0)]
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = 5.0 / 3.0
    x1_spacing: CellSpacing = CellSpacing.GEOMETRIC
    x1_spacing_ratio: float = 0.97
    end_time: float = 0.05
    checkpoint_interval: float = 0.05

    @property
    def source_expressions(self) -> list[ExpressionDict]:
        graph = expr.ExprGraph()
        x = expr.variable("x1", graph)
        heating = expr.tabulated_1d(
            x,
            [0.0, 0.4, 0.8, 1.0],
            [0.0, 0.01, 0.03, 0.0],
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
