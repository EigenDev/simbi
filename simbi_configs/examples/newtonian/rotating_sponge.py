# =============================================================================
# rotating_sponge.py
#
# a rotating-frame force and full-state outer sponge compose as independent
# source entries on a small cartesian flow.
#
# validate:
#  simbi run simbi_configs/examples/newtonian/rotating_sponge.py --validate
# expected diagnostic:
#  validation passed
# =============================================================================
from typing import Annotated

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import CoordSystem, Regime
from simbi.types.typing import ExpressionDict, GasStateGenerator, InitialStateType


class RotatingSponge(SimbiProblem):
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((24, 24), cli=True, description="grid resolution"),
    ]
    bounds: list[tuple[float, float]] = [(-1.0, 1.0), (-1.0, 1.0)]
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = 5.0 / 3.0
    end_time: float = 0.05
    checkpoint_interval: float = 0.05

    @property
    def source_expressions(self) -> list[ExpressionDict]:
        graph = expr.ExprGraph()
        rotating = graph.compile(
            [
                expr.constant(0.25, graph),
                expr.constant(0.0, graph),
                expr.constant(0.0, graph),
            ]
        ).serialize_source(expr.SourceKind.ROTATING_FRAME, dim=2)

        graph = expr.ExprGraph()
        x = expr.variable("x1", graph)
        y = expr.variable("x2", graph)
        radius_sq = x * x + y * y
        kappa = expr.where(
            radius_sq > 0.64,
            expr.constant(2.0, graph),
            expr.constant(0.0, graph),
        )
        sponge = graph.compile(
            [
                kappa,
                expr.constant(1.0, graph),
                expr.constant(0.0, graph),
                expr.constant(0.0, graph),
                expr.constant(1.5, graph),
            ]
        ).serialize_source(
            expr.SourceKind.SPONGE,
            dim=2,
            params=[1.0 / (self.adiabatic_index - 1.0)],
        )
        return [rotating, sponge]

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> GasStateGenerator:
            for _jj in range(self.resolution[1]):
                for _ii in range(self.resolution[0]):
                    yield (1.0, 0.0, 0.0, 1.0)

        return gas_state
