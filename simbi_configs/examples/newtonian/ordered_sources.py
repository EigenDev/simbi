# =============================================================================
# ordered_sources.py
#
# three independent source payloads deposit density, momentum, and energy into
# a small two-dimensional newtonian flow.
#
# validate:
#  simbi run simbi_configs/examples/newtonian/ordered_sources.py --validate
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


class OrderedSources(SimbiProblem):
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((24, 16), cli=True, description="grid resolution"),
    ]
    bounds: list[tuple[float, float]] = [(0.0, 1.0), (0.0, 1.0)]
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = 5.0 / 3.0
    end_time: float = 0.05
    checkpoint_interval: float = 0.05

    @property
    def source_expressions(self) -> list[ExpressionDict]:
        # each slot is its own graph, since a raw source reaches one conserved
        # slot at a time. the blob is the same disc in every one of them.
        def blob() -> tuple:
            x, y = expr.coords(2)
            return x, y, ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) < 0.04

        _, _, region = blob()
        density = expr.raw(
            [0.02 * region], dim=2, target=expr.ConservedField.DENSITY
        )

        _, _, region = blob()
        momentum = expr.raw(
            [0.01 * region, 0.0 * region],
            dim=2,
            target=expr.ConservedField.MOMENTUM,
        )

        _, _, region = blob()
        energy = expr.raw(
            [0.03 * region], dim=2, target=expr.ConservedField.ENERGY
        )
        return [density, momentum, energy]

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> GasStateGenerator:
            for _jj in range(self.resolution[1]):
                for _ii in range(self.resolution[0]):
                    yield (1.0, 0.0, 0.0, 1.0)

        return gas_state
