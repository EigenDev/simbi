# =============================================================================
# marti_muller.py
#
# marti & muller (2003), relativistic shock tube problem on 1d mesh.
# =============================================================================

from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import CellSpacing, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class MartiMuller(SimbiProblem):
    """marti & muller (2003), relativistic shock tube problem on 1d mesh."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index")
    ]

    # domain
    resolution: Annotated[
        int, ProblemParam(1000, cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.SRHD, description="physics regime")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(
            CellSpacing.LINEAR, description="grid spacing in x1 direction"
        ),
    ]

    # simulation control
    end_time: Annotated[
        float,
        ProblemParam(
            0.4,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for marti & muller shock tube."""

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / nx

            for ii in range(nx):
                xi = xmin + (ii + 0.5) * dx
                if xi <= 0.5 * xextent:
                    yield (10.0, 0.0, 13.33)
                else:
                    yield (1.0, 0.0, 1e-10)

        return gas_state
