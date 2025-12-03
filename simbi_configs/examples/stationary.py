# =============================================================================
# stationary.py
#
# stationary wave test problems in 1d newtonian fluid.
# tests contact discontinuity preservation.
# =============================================================================
from pathlib import Path
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import CellSpacing, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType


class StationaryWaveHLL(SimbiProblem):
    """stationary wave test using hll solver."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]

    # domain
    resolution: Annotated[
        int, ProblemParam(400, cli=True, description="grid resolution")
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
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(
            CellSpacing.LINEAR, description="grid spacing in x1 direction"
        ),
    ]

    # numerics
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLE, description="numerical solver")
    ]

    # simulation control
    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/stationary/hlle"),
            cli=True,
            checkpoint_safe=True,
            description="output data directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            1.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for stationary wave."""

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / nx

            for ii in range(nx):
                x = xmin + (ii + 0.5) * dx

                if x < 0.5 * xextent:
                    yield (1.4, 0.0, 1.0)
                else:
                    yield (1.0, 0.0, 1.0)

        return gas_state


class StationaryWaveHLLC(StationaryWaveHLL):
    """stationary wave test using hllc solver."""

    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="hllc numerical solver")
    ]
    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/stationary/hllc"),
            cli=True,
            checkpoint_safe=True,
            description="output data directory for hllc solver",
        ),
    ]
