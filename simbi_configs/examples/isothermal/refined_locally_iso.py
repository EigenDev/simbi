# =============================================================================
# refined_locally_iso.py
#
# a locally isothermal 2d run on a refined mesh. rho is uniform and the gas is at
# rest, but the frozen temperature cs^2(x) varies — so the pressure p = cs^2(x)*rho
# has a gradient that drives motion. the test of the AMR path: the per-cell cs^2(x)
# (which lives in the iso kernel-set) must be prolongated to the
# fine level, or the fine box would fall back to a uniform cs^2 and not move.
# =============================================================================
import math
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class RefinedLocallyIso(SimbiProblem):
    """locally-isothermal 2d with a sinusoidal cs^2(x), one refined box."""

    adiabatic_index: Annotated[
        float, ProblemParam(1.0, description="unused (isothermal)")
    ]
    sound_speed: Annotated[
        float, ProblemParam(1.0, description="reference cs (cs^2(x) is derived)")
    ]

    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((64, 64, 1), cli=True, description="coarse (nx,ny,1)"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="domain bounds"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.PERIODIC], description="boundary conditions"
        ),
    ]

    refinement_enabled: Annotated[
        bool, ProblemParam(True, description="enable mesh refinement")
    ]
    refinement_max_levels: Annotated[
        int, ProblemParam(2, description="coarse + 1 fine")
    ]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam(
            [[0.25, 0.75, 0.25, 0.75]], description="central fine box"
        ),
    ]
    refinement_ratios: Annotated[
        list[int], ProblemParam([2], description="coarse->fine ratio")
    ]

    end_time: Annotated[
        float,
        ProblemParam(0.05, cli=True, checkpoint_safe=True, description="end"),
    ]

    @computed_field
    @property
    def locally_isothermal(self) -> bool:
        return True

    @computed_field
    @property
    def ambient_sound_speed(self) -> float:
        return self.sound_speed

    def initial_primitive_state(self) -> InitialStateType:
        """uniform density at rest; varying initial pressure p = cs^2(x)*rho."""

        def cs2_of(x: float) -> float:
            return 0.5 + 0.3 * math.sin(2.0 * math.pi * x)

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            (x0, x1), (y0, y1) = self.bounds[0], self.bounds[1]
            dx = (x1 - x0) / ni
            dy = (y1 - y0) / nj
            rho = 1.0
            for _kk in range(nk):
                for _jj in range(nj):
                    for ii in range(ni):
                        x = x0 + (ii + 0.5) * dx
                        # locally isothermal: (rho, vx, vy, p_local)
                        yield (rho, 0.0, 0.0, cs2_of(x) * rho)

        return gas_state
