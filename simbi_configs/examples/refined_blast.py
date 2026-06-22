# =============================================================================
# refined_blast.py
#
# a 2d Sedov-like blast on a STATICALLY REFINED mesh: a coarse base grid with one
# fine box around the central overpressure region. exercises the AMR path end to
# end — the hierarchy is checkpointed as level_0 (coarse) + level_1 (fine) in one
# file. the fine interior is seeded by prolongation from the coarse IC.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType

BOUND = 0.5


class RefinedBlast(SimbiProblem):
    """central blast on a coarse grid with one refined central box."""

    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    p_blast: Annotated[
        float, ProblemParam(10.0, description="central overpressure")
    ]
    r_blast: Annotated[
        float, ProblemParam(0.1, description="blast radius")
    ]

    # coarse base grid
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((64, 64, 1), cli=True, description="coarse (nx, ny, 1)"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(-BOUND, BOUND), (-BOUND, BOUND)], description="domain bounds"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.OUTFLOW], description="boundary conditions"
        ),
    ]

    # ---- static mesh refinement: one fine box covering the blast ----
    refinement_enabled: Annotated[
        bool, ProblemParam(True, description="enable mesh refinement")
    ]
    refinement_max_levels: Annotated[
        int, ProblemParam(2, description="coarse + 1 fine")
    ]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam(
            [[-0.25, 0.25, -0.25, 0.25]],
            description="one fine box: x in [-0.25,0.25], y in [-0.25,0.25]",
        ),
    ]
    refinement_ratios: Annotated[
        list[int],
        ProblemParam([2], description="coarse->fine refinement ratio per jump"),
    ]

    end_time: Annotated[
        float,
        ProblemParam(0.1, cli=True, checkpoint_safe=True, description="end time"),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """central overpressure, at rest; primitive is (rho, vx, vy, p)."""

        def gas_state() -> GasStateGenerator:
            nx, ny, _ = self.resolution
            (xmin, xmax), (ymin, ymax) = self.bounds[0], self.bounds[1]
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            p_amb = 0.1
            for _kk in range(1):
                for jj in range(ny):
                    y = ymin + (jj + 0.5) * dy
                    for ii in range(nx):
                        x = xmin + (ii + 0.5) * dx
                        r = (x * x + y * y) ** 0.5
                        pre = self.p_blast if r <= self.r_blast else p_amb
                        yield (1.0, 0.0, 0.0, pre)

        return gas_state
