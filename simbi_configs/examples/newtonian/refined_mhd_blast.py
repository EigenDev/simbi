# =============================================================================
# refined_mhd_blast.py
#
# the magnetized blast (Gardiner & Stone 2005, section 8.7) on a statically
# refined 2.5D mesh: a central overpressure in a uniform oblique field, with one
# fine box of ratio two around it and outflow walls. the in-plane field is
# uniform on the staggered faces, so div B starts at machine zero; the fine
# faces are seeded by divergence-free prolongation of the coarse faces. a
# uniform out-of-plane field b_z rides with the gas, so the out-of-plane cell
# field crosses the coarse-fine faces as a refluxed component.
#
# usage:
#  simbi run refined_mhd_blast.py
#  simbi launch refined_mhd_blast.py --layout layout.toml
# =============================================================================
import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)

BOUND = 0.5


class RefinedMhdBlast(SimbiProblem):
    """magnetized blast on a coarse plane with one refined central box."""

    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    p_blast: Annotated[
        float, ProblemParam(10.0, description="central overpressure")
    ]
    p_ambient: Annotated[
        float, ProblemParam(0.1, description="ambient pressure")
    ]
    r_blast: Annotated[float, ProblemParam(0.1, description="blast radius")]
    b0: Annotated[
        float,
        ProblemParam(
            1.0, description="in-plane field strength |B|, at 45 degrees"
        ),
    ]
    b_z: Annotated[
        float, ProblemParam(0.1, description="uniform out-of-plane field")
    ]

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
        Regime, ProblemParam(Regime.NMHD, description="physics regime")
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLD, cli=True, description="numerical solver")
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
        """central overpressure at rest; primitive is (rho, vx, vy, vz, p)."""

        def gas_state() -> GasStateGenerator:
            nx, ny, _ = self.resolution
            (xmin, xmax), (ymin, ymax) = self.bounds[0], self.bounds[1]
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            for jj in range(ny):
                y = ymin + (jj + 0.5) * dy
                for ii in range(nx):
                    x = xmin + (ii + 0.5) * dx
                    r = math.hypot(x, y)
                    pre = self.p_blast if r <= self.r_blast else self.p_ambient
                    yield (1.0, 0.0, 0.0, 0.0, pre)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            nx, ny, _ = self.resolution
            value = {"bx": self.b0 / math.sqrt(2.0), "by": self.b0 / math.sqrt(2.0), "bz": self.b_z}[bn]
            for _jj in range(ny + (bn == "by")):
                for _ii in range(nx + (bn == "bx")):
                    yield value

        return (
            gas_state,
            partial(b_field, "bx"),
            partial(b_field, "by"),
            partial(b_field, "bz"),
        )
