# =============================================================================
# fofc_periodic_blast.py
#
# periodic 1d colliding ultra-relativistic cold streams (v = +0.999 / -0.99,
# p = 1e-8): the collision shocks at x = 0.5 and at the periodic wrap drive the
# high-order c2p unphysical, so the first-order flux-correction fallback fires
# (a marti & muller pressure jump alone does NOT fire it at this resolution —
# the probe finds no unphysical zone and the pass early-returns). the slight
# velocity asymmetry keeps the total momentum drift from cancelling by symmetry.
# periodicity makes the total conserved state (sum of D, S, tau over the
# interior) an exact invariant of any face-telescoping finite-volume update —
# the instrument for conservation gates on the fallback.
# =============================================================================

from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class FofcPeriodicBlast(SimbiProblem):
    """periodic marti & muller double shock tube (fofc conservation instrument)."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index")
    ]
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
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    boundary_conditions: Annotated[
        BoundaryCondition,
        ProblemParam(BoundaryCondition.PERIODIC, description="boundary conditions"),
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="grid spacing in x1 direction"),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            0.1, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """cold ultra-relativistic streams colliding at x = 0.5 and the wrap."""

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            xmin, xmax = self.bounds[0]
            dx = (xmax - xmin) / nx
            for ii in range(nx):
                xi = xmin + (ii + 0.5) * dx
                v = 0.999 if xi <= 0.5 * (xmax - xmin) else -0.99
                yield (1.0, v, 1e-8)

        return gas_state
