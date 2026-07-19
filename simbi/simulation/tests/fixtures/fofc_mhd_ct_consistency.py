# =============================================================================
# fofc_mhd_ct_consistency.py
#
# periodic 2d colliding magnetized ultra-relativistic streams (v = +0.999 / -0.999,
# p = 1e-6, uniform transverse By = 0.23), run SHORT. the magnetized collision drives
# the high-order rmhd c2p unphysical on a CORRECTOR substage, firing the first-order
# flux-correction fallback. this is the C2 constrained-transport instrument: after the
# FOFC redo the CELL B must be consistent with the staggered FACE field,
# bcell == interp(bface). WITHOUT the §3'' CT re-run the redo re-advances cell B from the
# first-order induction flux and never re-interpolates it, so bcell diverges from
# interp(bface) (measured 1.3e-2 at this b0 on the reverted code). WITH the fix (splice the
# induction flux + the edge EMF, re-curl the pre-curl face field, then bcell_from_bface) the
# two agree to roundoff.
#
# by = 0.23 is chosen in the narrow discriminating window [0.22, 0.24]: below ~0.22 the
# collision fires FOFC only on the PREDICTOR (no bcell_from_bface, so bcell stays consistent
# and the gate is vacuous); at >= ~0.25 the field is strong enough that the base-scheme
# magnetic-energy patch (a SEPARATE, pre-existing non-conservation, patch ~ B^2) shocks the
# c2p into a persistent-freeze halt — out of scope for the C2 fix. run SHORT (a few steps,
# below the 16-substage freeze halt); deterministic at these times.
# =============================================================================

from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType, StaggeredBFieldGenerator


class FofcMhdCtConsistency(SimbiProblem):
    """periodic 2d magnetized colliding streams (C2 CT-consistency instrument)."""

    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0, description="adiabatic index")]
    b0: Annotated[float, ProblemParam(0.23, cli=True, description="transverse By")]
    resolution: Annotated[
        tuple[int, int, int], ProblemParam((128, 8, 1), cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLD, description="numerical solver")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.PERIODIC], description="boundary conditions"),
    ]
    x1_spacing: Annotated[
        CellSpacing, ProblemParam(CellSpacing.LINEAR, description="grid spacing in x1 direction")
    ]
    end_time: Annotated[
        float, ProblemParam(0.03, cli=True, checkpoint_safe=True, description="simulation end time")
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """magnetized ultra-relativistic streams colliding at x = 0.5 and the wrap."""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            xb = self.bounds[0]
            dx = (xb[1] - xb[0]) / ni
            for _kk in range(nk):
                for _jj in range(nj):
                    for ii in range(ni):
                        xi = xb[0] + (ii + 0.5) * dx
                        v = 0.999 if xi <= 0.5 * (xb[1] - xb[0]) else -0.999
                        yield (1.0, v, 0.0, 0.0, 1e-6)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj, nk = self.resolution
            for _kk in range(nk + (bn == "bz")):
                for _jj in range(nj + (bn == "by")):
                    for _ii in range(ni + (bn == "bx")):
                        yield self.b0 if bn == "by" else 0.0

        return (gas_state, partial(b_field, "bx"), partial(b_field, "by"), partial(b_field, "bz"))
