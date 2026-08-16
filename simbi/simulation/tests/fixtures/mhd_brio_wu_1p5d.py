# =============================================================================
# mhd_brio_wu_1p5d.py
#
# the canonical Brio & Wu (1988) MHD shock tube on a genuine 1D grid (D=1, DOF=3) —
# the 1.5D newtonian-MHD instrument. 1.5D has no constrained transport (C(1,2)=0 edges),
# so the normal field Bx is carried on its (thin) face and never curled (it must stay at
# its constant IC), while the transverse By,Bz are cell-centered conserved variables
# evolved by the out-of-plane cell-B flux predictor. this exercises the out-of-plane
# predictor end-to-end through the python runner.
#
# IC (gamma=2, x0=0.5): left rho=1, p=1, B=(0.75, 1, 0); right rho=0.125, p=0.1,
# B=(0.75, -1, 0); v=0 both sides. the characteristic compound wave reverses By, so it
# changes sign in the interior — the discriminating signature the out-of-plane predictor
# must reproduce (a frozen By keeps the sharp +-1 IC jump and drives p < 0).
# =============================================================================

from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType, StaggeredBFieldGenerator


class MhdBrioWu1p5d(SimbiProblem):
    """1.5D newtonian-MHD Brio-Wu shock tube (out-of-plane By,Bz predictor instrument)."""

    adiabatic_index: Annotated[float, ProblemParam(2.0, description="adiabatic index")]
    bx: Annotated[float, ProblemParam(0.75, cli=True, description="constant normal field Bx")]
    resolution: Annotated[
        tuple[int, int, int], ProblemParam((400, 1, 1), cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(0.0, 1.0)], description="domain boundaries")
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.NMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, description="numerical solver")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.OUTFLOW], description="boundary conditions"),
    ]
    x1_spacing: Annotated[
        CellSpacing, ProblemParam(CellSpacing.LINEAR, description="grid spacing in x1 direction")
    ]
    end_time: Annotated[
        float, ProblemParam(0.1, cli=True, checkpoint_safe=True, description="simulation end time")
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """left/right Brio-Wu states; Bx on the x-faces (const), By,Bz cell-centered."""
        ni = self.resolution[0]
        xb = self.bounds[0]
        dx = (xb[1] - xb[0]) / ni
        x_mid = 0.5 * (xb[0] + xb[1])

        def gas_state() -> GasStateGenerator:
            for ii in range(ni):
                xi = xb[0] + (ii + 0.5) * dx
                if xi < x_mid:
                    yield (1.0, 0.0, 0.0, 0.0, 1.0)
                else:
                    yield (0.125, 0.0, 0.0, 0.0, 0.1)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            # in-grid axis 0 (bx) seeds the x-faces: ni+1 values, constant. the transverse
            # axes 1,2 (by,bz) seed cell-centered B: ni values each.
            if bn == "bx":
                for _ii in range(ni + 1):
                    yield self.bx
            else:
                for ii in range(ni):
                    xi = xb[0] + (ii + 0.5) * dx
                    if bn == "by":
                        yield 1.0 if xi < x_mid else -1.0
                    else:  # bz
                        yield 0.0

        return (gas_state, partial(b_field, "bx"), partial(b_field, "by"), partial(b_field, "bz"))
