# =============================================================================
# mhd_bz_advection_2p5d.py
#
# a force-balanced OUT-OF-PLANE field advection on a 2.5D cartesian grid (D=2, DOF=3):
# the transverse Bz has no staggered face (only the in-plane Bx,By are CT), so it is a
# cell-centered conserved variable evolved SOLELY by the out-of-plane cell-B flux predictor.
#
# the state is a rigid advecting equilibrium: uniform rho and velocity, Bx=By=0, and a
# smooth Bz(x,y) whose magnetic pressure is exactly cancelled by the gas pressure
# (p_gas = P0 - Bz^2/2), so the TOTAL pressure is uniform and there is no force. with no
# magnetic tension (Bz-only field, d/dz=0), Bz advects rigidly at the flow velocity:
#   Bz(x, y, t) = Bz(x - vx t, y - vy t, 0)
# a working out-of-plane predictor reproduces this translation; a frozen predictor leaves
# Bz at its IC (the regression). the domain is periodic; at a HALF period the exact solution
# is the IC shifted by half the domain in each direction — far from the IC, so it separates a
# genuinely-advected Bz from a frozen one (a full period would return to the IC either way).
# =============================================================================

import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType, StaggeredBFieldGenerator

_P0 = 3.0  # uniform total pressure floor (keeps p_gas = P0 - Bz^2/2 well positive)
_VX = 1.0
_VY = 1.0


def _bz(x: float, y: float) -> float:
    """smooth out-of-plane field, varying in BOTH grid directions (exercises both fluxes)."""
    return 1.0 + 0.3 * math.sin(2.0 * math.pi * x) + 0.2 * math.cos(2.0 * math.pi * y)


class MhdBzAdvection2p5d(SimbiProblem):
    """2.5D cartesian out-of-plane Bz advection (force-balanced; predictor instrument)."""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="adiabatic index")]
    resolution: Annotated[
        tuple[int, int, int], ProblemParam((64, 64, 1), cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.NMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLD, description="numerical solver")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.PERIODIC], description="boundary conditions"),
    ]
    x1_spacing: Annotated[
        CellSpacing, ProblemParam(CellSpacing.LINEAR, description="grid spacing in x1 direction")
    ]
    end_time: Annotated[
        float, ProblemParam(0.5, cli=True, checkpoint_safe=True, description="half advection period (v=1, L=1)")
    ]

    def initial_primitive_state(self) -> InitialStateType:
        ni, nj, _nk = self.resolution
        xb, yb = self.bounds[0], self.bounds[1]
        dx = (xb[1] - xb[0]) / ni
        dy = (yb[1] - yb[0]) / nj

        def gas_state() -> GasStateGenerator:
            for jj in range(nj):
                yc = yb[0] + (jj + 0.5) * dy
                for ii in range(ni):
                    xc = xb[0] + (ii + 0.5) * dx
                    pre = _P0 - 0.5 * _bz(xc, yc) ** 2
                    yield (1.0, _VX, _VY, 0.0, pre)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            # in-grid axes 0,1 (bx,by) seed the CT faces = 0; the transverse axis 2 (bz) seeds
            # cell-centered B, ni*nj values in axis-0-fastest (jj-outer, ii-inner) order.
            if bn == "bx":
                for _ in range((ni + 1) * nj):
                    yield 0.0
            elif bn == "by":
                for _ in range(ni * (nj + 1)):
                    yield 0.0
            else:  # bz, cell-centered
                for jj in range(nj):
                    yc = yb[0] + (jj + 0.5) * dy
                    for ii in range(ni):
                        xc = xb[0] + (ii + 0.5) * dx
                        yield _bz(xc, yc)

        return (gas_state, partial(b_field, "bx"), partial(b_field, "by"), partial(b_field, "bz"))
