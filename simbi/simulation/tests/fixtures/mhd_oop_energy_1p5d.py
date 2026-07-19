# =============================================================================
# mhd_oop_energy_1p5d.py
#
# a smooth 1.5D RELATIVISTIC-MHD out-of-plane energy instrument (D=1, DOF=3). the
# transverse By has no staggered face and is a cell-centered conserved variable evolved by
# the out-of-plane cell-B flux predictor; the in-plane (normal) Bx = const rides its thin
# face. the state is a smooth, force-balanced periodic profile — uniform rho and velocity,
# a smooth By(x), and gas pressure p = P0 - By^2/2 so the total pressure is uniform — so By
# advects with the flow and no shock forms.
#
# the point is ENERGY: with the non-conservative magnetic-energy patch removed (spec §6),
# the total energy tau must be conserved to machine roundoff by the Poynting-carrying gas
# flux ALONE, even though the out-of-plane magnetic energy By^2/2 lives on a cell-centered
# component that is flux-evolved (not CT-interpolated). the relativistic c2p is a delicate
# nonlinear solve, so a tau that drifts out of sync with |B|^2 would fail to recover — a
# roundoff-tight, physical run witnesses the out-of-plane energy bookkeeping is exact.
# =============================================================================

import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType, StaggeredBFieldGenerator

_P0 = 3.0  # uniform total pressure (keeps p_gas = P0 - By^2/2 positive)
_VX = 0.3  # advection speed (subluminal for the relativistic regime)
_BX = 0.5  # constant in-plane normal field


def _by(x: float) -> float:
    return 1.0 + 0.3 * math.sin(2.0 * math.pi * x)


class MhdOopEnergy1p5d(SimbiProblem):
    """1.5D relativistic-MHD smooth out-of-plane By advection (energy-conservation instrument)."""

    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0, description="adiabatic index")]
    resolution: Annotated[
        tuple[int, int, int], ProblemParam((128, 1, 1), cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(0.0, 1.0)], description="domain boundaries")
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
        float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="simulation end time")
    ]

    def initial_primitive_state(self) -> InitialStateType:
        ni = self.resolution[0]
        xb = self.bounds[0]
        dx = (xb[1] - xb[0]) / ni

        def gas_state() -> GasStateGenerator:
            for ii in range(ni):
                xc = xb[0] + (ii + 0.5) * dx
                yield (1.0, _VX, 0.0, 0.0, _P0 - 0.5 * _by(xc) ** 2)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            # in-grid axis 0 (bx) seeds the x-faces (ni+1, constant); transverse axes 1,2
            # (by,bz) seed cell-centered B (ni each).
            if bn == "bx":
                for _ii in range(ni + 1):
                    yield _BX
            elif bn == "by":
                for ii in range(ni):
                    yield _by(xb[0] + (ii + 0.5) * dx)
            else:  # bz
                for _ii in range(ni):
                    yield 0.0

        return (gas_state, partial(b_field, "bx"), partial(b_field, "by"), partial(b_field, "bz"))
