# =============================================================================
# mhd_energy_conservation.py
#
# periodic 2d relativistic MHD shock (colliding v = +0.95 / -0.94 streams, p = 1e-4,
# uniform transverse By) — a magnetized shock that does NOT fire the FOFC fallback, so
# it exercises the BASE constrained-transport scheme. the total energy tau is the sum
# of the conserved buffer over the periodic interior; a conservative scheme holds it to
# machine roundoff at EVERY resolution.
#
# this is the base-scheme energy-conservation instrument. a magnetic-energy PATCH
# (`nrg += 1/2 d|bcell|^2`, applied outside the flux) makes tau drift ~2e-4 at nx=256
# and GROW with resolution (6e-4 at nx=512) — the signature of genuine
# non-conservation, since truncation error would shrink with resolution. deriving cell B
# from the CT face field with no patch leaves tau conserved by the Poynting-carrying
# Godunov flux, at roundoff drift that does NOT grow with resolution.
# =============================================================================

from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType, StaggeredBFieldGenerator


class MhdEnergyConservation(SimbiProblem):
    """periodic 2d magnetized relativistic shock (base-scheme energy-conservation instrument)."""

    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0, description="adiabatic index")]
    b0: Annotated[float, ProblemParam(0.2, cli=True, description="transverse By")]
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
        float, ProblemParam(0.1, cli=True, checkpoint_safe=True, description="simulation end time")
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """magnetized relativistic streams colliding at x = 0.5 and the periodic wrap."""

        def gas_state() -> GasStateGenerator:
            ni, nj, _nk = self.resolution
            xb = self.bounds[0]
            dx = (xb[1] - xb[0]) / ni
            for _kk in range(1):
                for _jj in range(nj):
                    for ii in range(ni):
                        xi = xb[0] + (ii + 0.5) * dx
                        v = 0.95 if xi <= 0.5 * (xb[1] - xb[0]) else -0.94
                        yield (1.0, v, 0.0, 0.0, 1e-4)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj, _nk = self.resolution
            for _kk in range(1 + (bn == "bz")):
                for _jj in range(nj + (bn == "by")):
                    for _ii in range(ni + (bn == "bx")):
                        yield self.b0 if bn == "by" else 0.0

        return (gas_state, partial(b_field, "bx"), partial(b_field, "by"), partial(b_field, "bz"))
