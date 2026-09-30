# =============================================================================
# rz_pulse.py
#
# a pressure pulse on the axisymmetric cylindrical (r, z) plane with the
# symmetry axis on the grid: the coordinate axis at r = 0, outflow at r = 1 and
# at both z ends. the pulse sits off the axis at (r, z) = (0.4, 0.5) and a
# radial flow v_r = 0.05 sin(pi r), zero on the axis, stirs the plane so the
# axis fill matters from the first step.
#
# usage:
#   simbi run simbi_configs/examples/newtonian/rz_pulse.py --resolution 64,64
# =============================================================================
import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class RzPulse(SimbiProblem):
    """an off-axis pressure pulse on the cylindrical (r, z) plane."""

    adiabatic_index: Annotated[float, ProblemParam(1.4, description="adiabatic index")]
    resolution: Annotated[tuple[int, int], ProblemParam((64, 64), cli=True, description="grid resolution (r, z)")]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="(r, z) bounds")
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CYLINDRICAL, description="the (r, z) plane")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.AXIS, BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW],
            description="boundary conditions [r_lo, r_hi, z_lo, z_hi]",
        ),
    ]
    end_time: Annotated[float, ProblemParam(0.5, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, v_r, v_z, p) = (1 + 0.5 b, 0.05 sin(pi r), 0, 1 + b), b = exp(-|x - (0.4, 0.5)|^2 / 0.01)"""

        def gas_state() -> GasStateGenerator:
            nr, nz = self.resolution
            (r_lo, r_hi), (z_lo, z_hi) = self.bounds
            dr, dz = (r_hi - r_lo) / nr, (z_hi - z_lo) / nz
            for jj in range(nz):
                z = z_lo + (jj + 0.5) * dz
                for ii in range(nr):
                    r = r_lo + (ii + 0.5) * dr
                    b = math.exp(-((r - 0.4) ** 2 + (z - 0.5) ** 2) / 0.01)
                    yield (1.0 + 0.5 * b, 0.05 * math.sin(math.pi * r), 0.0, 1.0 + b)

        return gas_state
