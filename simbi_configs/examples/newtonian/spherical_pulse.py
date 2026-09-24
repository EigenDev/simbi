# =============================================================================
# spherical_pulse.py
#
# a radial pressure pulse shaded in latitude on a spherical (r, theta) shell
# with a logarithmic radial axis and both poles on the grid: outflow at the
# radial ends, the coordinate axis at the poles. a meridional flow that
# vanishes at the poles stirs the shell so the polar fill matters from the
# first step.
#
# usage:
#   simbi run simbi_configs/examples/newtonian/spherical_pulse.py --resolution 64,64
# =============================================================================

import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class SphericalPulse(SimbiProblem):
    """a latitude-shaded radial pulse on a log-radial spherical shell."""

    resolution: Annotated[
        tuple[int, int],
        ProblemParam((64, 64), cli=True, description="grid resolution (r, theta)"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(1.0, 2.0), (0.0, math.pi)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.SPHERICAL, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]
    adiabatic_index: Annotated[
        float, ProblemParam(1.4, description="adiabatic index")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LOG, description="radial spacing"),
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.AXIS,
                BoundaryCondition.AXIS,
            ],
            description="boundary conditions [r_lo, r_hi, theta_lo, theta_hi]",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            0.5, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """a gaussian pulse at r = 1.5 shaded by cos(theta), with v_theta = 0.05 sin(2 theta).
        the primitive is (rho, v_r, v_theta, p), evaluated at the cell centers of the
        log-radial, uniform-theta grid."""

        def gas_state() -> GasStateGenerator:
            nr, nt = self.resolution
            (r_lo, r_hi), (t_lo, t_hi) = self.bounds
            slope = math.log10(r_hi / r_lo) / nr
            dt = (t_hi - t_lo) / nt
            for jj in range(nt):
                theta = t_lo + (jj + 0.5) * dt
                for ii in range(nr):
                    r = math.sqrt(r_lo * 10.0 ** (ii * slope) * r_lo * 10.0 ** ((ii + 1) * slope))
                    b = 0.2 * math.exp(-(((r - 1.5) / 0.08) ** 2))
                    yield (
                        1.0 + b * (1.0 + 0.5 * math.cos(theta)),
                        0.0,
                        0.05 * math.sin(2.0 * theta),
                        1.0 + b,
                    )

        return gas_state
