# =============================================================================
# isothermal_spherical_pulse.py
#
# isothermal hydro on a spherical (r, theta) shell with a logarithmic radial
# axis and both poles on the grid: a gaussian overdensity at r = 1.5 shaded by
# cos(theta), stirred by a meridional flow v_theta = 0.05 sin(2 theta) that
# vanishes at the poles. outflow at the radial ends, the coordinate axis at the
# poles.
#
# usage:
#   simbi run simbi_configs/examples/isothermal/isothermal_spherical_pulse.py --resolution 64,64
# =============================================================================
import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class IsothermalSphericalPulse(SimbiProblem):
    """a latitude-shaded isothermal overdensity on a log-radial spherical shell."""

    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]
    resolution: Annotated[tuple[int, int], ProblemParam((64, 64), cli=True, description="grid resolution (r, theta)")]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(1.0, 2.0), (0.0, math.pi)], description="domain boundaries")
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.SPHERICAL, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")]
    x1_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.LOG, description="radial spacing")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW, BoundaryCondition.AXIS, BoundaryCondition.AXIS],
            description="boundary conditions [r_lo, r_hi, theta_lo, theta_hi]",
        ),
    ]
    end_time: Annotated[float, ProblemParam(0.5, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, v_r, v_theta) = (1 + b (1 + 0.5 cos(theta)), 0, 0.05 sin(2 theta)) with
        b = 0.2 exp(-((r - 1.5) / 0.08)^2), at the cell centers of the radial grid (log or linear)"""

        def gas_state() -> GasStateGenerator:
            nr, nt = self.resolution
            (r_lo, r_hi), (t_lo, t_hi) = self.bounds
            slope = math.log10(r_hi / r_lo) / nr
            dr = (r_hi - r_lo) / nr
            log_radial = self.x1_spacing == CellSpacing.LOG
            dt = (t_hi - t_lo) / nt
            for jj in range(nt):
                theta = t_lo + (jj + 0.5) * dt
                for ii in range(nr):
                    if log_radial:
                        r = math.sqrt(r_lo * 10.0 ** (ii * slope) * r_lo * 10.0 ** ((ii + 1) * slope))
                    else:
                        r = r_lo + (ii + 0.5) * dr
                    b = 0.2 * math.exp(-(((r - 1.5) / 0.08) ** 2))
                    yield (1.0 + b * (1.0 + 0.5 * math.cos(theta)), 0.0, 0.05 * math.sin(2.0 * theta))

        return gas_state
