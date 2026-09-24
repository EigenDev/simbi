# =============================================================================
# spherical_pulse_3d.py
#
# a radial pressure pulse shaded in latitude and longitude on a spherical
# (r, theta, phi) shell gridded over both poles with a periodic azimuth:
# outflow at the radial ends, the coordinate axis at the poles. meridional
# and azimuthal flows that vanish at the poles stir the shell so the polar
# fill matters from the first step.
#
# usage:
#   simbi run simbi_configs/examples/newtonian/spherical_pulse_3d.py --resolution 32,32,32
# =============================================================================

import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class SphericalPulse3D(SimbiProblem):
    """a latitude- and longitude-shaded radial pulse on a spherical shell."""

    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((32, 32, 32), cli=True, description="grid resolution (r, theta, phi)"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(1.0, 2.0), (0.0, math.pi), (0.0, 2.0 * math.pi)],
            description="domain boundaries",
        ),
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
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.AXIS,
                BoundaryCondition.AXIS,
                BoundaryCondition.PERIODIC,
                BoundaryCondition.PERIODIC,
            ],
            description="boundary conditions [r_lo, r_hi, theta_lo, theta_hi, phi_lo, phi_hi]",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            0.5, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """a gaussian pulse at r = 1.5 shaded by cos(theta) and cos(phi), with
        v_theta = 0.05 sin(2 theta) and v_phi = 0.05 sin(theta) cos(phi). the primitive is
        (rho, v_r, v_theta, v_phi, p) at the cell centers."""

        def gas_state() -> GasStateGenerator:
            nr, nt, np_ = self.resolution
            (r_lo, r_hi), (t_lo, t_hi), (p_lo, p_hi) = self.bounds
            dr = (r_hi - r_lo) / nr
            dt = (t_hi - t_lo) / nt
            dp = (p_hi - p_lo) / np_
            for kk in range(np_):
                phi = p_lo + (kk + 0.5) * dp
                for jj in range(nt):
                    theta = t_lo + (jj + 0.5) * dt
                    for ii in range(nr):
                        r = r_lo + (ii + 0.5) * dr
                        b = 0.2 * math.exp(-(((r - 1.5) / 0.08) ** 2))
                        yield (
                            1.0 + b * (1.0 + 0.5 * math.cos(theta)) * (1.0 + 0.3 * math.cos(phi)),
                            0.0,
                            0.05 * math.sin(2.0 * theta),
                            0.05 * math.sin(theta) * math.cos(phi),
                            1.0 + b,
                        )

        return gas_state
