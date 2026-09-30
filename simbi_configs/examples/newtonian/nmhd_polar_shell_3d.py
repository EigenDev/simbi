# =============================================================================
# nmhd_polar_shell_3d.py
#
# newtonian ideal MHD on a 3D spherical shell spanning both poles, threaded by
# a uniform vertical field B0 z-hat: a radial pressure pulse at r = 1.5 shaded
# by cos(theta) and cos(phi), with a swirl v_phi = 0.1 sin(theta)
# (1 + 0.3 cos(phi)) that winds the field and carries structure around the
# axis. the face values are the curl of A_phi = B0 r sin(theta) / 2 through the
# constrained-transport face weights, so the divergence the scheme preserves
# vanishes on every cell:
#   B_r(r; theta_1, theta_2) = B0 (sin^2 theta_2 - sin^2 theta_1) / (2 sin(theta_c) dtheta)
#   B_theta(theta)           = -B0 sin(theta)
#   B_phi                    = 0
# outflow at the radial ends, the coordinate axis at both poles, a periodic
# azimuth.
# =============================================================================
import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType, StaggeredBFieldGenerator


class NewtonianPolarShell3D(SimbiProblem):
    """a swirling, azimuthally shaded pulse on a 3D spherical shell in a vertical field."""

    adiabatic_index: Annotated[float, ProblemParam(1.4, description="adiabatic index")]
    b0: Annotated[float, ProblemParam(0.3, cli=True, description="vertical field strength")]
    resolution: Annotated[
        tuple[int, int, int], ProblemParam((32, 32, 32), cli=True, description="(n_r, n_theta, n_phi)")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(1.0, 2.0), (0.0, math.pi), (0.0, 2.0 * math.pi)], description="(r, theta, phi) bounds"),
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.SPHERICAL, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.NMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLD, description="numerical solver")]
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
    end_time: Annotated[float, ProblemParam(0.5, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, v_r, v_theta, v_phi, p) = (1 + b s, 0, 0, 0.1 sin(theta) (1 + 0.3 cos(phi)), 1 + b s),
        b = 0.2 exp(-((r - 1.5) / 0.08)^2), s = (1 + 0.5 cos(theta)) (1 + 0.3 cos(phi))"""

        def gas_state() -> GasStateGenerator:
            nr, nt, np_ = self.resolution
            (r_lo, r_hi), (t_lo, t_hi), (p_lo, p_hi) = self.bounds
            dr, dt, dp = (r_hi - r_lo) / nr, (t_hi - t_lo) / nt, (p_hi - p_lo) / np_
            for kk in range(np_):
                phi = p_lo + (kk + 0.5) * dp
                for jj in range(nt):
                    theta = t_lo + (jj + 0.5) * dt
                    for ii in range(nr):
                        r = r_lo + (ii + 0.5) * dr
                        b = 0.2 * math.exp(-(((r - 1.5) / 0.08) ** 2))
                        s = (1.0 + 0.5 * math.cos(theta)) * (1.0 + 0.3 * math.cos(phi))
                        v_phi = 0.1 * math.sin(theta) * (1.0 + 0.3 * math.cos(phi))
                        yield (1.0 + b * s, 0.0, 0.0, v_phi, 1.0 + b * s)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            nr, nt, np_ = self.resolution
            (t_lo, t_hi) = self.bounds[1]
            dt = (t_hi - t_lo) / nt
            for _kk in range(np_ + (bn == "bz")):
                for jj in range(nt + (bn == "by")):
                    for _ii in range(nr + (bn == "bx")):
                        if bn == "bx":
                            t1, t2 = t_lo + jj * dt, t_lo + (jj + 1) * dt
                            yield self.b0 * (math.sin(t2) ** 2 - math.sin(t1) ** 2) / (2.0 * math.sin(0.5 * (t1 + t2)) * dt)
                        elif bn == "by":
                            yield -self.b0 * math.sin(t_lo + jj * dt)
                        else:
                            yield 0.0

        return (gas_state, partial(b_field, "bx"), partial(b_field, "by"), partial(b_field, "bz"))
