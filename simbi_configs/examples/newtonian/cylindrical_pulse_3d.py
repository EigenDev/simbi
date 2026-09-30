# =============================================================================
# cylindrical_pulse_3d.py
#
# a pressure pulse on a three-dimensional cylindrical (r, phi, z) annulus:
# r in [0.1, 1], the full periodic azimuth, z in [0, 1] with outflow at the
# radial and axial ends. the pulse at r = 0.5 is shaded by cos(phi) and
# cos(2 pi z), and a rotation v_phi = 0.1 r carries it across the azimuthal
# seam.
# =============================================================================
import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class CylindricalPulse3D(SimbiProblem):
    """a shaded pressure pulse on a rotating cylindrical annulus."""

    adiabatic_index: Annotated[float, ProblemParam(1.4, description="adiabatic index")]
    resolution: Annotated[
        tuple[int, int, int], ProblemParam((32, 32, 32), cli=True, description="grid resolution (r, phi, z)")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.1, 1.0), (0.0, 2.0 * math.pi), (0.0, 1.0)], description="(r, phi, z) bounds"),
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CYLINDRICAL, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.PERIODIC,
                BoundaryCondition.PERIODIC,
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.OUTFLOW,
            ],
            description="boundary conditions [r_lo, r_hi, phi_lo, phi_hi, z_lo, z_hi]",
        ),
    ]
    end_time: Annotated[float, ProblemParam(0.5, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, v_r, v_phi, v_z, p) = (1 + 0.5 b, 0, 0.1 r, 0, 1 + b) with
        b = 0.2 exp(-((r - 0.5) / 0.1)^2) (1 + 0.5 cos(phi)) (1 + 0.5 cos(2 pi z))"""

        def gas_state() -> GasStateGenerator:
            nr, np_, nz = self.resolution
            (r_lo, r_hi), (p_lo, p_hi), (z_lo, z_hi) = self.bounds
            dr, dp, dz = (r_hi - r_lo) / nr, (p_hi - p_lo) / np_, (z_hi - z_lo) / nz
            for kk in range(nz):
                z = z_lo + (kk + 0.5) * dz
                for jj in range(np_):
                    phi = p_lo + (jj + 0.5) * dp
                    for ii in range(nr):
                        r = r_lo + (ii + 0.5) * dr
                        b = 0.2 * math.exp(-(((r - 0.5) / 0.1) ** 2))
                        b *= (1.0 + 0.5 * math.cos(phi)) * (1.0 + 0.5 * math.cos(2.0 * math.pi * z))
                        yield (1.0 + 0.5 * b, 0.0, 0.1 * r, 0.0, 1.0 + b)

        return gas_state
