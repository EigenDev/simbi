# =============================================================================
# gr_cylindrical_3d_ks_bh.py
#
# a schwarzschild black hole in the FULL 3D cylindrical kerr-schild chart —
# (R, phi, z) all gridded, the non-axisymmetric case. same metric as the
# 2.5D (R, z) view (gamma_RR = 1 + 2H R^2/r^2, gamma_zz = 1 + 2H z^2/r^2, gamma_Rz =
# 2H Rz/r^2, gamma_phi-phi = R^2, r = sqrt(R^2 + z^2)); here phi is a resolved grid axis
# rather than the swirl DOF. an off-axis (R > 0), z-symmetric, phi-periodic patch;
# uniform gas at rest free-falls radially + vertically under the covariant geodesic source.
#
# the metric is AXISYMMETRIC (phi-independent), so a phi-uniform initial state must stay
# phi-uniform to roundoff even with phi FULLY GRIDDED — the correctness gate that the 3D
# path treats the resolved azimuth right. horizon-penetrating, no floors.
#
# usage:
#   simbi run gr_cylindrical_3d_ks_bh.py --end-time 0.5
# =============================================================================

import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Spacetime
from simbi.types.typing import GasStateGenerator, InitialStateType


class GrCylindrical3DKsBH(SimbiProblem):
    """a schwarzschild BH in the full 3D (R, phi, z) cylindrical kerr-schild chart."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.KERR_SCHILD, description="cylindrical kerr-schild background"),
    ]
    schwarzschild_mass: Annotated[
        float, ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1)")
    ]
    rho_ambient: Annotated[
        float, ProblemParam(1.0, cli=True, description="ambient rest-mass density")
    ]
    p_ambient: Annotated[
        float, ProblemParam(1.0e-2, cli=True, description="ambient pressure")
    ]

    # off-axis (R > 0), phi full, z symmetric about 0. r = sqrt(R^2 + z^2) stays outside 2M.
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(4.0, 10.0), (0.0, 2.0 * math.pi), (-3.0, 3.0)],
            description="(R, phi, z) domain",
        ),
    ]
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((20, 12, 16), cli=True, description="grid resolution (nR, nphi, nz)"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CYLINDRICAL, description="cylindrical (R, phi, z)")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [
                BoundaryCondition.OUTFLOW,   # R inner
                BoundaryCondition.OUTFLOW,   # R outer
                BoundaryCondition.PERIODIC,  # phi lo
                BoundaryCondition.PERIODIC,  # phi hi
                BoundaryCondition.OUTFLOW,   # z lo
                BoundaryCondition.OUTFLOW,   # z hi
            ],
            description="R outflow, phi periodic, z outflow",
        ),
    ]

    end_time: Annotated[
        float,
        ProblemParam(0.5, cli=True, checkpoint_safe=True, description="simulation end time"),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        rho, pre = self.rho_ambient, self.p_ambient
        n_r, n_phi, n_z = self.resolution

        def gas_state() -> GasStateGenerator:
            # uniform gas at rest — the 5-tuple (rho, v_R, v_phi, v_z, pre); phi-independent.
            for _ in range(n_r * n_phi * n_z):
                yield (rho, 0.0, 0.0, 0.0, pre)

        return gas_state
