# =============================================================================
# gr_disk_ks_bh.py
#
# a schwarzschild black hole seen by an equatorial (R, phi) accretion DISK in the
# cylindrical kerr-schild chart — the razor-thin z = 0 slice. on the
# equator the spherical and cylindrical radii coincide (r = R), so the kerr-schild
# off-diagonal vanishes and the metric is DIAGONAL: gamma = diag(1 + 2M/R, R^2),
# alpha = 1/sqrt(1 + 2M/R), shift beta^R = 2M/(R + 2M) (beta^phi = 0), alpha sqrt(gamma)
# = R. this is the classic thin-disk chart (planar_cylindrical -> the (R, phi) plane,
# DOF = 2). the domain is an off-axis (R > 0) full annulus (phi periodic); uniform gas
# at rest free-falls radially under the covariant geodesic source.
#
# the metric is AXISYMMETRIC (phi-independent: it never reads phi), so a phi-uniform
# initial state must stay phi-uniform to roundoff — the symmetry itself is the gate for
# the disk chart, catching any azimuthal coordinate-role bug. horizon-penetrating, no floors.
#
# usage:
#   imported by test_disk_ks_bh.py
# =============================================================================

import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Spacetime
from simbi.types.typing import GasStateGenerator, InitialStateType


class GrDiskKsBH(SimbiProblem):
    """a schwarzschild BH in the equatorial (R, phi) disk chart (diagonal cylindrical kerr-schild)."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(
            Spacetime.SCHWARZSCHILD_KS, description="cylindrical kerr-schild background"
        ),
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

    # an off-axis (R > 0) full annulus: R from the near-hole zone out; phi wraps [0, 2pi).
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(4.0, 12.0), (0.0, 2.0 * math.pi)],
            description="(R, phi) domain — R off-axis, phi full",
        ),
    ]
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((48, 24), cli=True, description="grid resolution (nR, nphi)"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(
            CoordSystem.PLANAR_CYLINDRICAL,
            description="the (R, phi) equatorial disk plane",
        ),
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.PERIODIC,
                BoundaryCondition.PERIODIC,
            ],
            description="R outflow, phi periodic",
        ),
    ]

    end_time: Annotated[
        float,
        ProblemParam(
            2.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        rho, pre = self.rho_ambient, self.p_ambient
        n_r, n_phi = self.resolution

        def gas_state() -> GasStateGenerator:
            # uniform gas at rest — the 4-tuple (rho, v_R, v_phi, pre) for the (R, phi) disk (DOF = 2).
            # phi-independent (axisymmetric).
            for _ in range(n_r * n_phi):
                yield (rho, 0.0, 0.0, pre)

        return gas_state
