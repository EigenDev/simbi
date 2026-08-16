# =============================================================================
# gr_cylindrical_ks_bh.py
#
# a schwarzschild black hole in the cylindrical kerr-schild chart
# — the 2.5D axisymmetric (R, z) grid carrying the azimuthal v_phi DOF, the natural
# chart for relativistic jets and accretion disks around a hole. the metric is
# gamma_RR = 1 + 2H R^2/r^2, gamma_zz = 1 + 2H z^2/r^2, gamma_Rz = 2H Rz/r^2,
# gamma_phi-phi = R^2, with r = sqrt(R^2 + z^2) the spherical (BH) radius; alpha =
# 1/sqrt(1 + 2H), shift beta^R, beta^z (beta^phi = 0), alpha sqrt(gamma) = R. the
# domain is an off-axis (R > 0), z-symmetric annular patch so no interior excision is
# needed and the metric is smooth throughout. uniform gas at rest free-falls under the
# covariant geodesic source.
#
# the metric is exactly symmetric under z -> -z (r = sqrt(R^2 + z^2) is even in z, and
# gamma_Rz / beta^z flip sign with the z-momentum), so a z-symmetric initial state on a
# grid symmetric about z = 0 evolves z-reflection symmetrically to roundoff: rho / p
# even in z, v_z odd, v_R / v_phi even. the symmetry itself is the correctness gate for the
# cylindrical chart — the analog of the cartesian x <-> y test — catching any
# coordinate-role or one-axis-shift bug (e.g. the densitization lapse using R as the
# radius where the metric radius sqrt(R^2 + z^2) is required). horizon-penetrating, no floors.
#
# usage:
#   simbi run gr_cylindrical_ks_bh.py --resolution 48 --end-time 2.0
# =============================================================================

from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Spacetime
from simbi.types.typing import GasStateGenerator, InitialStateType


class GrCylindricalKsBH(SimbiProblem):
    """a schwarzschild BH in the cylindrical kerr-schild chart, on a z-symmetric (R, z) patch."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.SCHWARZSCHILD_KS, description="cylindrical kerr-schild background"),
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

    # an off-axis (R > 0), z-symmetric annular patch: r = sqrt(R^2 + z^2) spans ~4 to ~12.6,
    # all outside the horizon r = 2M — no interior excision, smooth metric. the z bounds are
    # equal-and-opposite so z -> -z is an exact grid symmetry.
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(4.0, 12.0), (-4.0, 4.0)], description="(R, z) domain — R off-axis, z symmetric about 0"
        ),
    ]
    resolution: Annotated[
        tuple[int, int], ProblemParam((48, 48), cli=True, description="grid resolution (nR, nz)")
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CYLINDRICAL, description="cylindrical (R, z) axisymmetric"),
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [BoundaryCondition.OUTFLOW] * 4,
            description="outflow on all four edges (symmetric under z -> -z)",
        ),
    ]

    end_time: Annotated[
        float,
        ProblemParam(2.0, cli=True, checkpoint_safe=True, description="simulation end time"),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        rho, pre = self.rho_ambient, self.p_ambient
        n_r, n_z = self.resolution

        def gas_state() -> GasStateGenerator:
            # uniform gas at rest — the 5-tuple (rho, v_R, v_phi, v_z, pre) for the 2.5D swirl DOF.
            # z-symmetric (independent of z).
            for _ in range(n_r * n_z):
                yield (rho, 0.0, 0.0, 0.0, pre)

        return gas_state
