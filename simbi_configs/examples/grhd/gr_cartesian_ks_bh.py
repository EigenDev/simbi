# =============================================================================
# gr_cartesian_ks_bh.py
#
# a schwarzschild black hole in the CARTESIAN kerr-schild chart (design 45) — the
# (x, y) equatorial slice of the horizon-penetrating spacetime, on a cartesian
# patch OFF the origin so no interior excision is needed. the metric is
# gamma_ij = delta_ij + 2M x_i x_j / r^3 (NON-diagonal), alpha = 1/sqrt(1 + 2M/r),
# shift beta^i = 2M x_i / (r^2 (r + 2M)) (nonzero on BOTH axes) — the first GR run
# in a non-spherical chart. uniform gas at rest free-falls under the covariant
# geodesic source; the flow is transported by the metric-aware Valencia flux with
# the shift on every sweep and the state-independent light-cone CFL.
#
# the metric is EXACTLY symmetric under x <-> y (r = sqrt(x^2 + y^2) is symmetric,
# and gamma_ij / beta^i map into each other under the index swap), so a symmetric
# initial state on a square patch with symmetric boundaries must evolve x <-> y
# symmetrically to ROUNDOFF. this is the oracle-free correctness gate for the whole
# chart-generic GR chain: any coordinate-role bug (an axis treated as radial, a
# shift applied on one axis only) breaks the symmetry exactly. horizon-penetrating,
# no floors.
#
# usage:
#   simbi run gr_cartesian_ks_bh.py --resolution 64 --end-time 2.0
# =============================================================================

from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Spacetime
from simbi.types.typing import GasStateGenerator, InitialStateType


class GrCartesianKsBH(SimbiProblem):
    """a schwarzschild BH in the cartesian kerr-schild chart, on an off-origin (x, y) patch."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.KERR_SCHILD, description="cartesian kerr-schild background"),
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

    # a SQUARE patch off the origin: r spans ~sqrt(2)*4 to ~sqrt(2)*12, all outside the
    # horizon r = 2M, so no interior excision — the metric is smooth over the whole
    # domain. the square + EQUAL bounds on both axes make the x <-> y symmetry exact.
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(4.0, 12.0), (4.0, 12.0)], description="(x, y) domain — equal, off-origin"),
    ]
    resolution: Annotated[
        tuple[int, int], ProblemParam((64, 64), cli=True, description="grid resolution (nx, ny)")
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [BoundaryCondition.OUTFLOW] * 4,
            description="outflow on all four edges (symmetric under x <-> y)",
        ),
    ]

    end_time: Annotated[
        float,
        ProblemParam(2.0, cli=True, checkpoint_safe=True, description="simulation end time"),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        rho, pre = self.rho_ambient, self.p_ambient
        nx, ny = self.resolution

        def gas_state() -> GasStateGenerator:
            # uniform gas at rest: (rho, v_x, v_y, pre). symmetric under x <-> y.
            for _ in range(nx * ny):
                yield (rho, 0.0, 0.0, pre)

        return gas_state
