# =============================================================================
# gr_cartesian_ks_excised.py
#
# a schwarzschild black hole with the horizon ON the grid: an origin-containing
# square in the cartesian kerr-schild chart, with the region inside the horizon
# handled by SDF excision — every cell inside r_exc is overwritten each step with
# a zero-gradient copy of its outward neighbor's primitives (plus a local
# conserved rebuild), and the metric's coordinate singularity at r = 0 is clamped
# below M/2. no inner boundary exists: causality seals the horizon, and the
# excision surface sits strictly inside it, so nothing done there can reach the
# exterior. uniform gas at rest free-falls onto the hole (the gr_bondi analog
# with the hole in the box).
#
# the certificate this config exists to run:
#   1. x <-> y symmetry to roundoff (square domain, symmetric IC, origin-centered
#      excision sphere — any excision-induced asymmetry breaks it exactly),
#   2. the exterior is INDEPENDENT of r_exc (run --excision-radius 1.2 vs 1.6:
#      exterior fields agree to truncation — the causal-protection theorem,
#      numerically),
#   3. the ring flux Mdot(r_ex) is r_ex-invariant once the inner flow is steady:
#      from simbi.reader import read_simulation
#      from simbi.reader.gr_accretion import ring_accretion_from_checkpoint
#      mdot, cert = ring_accretion_from_checkpoint(read_simulation("...h5"))
#
# usage:
#   simbi run gr_cartesian_ks_excised.py --resolution 256,256 --end-time 100
#   simbi run gr_cartesian_ks_excised.py --excision-radius 1.2   (the r_exc gate)
# =============================================================================

from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Spacetime
from simbi.types.typing import GasStateGenerator, InitialStateType


class GrCartesianKsExcised(SimbiProblem):
    """a schwarzschild BH with the horizon on the grid: origin-containing cartesian
    kerr-schild square, SDF-excised inside the horizon."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.KERR_SCHILD, description="cartesian kerr-schild background"),
    ]
    schwarzschild_mass: Annotated[
        float, ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1); r_+ = 2M")
    ]
    excision_radius: Annotated[
        float,
        ProblemParam(
            1.4,
            cli=True,
            description="excision sphere radius; must sit in (M/2, 2M) — "
            "strictly inside the horizon, above the metric-guard radius",
        ),
    ]
    rho_ambient: Annotated[
        float, ProblemParam(1.0, cli=True, description="ambient rest-mass density")
    ]
    p_ambient: Annotated[
        float, ProblemParam(1.0e-2, cli=True, description="ambient pressure")
    ]

    # an origin-containing SQUARE with equal bounds: the horizon r_+ = 2M sits well
    # inside, and the x <-> y symmetry of the chart + IC + excision sphere is exact.
    # EVEN resolution keeps cell centers off the axes (the excision fill's outward
    # selects are never on the sign tie-point).
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(-16.0, 16.0), (-16.0, 16.0)],
            description="(x, y) domain — equal, origin-containing",
        ),
    ]
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((256, 256), cli=True, description="grid resolution (nx, ny); keep even"),
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
        ProblemParam(
            100.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        rho, pre = self.rho_ambient, self.p_ambient
        nx, ny = self.resolution

        def gas_state() -> GasStateGenerator:
            # uniform gas at rest: (rho, v_x, v_y, pre). symmetric under x <-> y.
            # the excised interior is overwritten by the fill from the first step,
            # so the (unphysical) uniform state inside the horizon never matters.
            for _ in range(nx * ny):
                yield (rho, 0.0, 0.0, pre)

        return gas_state
