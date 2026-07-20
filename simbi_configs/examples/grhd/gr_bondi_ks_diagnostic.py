# =============================================================================
# gr_bondi_ks_diagnostic.py
#
# spherical (bondi) accretion onto a schwarzschild black hole in ingoing
# kerr-schild coordinates, instrumented for accretion-rate diagnostics. the same
# horizon-penetrating physical setup as gr_bondi_ks.py, extended to resolve and
# measure the three characteristic surfaces of transonic accretion:
#   - the event horizon r_+ = 2M (the inner boundary sits BELOW it, at r < 2M),
#   - the sonic surface r_s ~ (5 - 3 gamma)/4 * r_bondi (where the inflow turns
#     supersonic), and
#   - the bondi radius r_bondi = M / c_inf^2 (where gravity overtakes pressure).
#
# the box spans from inside the horizon to several bondi radii, with log-spaced
# radial zones so all three surfaces are resolved at once: geometric-mean cell
# centers pack the finest zones near the inner boundary where the flow is
# ultra-relativistic, giving the multi-scale resolution that static mesh
# refinement provides on a cartesian grid (mesh refinement is cartesian-only, as
# its coarse-fine transfer ignores curvilinear cell volumes). the outer radial
# boundary is DRIVEN to the ambient reservoir (rho_inf, v_r = 0, p_inf): a
# relativistic conserved-state sponge is unavailable, so a dirichlet reservoir
# holds the far field while the inner flow develops.
#
# the diagnostic is the rest-mass flux through a coordinate sphere,
#   Mdot(r) = -4 pi r^2 sqrt(gamma) D tilde-v^r,
# well-posed OUTSIDE the horizon (a flux through r < 2M would be causally
# ambiguous). once the inflow is steady, Mdot(r) is r-invariant; that invariance
# is the accretion certificate:
#   from simbi.reader import read_simulation
#   from simbi.reader.gr_accretion import accretion_from_checkpoint
#   mdot, cert = accretion_from_checkpoint(read_simulation("...h5"))
#
# usage:
#   simbi run gr_bondi_ks_diagnostic.py --resolution 1024 --end-time 400
#   simbi run gr_bondi_ks_diagnostic.py --p-ambient 0.05   (sets r_bondi)
# =============================================================================

import math
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Spacetime,
)
from simbi.types.typing import GasStateGenerator, InitialStateType


class GrBondiKsDiagnostic(SimbiProblem):
    """kerr-schild spherical bondi accretion, instrumented with a driven outer
    reservoir and the shell rest-mass-flux diagnostic; log-spaced radial zones
    resolve the horizon, sonic, and bondi surfaces at once."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma (well-posed bondi needs gamma < 5/3)")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(
            Spacetime.KERR_SCHILD, description="background spacetime (ingoing kerr-schild)"
        ),
    ]
    schwarzschild_mass: Annotated[
        float, ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1); r_+ = 2M")
    ]
    rho_ambient: Annotated[
        float, ProblemParam(1.0, cli=True, description="ambient rest-mass density rho_inf")
    ]
    p_ambient: Annotated[
        float,
        ProblemParam(0.05, cli=True, description="ambient pressure (sets c_inf and r_bondi)"),
    ]

    # domain — radial, from inside the horizon to several bondi radii. r_max is
    # computed from r_bondi at build so the box always contains the sonic surface
    # and the bondi radius regardless of the ambient pressure.
    resolution: Annotated[
        int, ProblemParam(1024, cli=True, description="radial base-grid resolution")
    ]
    inner_radius: Annotated[
        float,
        ProblemParam(1.5, cli=True, description="inner boundary radius (must sit below r_+ = 2M)"),
    ]
    domain_radius: Annotated[
        float,
        ProblemParam(6.0, cli=True, description="outer radius in units of r_bondi"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(1.5, 100.0)], description="radial bounds (r_max overwritten at build from r_bondi)"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.SPHERICAL, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    x1_spacing: Annotated[
        CellSpacing, ProblemParam(CellSpacing.LOG, description="log-spaced radial zones")
    ]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [BoundaryCondition.OUTFLOW, BoundaryCondition.DYNAMIC],
            description="inner outflow (causal, below the horizon); outer driven reservoir",
        ),
    ]

    end_time: Annotated[
        float,
        ProblemParam(400.0, cli=True, checkpoint_safe=True, description="simulation end time"),
    ]

    # =========================================================================
    # derived scales
    # =========================================================================
    def _c_inf(self) -> float:
        """the ambient sound speed c_inf = sqrt(gamma p / rho); sets the bondi radius."""
        return (self.adiabatic_index * self.p_ambient / self.rho_ambient) ** 0.5

    def _r_bondi(self) -> float:
        """r_bondi = M / c_inf^2, where gravity overtakes pressure and the flow goes transonic."""
        cs = self._c_inf()
        return self.schwarzschild_mass / (cs * cs)

    def _r_sonic(self) -> float:
        """the transonic radius r_s = (5 - 3 gamma)/4 * r_bondi (the classic bondi estimate;
        the exact GR value shifts slightly but this locates the sonic surface for the setup print)."""
        return 0.25 * (5.0 - 3.0 * self.adiabatic_index) * self._r_bondi()

    def _configure(self) -> None:
        """place the outer radius at domain_radius * r_bondi so the box always contains the
        sonic surface and the bondi radius. idempotent; safe to call from build."""
        r_max = self.domain_radius * self._r_bondi()
        object.__setattr__(self, "bounds", [(self.inner_radius, r_max)])

    # =========================================================================
    # driven outer reservoir: the far field held at the ambient gas at rest
    # =========================================================================
    @computed_field
    @property
    def bx1_outer_expressions(self) -> dict:
        """the outer radial boundary driven to the ambient reservoir (rho_inf, v_r = 0, p_inf).
        the reservoir holds the far-field conditions while the inner flow develops; the
        1D relativistic-hydro prim vector is [rho, v_r, pre]."""
        g = expr.ExprGraph()
        rho = expr.constant(self.rho_ambient, g)
        v_r = expr.constant(0.0, g)
        pre = expr.constant(self.p_ambient, g)
        return g.compile([rho, v_r, pre]).serialize_boundary(dim=1)

    # =========================================================================
    # initial conditions: uniform gas at rest
    # =========================================================================
    def initial_primitive_state(self) -> InitialStateType:
        self._configure()
        cs = self._c_inf()
        print(
            f"c_inf = {cs:.3e}, r_bondi = {self._r_bondi():.2f}, "
            f"r_sonic = {self._r_sonic():.2f}, r_+ = {2.0 * self.schwarzschild_mass:.2f}, "
            f"r_max = {self.bounds[0][1]:.1f}"
        )

        def gas_state() -> GasStateGenerator:
            for _ in range(self.resolution):
                yield (self.rho_ambient, 0.0, self.p_ambient)

        return gas_state
