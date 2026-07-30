# =============================================================================
# gr_bondi_cartesian.py
#
# 3D bondi accretion onto a schwarzschild black hole on a cartesian kerr-schild
# grid — the general-relativistic analog of the newtonian 3D cartesian bondi.py.
# a uniform gas at rest free-falls onto the hole at the origin; the flow develops
# a transonic inflow and crosses the horizon.
#
# the three roles bondi.py splits between an immersed point mass and a sponge are
# played here by the spacetime itself:
#   - gravity is the kerr-schild geodesic source (no immersed body),
#   - the accretion sink is HORIZON EXCISION: the cells inside r_exc = 0.7 r_+ are
#     overwritten each step (the physical horizon replaces the point-mass drain),
#   - the far field is held by DRIVEN boundary faces relaxed to the ambient
#     reservoir (rho_inf, v = 0, p_inf) — a relativistic conserved-state sponge is
#     unavailable, so a dirichlet reservoir is the buffer.
#
# the grid is a uniform cube [-L, L]^3 sized to contain the bondi radius. static
# mesh refinement is NOT used: a fine level telescoping onto the accretor would
# overlap the excised spheroid, which the excision gate forbids (a fine patch
# would evolve the excised cells and restrict them back over the fill). resolution
# comes from the base grid alone.
#
# the accretion diagnostic is the rest-mass flux through a coordinate sphere,
# well-posed OUTSIDE the horizon; its r-invariance once steady is the certificate:
#   from simbi.reader import read_simulation
#   from simbi.reader.gr_accretion import ring_accretion_from_checkpoint
#   mdot, cert = ring_accretion_from_checkpoint(read_simulation("...h5"))
#
# usage:
#   simbi run gr_bondi_cartesian.py --resolution 192,192,192 --end-time 300
#   simbi run gr_bondi_cartesian.py --p-ambient 0.05      (sets r_bondi)
#   simbi plot ... --draw-horizon                          (the black-disk overlay)
# =============================================================================

import math
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver, Spacetime
from simbi.types.typing import GasStateGenerator, InitialStateType


class GrBondiCartesian(SimbiProblem):
    """3D cartesian kerr-schild bondi accretion: a uniform gas at rest free-falls
    onto the hole, with horizon excision as the sink and a driven ambient reservoir."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma (well-posed bondi needs gamma < 5/3)")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.SCHWARZSCHILD_KS, description="cartesian kerr-schild background"),
    ]
    schwarzschild_mass: Annotated[
        float, ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1); r_+ = 2M")
    ]
    excision_radius: Annotated[
        float,
        ProblemParam(
            -1.0, cli=True, description="excision radius; negative = auto (0.7 r_+), strictly inside the horizon"
        ),
    ]
    rho_ambient: Annotated[
        float, ProblemParam(1.0, cli=True, description="ambient rest-mass density rho_inf")
    ]
    p_ambient: Annotated[
        float, ProblemParam(0.05, cli=True, description="ambient pressure (sets c_inf and r_bondi)")
    ]

    # domain — a uniform cube [-L, L]^3 sized (L = domain_radius * r_bondi) to
    # contain the sonic surface and the bondi radius.
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((192, 192, 192), cli=True, description="grid resolution (nx, ny, nz)"),
    ]
    domain_radius: Annotated[
        float, ProblemParam(3.0, cli=True, description="cube half-width in units of r_bondi")
    ]
    bounds: Annotated[
        list[tuple[float, float]] | None,
        ProblemParam(None, description="domain bounds (computed from r_bondi at build)"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    # the non-diagonal cartesian kerr-schild metric takes the fast-magnetosonic HLLE
    # fan; the diagonal-metric HLLD/HLLC wrappers do not apply.
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, description="riemann solver")]
    cfl_number: Annotated[float, ProblemParam(0.3, description="cfl number")]
    # all six cube faces are far-field: drive them to the ambient reservoir.
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [BoundaryCondition.DYNAMIC] * 6,
            description="all faces driven to the ambient reservoir (the buffer)",
        ),
    ]

    end_time: Annotated[
        float,
        ProblemParam(300.0, cli=True, checkpoint_safe=True, description="simulation end time"),
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
        """the transonic radius r_s = (5 - 3 gamma)/4 * r_bondi (the classic bondi estimate)."""
        return 0.25 * (5.0 - 3.0 * self.adiabatic_index) * self._r_bondi()

    def setup(self) -> None:
        """size the cube from r_bondi and auto-place the excision surface at 0.7 r_+."""
        super().setup()
        ll = self.domain_radius * self._r_bondi()
        if self.bounds is None:
            self.bounds = [(-ll, ll), (-ll, ll), (-ll, ll)]
        if self.excision_radius < 0.0:
            self.excision_radius = 0.7 * (2.0 * self.schwarzschild_mass)
        print(
            f"c_inf = {self._c_inf():.3e}, r_bondi = {self._r_bondi():.2f}, "
            f"r_sonic = {self._r_sonic():.2f}, r_+ = {2.0 * self.schwarzschild_mass:.2f}, "
            f"r_exc = {self.excision_radius:.2f}, L = {ll:.1f}"
        )

    # =========================================================================
    # driven reservoir: every cube face held at the ambient gas at rest
    # =========================================================================
    def _reservoir(self) -> dict:
        """the ambient reservoir prescription [rho, v_x, v_y, v_z, pre] = (rho_inf, 0, 0, 0, p_inf)
        as a constant boundary DAG; one state serves all six far-field faces."""
        g = expr.ExprGraph()
        rho = expr.constant(self.rho_ambient, g)
        zero = expr.constant(0.0, g)
        pre = expr.constant(self.p_ambient, g)
        return g.compile([rho, zero, zero, zero, pre]).serialize_boundary(dim=3)

    @computed_field
    @property
    def bx1_inner_expressions(self) -> dict:
        return self._reservoir()

    @computed_field
    @property
    def bx1_outer_expressions(self) -> dict:
        return self._reservoir()

    @computed_field
    @property
    def bx2_inner_expressions(self) -> dict:
        return self._reservoir()

    @computed_field
    @property
    def bx2_outer_expressions(self) -> dict:
        return self._reservoir()

    @computed_field
    @property
    def bx3_inner_expressions(self) -> dict:
        return self._reservoir()

    @computed_field
    @property
    def bx3_outer_expressions(self) -> dict:
        return self._reservoir()

    # =========================================================================
    # initial conditions: uniform gas at rest
    # =========================================================================
    def initial_primitive_state(self) -> InitialStateType:
        nx, ny, nz = self.resolution

        def gas_state() -> GasStateGenerator:
            # uniform gas at rest (rho, v_x, v_y, v_z, pre). the excised interior is
            # overwritten by the fill from the first step, so its initial value never matters.
            for _ in range(nx * ny * nz):
                yield (self.rho_ambient, 0.0, 0.0, 0.0, self.p_ambient)

        return gas_state
