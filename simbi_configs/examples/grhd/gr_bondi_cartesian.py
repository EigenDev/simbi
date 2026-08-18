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
#   - the accretion sink is horizon excision: the cells inside r_exc = 0.7 r_+ are
#     overwritten each step (the physical horizon replaces the point-mass drain),
#   - the far field is held by driven boundary faces relaxed to the ambient
#     reservoir (rho_inf, v = 0, p_inf) — a relativistic conserved-state sponge is
#     unavailable, so a dirichlet reservoir is the buffer.
#
# the grid is a uniform cube [-L, L]^3 sized to contain the bondi radius, and the
# resolution comes from that base grid alone. a static-mesh-refinement level
# telescoping onto the accretor would overlap the excised spheroid, which the excision
# gate forbids (a fine patch would evolve the excised cells and restrict them back over
# the fill).
#
# the accretion diagnostic is the rest-mass flux through a coordinate sphere,
# well-posed outside the horizon; its r-invariance once steady is the certificate:
#   from simbi.reader import read_simulation
#   from simbi.reader.gr_accretion import ring_accretion_from_checkpoint
#   mdot, cert = ring_accretion_from_checkpoint(read_simulation("...h5"))
#
# usage:
#   simbi run gr_bondi_cartesian.py --resolution 192,192,192 --end-time 300
#   simbi run gr_bondi_cartesian.py --p-ambient 0.05      (sets r_bondi)
#   simbi plot ... --draw-horizon                          (the black-disk overlay)
# =============================================================================

from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.functional import michel
from simbi.types import (
    BoundaryCondition,
    CoordSystem,
    Regime,
    Solver,
    Spacetime,
)
from simbi.types.typing import GasStateGenerator, InitialStateType


class GrBondiCartesian(SimbiProblem):
    """3D cartesian kerr-schild bondi accretion: a uniform gas at rest free-falls
    onto the hole, with horizon excision as the sink and a driven ambient reservoir."""

    # physics
    adiabatic_index: Annotated[
        float,
        ProblemParam(
            4.0 / 3.0,
            description="adiabatic index gamma (well-posed bondi needs gamma < 5/3)",
        ),
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(
            Spacetime.SCHWARZSCHILD_KS,
            description="cartesian kerr-schild background",
        ),
    ]
    schwarzschild_mass: Annotated[
        float,
        ProblemParam(
            1.0, cli=True, description="black-hole mass M (G=c=1); r_+ = 2M"
        ),
    ]
    excision_radius: Annotated[
        float,
        ProblemParam(
            -1.0,
            cli=True,
            description="excision radius; negative = auto (0.7 r_+), strictly inside the horizon",
        ),
    ]
    rho_ambient: Annotated[
        float,
        ProblemParam(
            1.0, cli=True, description="ambient rest-mass density rho_inf"
        ),
    ]
    p_ambient: Annotated[
        float,
        ProblemParam(
            0.05,
            cli=True,
            description="ambient pressure (sets c_inf and r_bondi)",
        ),
    ]

    # domain — a uniform cube [-L, L]^3 sized (L = domain_radius * r_bondi) to
    # contain the sonic surface and the bondi radius.
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam(
            (192, 192, 192),
            cli=True,
            description="grid resolution (nx, ny, nz)",
        ),
    ]
    domain_radius: Annotated[
        float,
        ProblemParam(
            3.0, cli=True, description="cube half-width in units of r_bondi"
        ),
    ]
    bounds: Annotated[
        list[tuple[float, float]] | None,
        ProblemParam(
            None, description="domain bounds (computed from r_bondi at build)"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.RHD, description="physics regime")
    ]
    # the non-diagonal cartesian kerr-schild metric takes the fast-magnetosonic HLLE
    # fan; the HLLD/HLLC wrappers apply to diagonal metrics.
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLE, description="riemann solver")
    ]
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
        ProblemParam(
            300.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
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

    def _critical_point(self) -> michel.CriticalPoint:
        """the michel transonic point for the ambient reservoir, solved once."""
        return michel.critical_point(
            gamma=self.adiabatic_index,
            density=self.rho_ambient,
            pressure=self.p_ambient,
            mass=self.schwarzschild_mass,
        )

    def _r_sonic(self) -> float:
        """the michel transonic radius, from `u_s^2 = M/(2 r_s)` closed against the
        bernoulli invariant. the newtonian estimate `(5 - 3 gamma)/4 * r_bondi` that
        stood here reported 3.75 M where the relativistic solution puts the surface at
        7.35 M, so a run sized and read against it looked for the sonic sphere at half
        its radius."""
        return self._critical_point().radius

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
        """the far-field reservoir prescription [rho, v_x, v_y, v_z, pre], held at the
        michel state the box face actually sits in rather than at the ambient state at
        infinity.

        clamping to ambient is not a small error here. the face at L = 3 r_bondi carries
        rho/rho_inf = 1.47 in the true solution, so an ambient dirichlet injects a forty-
        seven percent density deficit inward, continuously, for the whole run. the
        asymptotic michel state is closed form where u^2 is negligible against 2M/r and
        tracks the exact solution to under a percent everywhere beyond about three sonic
        radii, which is where every face of this cube lies.

        the velocity is the part a flat intuition gets wrong. the stored component is the
        valencia one, against the infalling eulerian observer of the kerr-schild chart, and
        a pressure-supported inflow falls slower than that observer: the prescription is
        OUTWARD, +0.019 at the box face, where the coordinate four-velocity is -0.024 and
        the previous reservoir said zero."""
        x1, x2, x3 = expr.coords(3)
        r = expr.sqrt(x1 * x1 + x2 * x2 + x3 * x3)
        crit = self._critical_point()
        rho, u_r, pre = michel.far_field(r, crit=crit)
        vx, vy, vz = michel.valencia_velocity(
            u_r, x1, x2, x3, r, mass=self.schwarzschild_mass
        )
        return expr.boundary([rho, vx, vy, vz, pre], dim=3)

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
            # uniform gas at rest (rho, v_x, v_y, v_z, pre). the fill overwrites the excised
            # interior from the first step, so the ambient value serves there as anywhere.
            for _ in range(nx * ny * nz):
                yield (self.rho_ambient, 0.0, 0.0, 0.0, self.p_ambient)

        return gas_state
