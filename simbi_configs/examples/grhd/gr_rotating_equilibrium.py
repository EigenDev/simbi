# =============================================================================
# gr_rotating_equilibrium.py
#
# the surface-free constant-l rotating equilibrium held on the (r, theta) wedge
# with DRIVEN boundaries — every ghost band pinned to the analytic state. the
# precision hold problem for the rotating balance (centrifugal + gravity +
# pressure, both sweeps): a theta-stratified equilibrium is mathematically
# incompatible with mirror/copy ghosts (they impose dp/dtheta = 0 where the state
# requires the centrifugal-balancing gradient), so all four faces prescribe the
# exact analytic continuation through the DYNAMIC boundary expressions.
#
# the state is the fishbone-moncrief constant-l potential with its integration
# constant anchored at the DOMAIN MINIMUM (see `RotatingEquilibrium` in
# gr_fishbone_moncrief.py), so the thermal margin p/rho is bounded below by
# `p_rho_ref` everywhere — no cold near-equipotential cells.
#
# usage:
#   simbi run gr_rotating_equilibrium.py
#   (the hold gate: simbi/simulation/tests/test_schwarzschild_rotating_equilibrium.py)
# =============================================================================

import math
from typing import Annotated

from pydantic import computed_field, model_validator

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Spacetime,
)
from simbi.types.typing import GasStateGenerator, InitialStateType
from simbi_configs.examples.grmhd.gr_fishbone_moncrief import RotatingEquilibrium


class GrRotatingEquilibrium(SimbiProblem):
    """the surface-free constant-l rotating equilibrium, all four faces pinned."""

    adiabatic_index: Annotated[
        float,
        ProblemParam(4.0 / 3.0, description="adiabatic index gamma"),
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(
            Spacetime.SCHWARZSCHILD, description="background spacetime"
        ),
    ]
    schwarzschild_mass: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1)"),
    ]
    r_pressure_max: Annotated[
        float,
        ProblemParam(
            16.0,
            cli=True,
            description="pressure-maximum radius of the rotation law",
        ),
    ]
    p_rho_ref: Annotated[
        float,
        ProblemParam(
            1.0e-2,
            description="p/rho at the coldest domain point (thermal margin)",
        ),
    ]

    nr: Annotated[
        int, ProblemParam(96, cli=True, description="radial resolution")
    ]
    npolar: Annotated[
        int, ProblemParam(32, cli=True, description="polar (theta) resolution")
    ]
    resolution: Annotated[
        tuple[int, int],
        ProblemParam(
            (0, 0), description="grid resolution (nr, npolar) — computed"
        ),
    ]
    theta_halfwidth: Annotated[
        float,
        ProblemParam(
            0.3, description="half-width of the equatorial theta wedge (rad)"
        ),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(0.0, 0.0), (0.0, 0.0)], description="domain bounds — computed"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.SPHERICAL, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.RHD, description="physics regime")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LOG, description="log-spaced radial zones"),
    ]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [
                BoundaryCondition.DYNAMIC,
                BoundaryCondition.DYNAMIC,
                BoundaryCondition.DYNAMIC,
                BoundaryCondition.DYNAMIC,
            ],
            description="all four faces pinned to the analytic equilibrium",
        ),
    ]

    end_time: Annotated[
        float,
        ProblemParam(
            100.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    @model_validator(mode="after")
    def compute_defaults(self) -> "GrRotatingEquilibrium":
        self.resolution = (self.nr, self.npolar)
        theta_c = math.pi / 2.0
        if self.kerr_spin != 0.0:
            # spinning: the kerr spacetime. the domain stays OUTSIDE the horizon —
            # a stationary constant-l azimuthal flow needs a timelike LNRF
            # (Delta > 0), so this state has no through-horizon continuation.
            self.spacetime = Spacetime.KERR_KS
        self.bounds = [
            (3.0, 100.0),
            (theta_c - self.theta_halfwidth, theta_c + self.theta_halfwidth),
        ]
        return self

    def equilibrium(self) -> RotatingEquilibrium:
        return RotatingEquilibrium(
            mass=self.schwarzschild_mass,
            r_max=self.r_pressure_max,
            gamma=self.adiabatic_index,
            rho_ref=1.0,
            p_rho_ref=self.p_rho_ref,
            bounds_r=self.bounds[0],
            bounds_theta=tuple(self.bounds[1]),
            spin=self.kerr_spin,
            chart="ks" if self.kerr_spin != 0.0 else "bl",
        )

    def _boundary_prescription(self) -> dict:
        """the complete prim prescription [rho, v_r, v_theta, v_phi, pre] of the
        equilibrium as coordinate expressions over (r, theta) — the SAME analytic
        state `RotatingEquilibrium.primitive` evaluates (FM 1976 eqs. 3.3/3.6 at
        general spin, in the configured chart), lowered to the rust boundary-DAG
        wire format. one prescription serves all four faces (the state is global)."""
        import simbi.expression as expr

        eq = self.equilibrium()
        gm = self.adiabatic_index
        gm1 = gm - 1.0
        mm = self.schwarzschild_mass
        a = self.kerr_spin
        l = eq.fm.ell

        g = expr.ExprGraph()
        r = expr.variable("r", g)
        th = expr.variable("theta", g)
        st = expr.sin(th)
        ct = expr.cos(th)
        sigma = r * r + (a * a) * ct * ct
        delta = r * r - (2.0 * mm) * r + a * a
        big_a = (r * r + a * a) * (r * r + a * a) - (a * a) * delta * st * st
        xx = (4.0 * l * l) * sigma * sigma * delta / (big_a * st * (big_a * st))
        sq = expr.sqrt(1.0 + xx)
        lnh = (
            0.5 * expr.log((1.0 + sq) / (sigma * delta / big_a))
            - 0.5 * sq
            - (2.0 * a * mm * l) * r / big_a
            - eq.cc
        )
        h = expr.exp(lnh)
        rho = ((h - 1.0) * (gm1 / (gm * eq.kk))) ** (1.0 / gm1)
        pre = eq.kk * rho**gm
        # eq. 3.3 in the LNRF, then the chart map. sign(l) folds into the constant.
        u_lnrf = math.copysign(1.0, l) * expr.sqrt(0.5 * (sq - 1.0))
        e_nu = expr.sqrt(sigma * delta / big_a)
        e_psi = expr.sqrt(big_a / sigma) * st
        omega = (2.0 * a * mm) * r / big_a
        u_t = expr.sqrt(1.0 + u_lnrf * u_lnrf) / e_nu
        u_p = omega * u_t + u_lnrf / e_psi
        zero = expr.constant(0.0, g)
        if self.kerr_spin != 0.0:
            # ks chart: v^r = b/sqrt(1+b) (the orbiter's drift against the infalling
            # eulerian observers), v^phi = u^phi sqrt(1+b) / u^t.
            b = (2.0 * mm) * r / sigma
            sq_b = expr.sqrt(1.0 + b)
            v_r = b / sq_b
            vphi = u_p * sq_b / u_t
        else:
            # bl chart at a = 0: alpha = e^nu, zero shift.
            v_r = zero
            vphi = u_p / (e_nu * u_t)

        compiled = g.compile([rho, v_r, zero, vphi, pre])
        return compiled.serialize_boundary(dim=3)

    @computed_field
    @property
    def bx1_inner_expressions(self) -> dict:
        return self._boundary_prescription()

    @computed_field
    @property
    def bx1_outer_expressions(self) -> dict:
        return self._boundary_prescription()

    @computed_field
    @property
    def bx2_inner_expressions(self) -> dict:
        return self._boundary_prescription()

    @computed_field
    @property
    def bx2_outer_expressions(self) -> dict:
        return self._boundary_prescription()

    def initial_primitive_state(self) -> InitialStateType:
        eq = self.equilibrium()
        nr, npolar = self.resolution
        (rmin, rmax) = self.bounds[0]
        (tmin, tmax) = self.bounds[1]
        q = (rmax / rmin) ** (1.0 / nr)
        dth = (tmax - tmin) / npolar

        def gas_state() -> GasStateGenerator:
            for jj in range(npolar):
                theta = tmin + (jj + 0.5) * dth
                for ii in range(nr):
                    rl = rmin * q**ii
                    rh = rl * q
                    r = 0.75 * (rh**4 - rl**4) / (rh**3 - rl**3)
                    rho, v_r, vphi, pre = eq.primitive(r, theta)
                    yield (rho, v_r, 0.0, vphi, pre)

        return gas_state
