# =============================================================================
# gr_fishbone_moncrief.py
#
# fishbone-moncrief (1976) rotating torus around a black hole — the exact
# stationary GRHD equilibrium with angular momentum, initialized on the 2D
# (r, theta) grid with the lifted azimuthal momentum (the `_sph_swirl` kernels).
# the canonical spinning-hole initial data: the same construction at a != 0
# seeds Kerr runs.
#
# floor-less caveats: the a = 0 torus is intrinsically cold (p/rho ~ 4e-3 in the
# core at the paper parameters; hotter placements unbind the equipotential), so
# its surface cells sit close to the physicality boundary tau + D >=
# sqrt(D^2 + |S|^2). the warm hydrostatic corona (h alpha = const, an exact
# equilibrium) sets the pressure-matched surface cut and with it the coldest
# surviving cell's thermal margin. the reflecting theta walls are consistent
# with the corona but not with the torus stratification. a narrow polar cutout
# avoids the spherical coordinate singularity while retaining nearly the full
# meridional domain; the torus surface stays off those walls.
#
# the solution (fm 1976; G = c = 1): constant specific angular momentum
# l = u^t u_phi, u^r = u^theta = 0, and the enthalpy potential ln h(r, theta).
# written here at a = 0 (boyer-lindquist == schwarzschild coordinates), where
# with f = 1 - 2M/r and g_pp = r^2 sin^2(theta):
#
#   chi           = 1 + 4 l^2 f / (r^2 sin^2 theta)
#   ln h(r,theta) = (1/2) ln[(1 + sqrt(chi))/f] - (1/2) sqrt(chi)  -  [at (r_in, pi/2)]
#   l             = kappa^{1/2} sqrt(M r_in^3) / (r_in - 3M)    (eqs. 3.8-3.9)
#   (u^t)^2       = [1 + sqrt(chi)] / (2 f)            (normalization with u_phi u^t = l)
#   v^phi         = l / (alpha g_pp (u^t)^2)           (valencia contravariant velocity)
#
# inside the torus (ln h > 0) the polytrope p = K rho^gamma closes the state via
# h = 1 + gamma/(gamma-1) K rho^(gamma-1), with K normalized so rho(r_max) = rho_max.
# outside sits the warm corona; the interface is a pressure-matched contact.
#
# usage:
#   torus = FishboneMoncrief(mass=1.0, r_in=6.0, gamma=4/3, rho_max=1.0, kappa=1.01)
#   rho, v_r, vphi, pre = torus.primitive(r, theta)   (disk point; None outside)
#   simbi run gr_fishbone_moncrief.py
# =============================================================================

import math
from typing import Annotated

from pydantic import model_validator

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Spacetime,
)
from simbi.types.typing import GasStateGenerator, InitialStateType

from simbi_configs.helpers.fishbone_moncrief import FishboneMoncrief


class GrFishboneMoncrief(SimbiProblem):
    """the fishbone-moncrief torus on a spherical black-hole grid."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.KERR_KS, description="background spacetime"),
    ]
    schwarzschild_mass: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1)"),
    ]
    r_in: Annotated[
        float,
        ProblemParam(6.0, cli=True, description="torus inner edge (equator)"),
    ]
    kappa: Annotated[
        float,
        ProblemParam(
            1.01,
            cli=True,
            description="FM angular-momentum parameter l = kappa^{1/2} l(r_in)",
        ),
    ]
    rho_torus_max: Annotated[
        float, ProblemParam(1.0, description="torus density normalization")
    ]
    # dilute hydrostatic isentropic corona outside the torus surface: relativistic
    # hydrostatic equilibrium on a static metric is h(r) alpha(r) = const, which the
    # well-balanced discrete pressure form holds quietly — no cold supersonic infall
    # to break the c2p at the torus skin. dynamically irrelevant to the torus core
    # (density contrast ~ 1e-5).
    # the corona must be warm: a stationary cell's distance from the physicality
    # boundary tau + D >= sqrt(D^2 + |S|^2) is its thermal margin ~ p/(tau + D), and
    # the discrete stationarity residual erodes that margin at a constant rate — a
    # cold surface cell (margin ~ 1e-4) dies in ~15 M, a warm one holds. the corona
    # pressure sets the pressure-matched cut, so it directly sets the margin of the
    # coldest surviving torus cell.
    atm_rho: Annotated[
        float,
        ProblemParam(1.0e-3, description="corona rest-mass density at r_max"),
    ]
    atm_pre_frac: Annotated[
        float,
        ProblemParam(0.5, description="corona p/rho at r_max (warm, HSE-supported)"),
    ]

    # domain: radial from outside the horizon through the torus, theta wedge wide
    # enough for the torus vertical extent (poles excluded).
    nr: Annotated[int, ProblemParam(192, cli=True, description="radial resolution")]
    npolar: Annotated[
        int, ProblemParam(128, cli=True, description="polar (theta) resolution")
    ]
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((0, 0), description="grid resolution (nr, npolar) — computed"),
    ]
    theta_cut: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            description="polar-axis cutout angle in radians",
        ),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 0.0), (0.0, 0.0)], description="domain bounds — computed"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.SPHERICAL, description="coordinate system"),
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LOG, description="log-spaced radial zones"),
    ]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.REFLECTING,
                BoundaryCondition.REFLECTING,
            ],
            description="boundary conditions (r inner, r outer, theta lo, theta hi)",
        ),
    ]

    end_time: Annotated[
        float,
        ProblemParam(
            400.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time (~ one orbit at r_max)",
        ),
    ]

    @model_validator(mode="after")
    def compute_defaults(self) -> "GrFishboneMoncrief":
        if not 0.0 < self.theta_cut < math.pi / 2.0:
            raise ValueError("theta_cut must lie strictly between 0 and pi/2")
        self.resolution = (self.nr, self.npolar)
        # the horizon-penetrating chart at every spin (a = 0 is the schwarzschild kerr-schild
        # metric), with the inner boundary below r_+ = M + sqrt(M^2 - a^2): the through-horizon
        # inflow is supersonic there, so the inner ghosts are causally decoupled from the torus.
        mm = self.schwarzschild_mass
        r_plus = mm + math.sqrt(max(mm * mm - self.kerr_spin**2, 0.0))
        r_lo = 0.85 * r_plus
        self.bounds = [
            (r_lo, 100.0),
            (self.theta_cut, math.pi - self.theta_cut),
        ]
        return self

    def torus(self) -> FishboneMoncrief:
        return FishboneMoncrief(
            mass=self.schwarzschild_mass,
            r_in=self.r_in,
            gamma=self.adiabatic_index,
            rho_max=self.rho_torus_max,
            kappa=self.kappa,
            spin=self.kerr_spin,
            chart="ks",
        )

    def atmosphere(self, r: float, r_ref: float) -> tuple[float, float]:
        """(rho, p) of the isentropic hydrostatic corona at radius r: the relativistic
        HSE invariant h(r) alpha(r) = const with its own polytropic K, normalized to
        (atm_rho, atm_pre_frac * atm_rho) at the reference radius (the torus pressure
        maximum)."""
        gm = self.adiabatic_index
        gm1 = gm - 1.0
        mm = self.schwarzschild_mass
        h_ref = 1.0 + gm / gm1 * self.atm_pre_frac
        # the static-observer redshift sqrt(-g_tt) = sqrt(1 - 2Mr/Sigma); on the
        # equator this is sqrt(1 - 2M/r) at every spin. no static observers exist
        # inside the ergosphere (r < 2M equatorially) — floor the factor there:
        # that corona gas sits at/inside the horizon and free-falls regardless.
        redshift = lambda rr: math.sqrt(max(1.0 - 2.0 * mm / rr, 1.0e-2))
        hs = h_ref * redshift(r_ref)
        k_atm = gm1 * (h_ref - 1.0) / (gm * self.atm_rho**gm1)
        h = hs / redshift(r)
        rho = (gm1 * (h - 1.0) / (gm * k_atm)) ** (1.0 / gm1)
        return rho, k_atm * rho**gm

    def initial_primitive_state(self) -> InitialStateType:
        torus = self.torus()
        omega_max = math.sqrt(self.schwarzschild_mass / torus.r_max**3)
        print(
            f"fm torus: l = {torus.ell:.4f}, K = {torus.kk:.5e}, "
            f"orbital period at r_max ~ {2.0 * math.pi / omega_max:.1f} M"
        )
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
                    rho_a, pre_a = self.atmosphere(r, torus.r_max)
                    state = torus.primitive(r, theta)
                    # pressure-matched surface: the corona replaces torus gas whose
                    # polytropic pressure falls below the local corona pressure. the
                    # interface is a pure contact (continuous p, a static-corona slip
                    # in v^phi that HLLE handles) — no crush wave, and the c2p-fragile
                    # p/rho -> 0 polytropic sliver never enters the grid.
                    if state is None or state[3] < pre_a:
                        yield (rho_a, 0.0, 0.0, 0.0, pre_a)
                    else:
                        rho, v_r, vphi, pre = state
                        yield (rho, v_r, 0.0, vphi, pre)

        return gas_state
