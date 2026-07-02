# =============================================================================
# gr_fishbone_moncrief.py
#
# fishbone-moncrief (1976) rotating torus around a black hole — the exact
# stationary GRHD equilibrium with angular momentum, initialized on the 2D
# (r, theta) grid with the lifted azimuthal momentum (the `_sph_swirl` kernels).
# the canonical spinning-hole initial data: the same construction at a != 0
# seeds Kerr runs.
#
# floor-less caveats: the a = 0 torus is intrinsically COLD (p/rho ~ 4e-3 in the
# core at the paper parameters; hotter placements unbind the equipotential), so
# its surface cells sit close to the physicality boundary tau + D >=
# sqrt(D^2 + |S|^2). the warm hydrostatic corona (h alpha = const, an exact
# equilibrium) sets the pressure-matched surface cut and with it the coldest
# surviving cell's thermal margin. the reflecting theta walls are consistent with
# the corona (theta-independent HSE) but not with the torus stratification —
# tolerable because the torus surface stays off the walls. the precision hold
# gate for the rotating balance is `gr_rotating_equilibrium.py` (surface-free
# constant-l state, all faces DRIVEN); this config is the rotating science
# demonstrator and the Kerr initial data.
#
# the solution (FM 1976; G = c = 1): constant specific angular momentum
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
#   rho, vphi, pre = torus.primitive(r, theta)   (torus point; None outside)
#   simbi run gr_fishbone_moncrief.py
# =============================================================================

import math
from typing import Annotated, Optional

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


class FishboneMoncrief:
    """the exact fishbone-moncrief torus at zero spin (schwarzschild coordinates).

    construction fixes the angular-momentum constant l from the pressure-maximum
    radius and the polytropic K from the density normalization; `primitive(r, theta)`
    returns the torus state (rho, v^phi, p) with v^phi the valencia CONTRAVARIANT
    azimuthal velocity, or None outside the torus surface (ln h <= 0 or r < r_in).
    """

    def __init__(
        self,
        mass: float,
        r_in: float,
        gamma: float,
        rho_max: float,
        kappa: float = 1.01,
    ) -> None:
        if r_in <= 3.0 * mass:
            raise ValueError(
                "the inner edge must sit outside the photon orbit r = 3M"
            )
        if not 1.0 < kappa < 2.0:
            raise ValueError("FM requires 1 < kappa < kappa_max < 2 (eq. 3.9)")
        self.mass = mass
        self.r_in = r_in
        self.gamma = gamma
        self.rho_max = rho_max
        # the FM parametrization (eqs. 3.8-3.9): l = kappa^{1/2} l(r_in), with l(r) the
        # a = 0 circular-orbit angular momentum per unit inertial mass sqrt(M r^3)/(r - 3M).
        # kappa > 1 makes the pressure gradient at the inner edge point outward (a disk);
        # the defaults (r_in = 6, kappa = 1.01) are the paper's fig. 2 schwarzschild disk:
        # (ln h)_max = 0.0153 at r = 16, equatorial extent 6 -> 73.812, minimum polar
        # angle 45.2 deg — the oracle reproduces all of these.
        ell_of = lambda r: math.sqrt(mass * r**3) / (r - 3.0 * mass)
        self.ell = math.sqrt(kappa) * ell_of(r_in)
        # the surface potential: ln h vanishes at the inner edge on the equator.
        self._lnh_in = self._lnh_raw(r_in, math.pi / 2.0)
        # the pressure maximum sits where l equals the local circular-orbit value again:
        # the outer root of ell_of(r) = ell (ell_of has its minimum at r = 6M and grows
        # outward, so bisect on [r_in + margin at the minimum side, far out]).
        lo, hi = max(r_in, 6.0 * mass) * 1.0000001, 1.0e4 * mass
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if ell_of(mid) < self.ell:
                lo = mid
            else:
                hi = mid
        self.r_max = 0.5 * (lo + hi)
        # polytropic K from rho(r_max, pi/2) = rho_max.
        h_max = math.exp(self._lnh(self.r_max, math.pi / 2.0))
        gm1 = gamma - 1.0
        self.kk = (h_max - 1.0) * gm1 / (gamma * rho_max**gm1)

    def _lnh_raw(self, r: float, theta: float) -> float:
        f = 1.0 - 2.0 * self.mass / r
        st = math.sin(theta)
        chi = 1.0 + 4.0 * self.ell**2 * f / (r * st) ** 2
        return 0.5 * math.log((1.0 + math.sqrt(chi)) / f) - 0.5 * math.sqrt(chi)

    def _lnh(self, r: float, theta: float) -> float:
        return self._lnh_raw(r, theta) - self._lnh_in

    def azimuthal_velocity(self, r: float, theta: float) -> float:
        """the valencia contravariant v^phi of the constant-l rotation law — a GLOBAL
        smooth field (it depends only on l and the metric), defined outside the torus
        surface too. the corona sheath co-rotates with it so the torus surface carries
        no velocity jump."""
        f = 1.0 - 2.0 * self.mass / r
        g_pp = (r * math.sin(theta)) ** 2
        chi = 1.0 + 4.0 * self.ell**2 * f / g_pp
        ut_sq = (1.0 + math.sqrt(chi)) / (2.0 * f)
        return self.ell / (math.sqrt(f) * g_pp * ut_sq)

    def primitive(
        self, r: float, theta: float
    ) -> Optional[tuple[float, float, float]]:
        """(rho, v^phi, p) inside the torus; None outside (atmosphere region)."""
        if r < self.r_in:
            return None
        lnh = self._lnh(r, theta)
        if lnh <= 0.0:
            return None
        gm1 = self.gamma - 1.0
        rho = ((math.exp(lnh) - 1.0) * gm1 / (self.gamma * self.kk)) ** (
            1.0 / gm1
        )
        pre = self.kk * rho**self.gamma
        return rho, self.azimuthal_velocity(r, theta), pre


class RotatingEquilibrium:
    """the SURFACE-FREE constant-l rotating equilibrium — the fishbone-moncrief
    potential with the integration constant moved so h > 1 over the whole domain:
    a smooth stationary rotating state filling the grid, no torus surface, no
    atmosphere, fat thermal margins everywhere. the precision hold gate for the
    rotating balance (centrifugal + gravity + pressure, both sweeps); the FM torus
    with its cold surface is the science configuration, not the gate.

    `p_rho_ref` sets p/rho at the COLDEST point of the domain — the integration
    constant is anchored to the domain MINIMUM of the potential (which sits on a
    wedge wall at intermediate radius, NOT at a corner: lnh_raw along a wall is
    non-monotonic in r), so the thermal margin p/(tau + D) is bounded below by
    ~p_rho_ref everywhere. anchoring at a warmer point would leave the potential
    minimum arbitrarily close to the h = 1 equipotential — a thin-margin cell that
    the discrete stationarity residual erodes to the physicality boundary.
    """

    def __init__(
        self,
        mass: float,
        r_max: float,
        gamma: float,
        rho_ref: float,
        p_rho_ref: float,
        bounds_r: tuple[float, float],
        bounds_theta: tuple[float, float],
    ) -> None:
        # place the potential's pressure maximum at r_max: with r_in = 0.9 r_max the
        # required kappa is (l(r_max)/l(r_in))^2 (both on the outer branch of l(r)).
        ell_of = lambda r: math.sqrt(mass * r**3) / (r - 3.0 * mass)
        self.fm = FishboneMoncrief(
            mass=mass,
            r_in=0.9 * r_max,
            gamma=gamma,
            rho_max=1.0,
            kappa=(ell_of(r_max) / ell_of(0.9 * r_max)) ** 2,
        )
        self.gamma = gamma
        # the domain minimum of the potential, by dense sampling (log in r — the
        # minimum tracks the walls at intermediate radius; the theta extreme is a wall).
        nscan = 512
        (r_lo, r_hi) = bounds_r
        lnh_min = min(
            self.fm._lnh_raw(r_lo * (r_hi / r_lo) ** (ii / (nscan - 1.0)), th)
            for ii in range(nscan)
            for th in (bounds_theta[0], math.pi / 2.0, bounds_theta[1])
        )
        h_ref = 1.0 + gamma / (gamma - 1.0) * p_rho_ref
        self.cc = lnh_min - math.log(h_ref)
        gm1 = gamma - 1.0
        self.kk = (h_ref - 1.0) * gm1 / (gamma * rho_ref**gm1)

    def primitive(self, r: float, theta: float) -> tuple[float, float, float]:
        """(rho, v^phi, p) of the equilibrium — defined everywhere in the domain."""
        gm1 = self.gamma - 1.0
        h = math.exp(self.fm._lnh_raw(r, theta) - self.cc)
        rho = ((h - 1.0) * gm1 / (self.gamma * self.kk)) ** (1.0 / gm1)
        return (
            rho,
            self.fm.azimuthal_velocity(r, theta),
            self.kk * rho**self.gamma,
        )


class GrFishboneMoncrief(SimbiProblem):
    """the fishbone-moncrief torus held on the (r, theta) schwarzschild grid."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
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
    # dilute HYDROSTATIC isentropic corona outside the torus surface: relativistic
    # hydrostatic equilibrium on a static metric is h(r) alpha(r) = const, which the
    # well-balanced discrete pressure form holds quietly — no cold supersonic infall
    # to break the c2p at the torus skin. dynamically irrelevant to the torus core
    # (density contrast ~ 1e-5).
    # the corona must be WARM: a stationary cell's distance from the physicality
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
        ProblemParam(
            0.5, description="corona p/rho at r_max (warm, HSE-supported)"
        ),
    ]

    # domain: radial from outside the horizon through the torus, theta wedge wide
    # enough for the torus vertical extent (poles excluded).
    nr: Annotated[
        int, ProblemParam(192, cli=True, description="radial resolution")
    ]
    npolar: Annotated[
        int, ProblemParam(64, cli=True, description="polar (theta) resolution")
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
            1.0, description="half-width of the equatorial theta wedge (rad)"
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
        self.resolution = (self.nr, self.npolar)
        theta_c = math.pi / 2.0
        self.bounds = [
            (3.0, 100.0),
            (theta_c - self.theta_halfwidth, theta_c + self.theta_halfwidth),
        ]
        return self

    def torus(self) -> FishboneMoncrief:
        return FishboneMoncrief(
            mass=self.schwarzschild_mass,
            r_in=self.r_in,
            gamma=self.adiabatic_index,
            rho_max=self.rho_torus_max,
            kappa=self.kappa,
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
        hs = h_ref * math.sqrt(1.0 - 2.0 * mm / r_ref)
        k_atm = gm1 * (h_ref - 1.0) / (gm * self.atm_rho**gm1)
        h = hs / math.sqrt(1.0 - 2.0 * mm / r)
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
                    # PRESSURE-MATCHED surface: the corona replaces torus gas whose
                    # polytropic pressure falls below the local corona pressure. the
                    # interface is a pure contact (continuous p, a static-corona slip
                    # in v^phi that HLLE handles) — no crush wave, and the c2p-fragile
                    # p/rho -> 0 polytropic sliver never enters the grid.
                    if state is None or state[2] < pre_a:
                        yield (rho_a, 0.0, 0.0, 0.0, pre_a)
                    else:
                        rho, vphi, pre = state
                        yield (rho, 0.0, 0.0, vphi, pre)

        return gas_state
