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
#   rho, v_r, vphi, pre = torus.primitive(r, theta)   (disk point; None outside)
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
    """the exact fishbone-moncrief disk at GENERAL spin (FM 1976, eqs. 3.3/3.6/3.8).

    boyer-lindquist quantities (G = c = 1): Delta = r^2 - 2Mr + a^2, Sigma = r^2 +
    a^2 cos^2(theta), A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta), and the locally
    nonrotating frame (LNRF) functions e^{2 nu} = Sigma Delta / A, e^{2 psi} =
    (A/Sigma) sin^2(theta), omega = 2 a M r / A. the constant of the motion is
    l = u^t u_phi (angular momentum per unit inertial mass); the potential is

      ln h = (1/2) ln[(1 + sqrt(1 + X)) / (Sigma Delta / A)] - (1/2) sqrt(1 + X)
             - 2 a M r l / A  -  [the same at (r_in, pi/2)],
      X    = 4 l^2 Sigma^2 Delta / (A sin(theta))^2,

    with l = kappa^{1/2} l(r_in) from the circular-orbit relation (eq. 3.8, the
    prograde/retrograde branch). the azimuthal 4-velocity follows from eq. 3.3:
    u_(phi)^2 = [sqrt(1 + X) - 1]/2 in the LNRF, u^t = e^{-nu} sqrt(1 + u_(phi)^2),
    u^phi = omega u^t + e^{-psi} u_(phi).

    `chart` selects the code primitives: "bl" (a = 0 only — the schwarzschild grid)
    gives v^r = 0; "ks" (the horizon-penetrating kerr grid) gives the orbiter's
    drift against the infalling eulerian observers, v^r = beta^r/alpha =
    b/sqrt(1 + b) with b = 2 M r / Sigma, and v^phi = u^phi sqrt(1 + b) / u^t.
    a purely azimuthal flow (u^r = 0) carries IDENTICAL 4-velocity components in
    both charts (the BL -> KS shift adds only u^r-proportional terms).

    `primitive(r, theta)` returns (rho, v^r, v^phi, p) inside the disk, None
    outside. certified against the paper's printed disks: fig. 2 schwarzschild
    (r_in = 6, kappa = 1.01) and extreme-kerr corotating (kappa = 1.411698), and
    fig. 3 extreme-kerr corotating (r_in = 2.78) and counterrotating (r_in = 7.75).
    """

    def __init__(
        self,
        mass: float,
        r_in: float,
        gamma: float,
        rho_max: float,
        kappa: float = 1.01,
        spin: float = 0.0,
        prograde: bool = True,
        chart: str = "bl",
    ) -> None:
        if abs(spin) >= mass:
            raise ValueError("kerr requires |a| < M")
        if chart not in ("bl", "ks"):
            raise ValueError("chart must be 'bl' or 'ks'")
        if chart == "bl" and spin != 0.0:
            raise ValueError("the boyer-lindquist chart is wired for a = 0 only")
        if not 1.0 < kappa < 2.0:
            raise ValueError("FM requires 1 < kappa < kappa_max < 2 (eq. 3.9)")
        self.mass = mass
        self.spin = spin
        self.r_in = r_in
        self.gamma = gamma
        self.rho_max = rho_max
        self.chart = chart
        # eq. 3.8: the circular-orbit angular momentum per unit inertial mass on the
        # prograde (upper-sign) / retrograde (lower-sign) branch; l = kappa^{1/2} l(r_in).
        self.prograde = prograde
        self.ell = math.sqrt(kappa) * self._ell_of(r_in)
        # the surface potential: ln h vanishes at the inner edge on the equator.
        self._lnh_in = self._lnh_raw(r_in, math.pi / 2.0)
        # the pressure maximum: the OUTER root of l(r) = l on the chosen branch
        # (|l(r)| falls to its minimum near the marginally stable orbit and grows
        # outward). dense log scan for the outer crossing, then bisection.
        target = abs(self.ell)
        nscan = 4096
        r_hi = 1.0e4 * mass
        rs = [r_in * (r_hi / r_in) ** (ii / (nscan - 1.0)) for ii in range(nscan)]
        vals = [abs(self._ell_of(r)) for r in rs]
        imin = min(range(nscan), key=lambda ii: vals[ii])
        idx = next(
            (ii for ii in range(imin, nscan) if vals[ii] >= target),
            None,
        )
        if idx is None or idx == 0:
            raise ValueError("no pressure maximum inside the scan range")
        lo, hi = rs[idx - 1], rs[idx]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if abs(self._ell_of(mid)) < target:
                lo = mid
            else:
                hi = mid
        self.r_max = 0.5 * (lo + hi)
        # polytropic K from rho(r_max, pi/2) = rho_max.
        h_max = math.exp(self._lnh(self.r_max, math.pi / 2.0))
        gm1 = gamma - 1.0
        self.kk = (h_max - 1.0) * gm1 / (gamma * rho_max**gm1)

    # ---- boyer-lindquist metric functions ----
    def _sdaw(self, r: float, theta: float) -> tuple[float, float, float, float]:
        """(Sigma, Delta, A, omega) at (r, theta)."""
        mm, a = self.mass, self.spin
        st = math.sin(theta)
        ct = math.cos(theta)
        sigma = r * r + a * a * ct * ct
        delta = r * r - 2.0 * mm * r + a * a
        big_a = (r * r + a * a) ** 2 - a * a * delta * st * st
        return sigma, delta, big_a, 2.0 * a * mm * r / big_a

    def _ell_of(self, r: float) -> float:
        """eq. 3.8: the circular-orbit l = u^t u_phi at radius r, on the chosen branch."""
        mm, a = self.mass, self.spin
        sq = math.sqrt(mm * r)
        sgn = 1.0 if self.prograde else -1.0
        num = r**4 + (r * a) ** 2 - 2.0 * mm * r * a * a - sgn * a * sq * (r * r - a * a)
        den = r * r - 3.0 * mm * r + sgn * 2.0 * a * sq
        return sgn * math.sqrt(mm / r**3) * num / den

    def _lnh_raw(self, r: float, theta: float) -> float:
        sigma, delta, big_a, _ = self._sdaw(r, theta)
        st = math.sin(theta)
        xx = 4.0 * (self.ell * sigma) ** 2 * delta / (big_a * st) ** 2
        sq = math.sqrt(1.0 + xx)
        return (
            0.5 * math.log((1.0 + sq) / (sigma * delta / big_a))
            - 0.5 * sq
            - 2.0 * self.spin * self.mass * r * self.ell / big_a
        )

    def _lnh(self, r: float, theta: float) -> float:
        return self._lnh_raw(r, theta) - self._lnh_in

    def _four_velocity(self, r: float, theta: float) -> tuple[float, float]:
        """(u^t, u^phi) of the constant-l azimuthal flow (LNRF projections, eq. 3.3)."""
        sigma, delta, big_a, omega = self._sdaw(r, theta)
        st = math.sin(theta)
        xx = 4.0 * (self.ell * sigma) ** 2 * delta / (big_a * st) ** 2
        u_lnrf = math.copysign(
            math.sqrt(0.5 * (math.sqrt(1.0 + xx) - 1.0)), self.ell
        )
        e_nu = math.sqrt(sigma * delta / big_a)
        e_psi = math.sqrt(big_a / sigma) * st
        u_t = math.sqrt(1.0 + u_lnrf * u_lnrf) / e_nu
        u_p = omega * u_t + u_lnrf / e_psi
        return u_t, u_p

    def azimuthal_velocity(self, r: float, theta: float) -> float:
        """the valencia contravariant v^phi in the configured chart — a GLOBAL smooth
        field (it depends only on l and the metric), defined outside the disk too."""
        u_t, u_p = self._four_velocity(r, theta)
        if self.chart == "bl":
            # a = 0: alpha = e^nu, zero shift.
            sigma, delta, big_a, _ = self._sdaw(r, theta)
            alpha = math.sqrt(sigma * delta / big_a)
            return u_p / (alpha * u_t)
        # ks: alpha = 1/sqrt(1 + b), zero azimuthal shift; the 4-velocity components
        # transfer verbatim from BL (u^r = 0).
        b = 2.0 * self.mass * r / (r * r + (self.spin * math.cos(theta)) ** 2)
        return u_p * math.sqrt(1.0 + b) / u_t

    def radial_velocity(self, r: float, theta: float) -> float:
        """the valencia contravariant v^r in the configured chart: zero in BL; the
        orbiter's drift against the infalling eulerian observers in KS,
        v^r = beta^r / alpha = b / sqrt(1 + b)."""
        if self.chart == "bl":
            return 0.0
        b = 2.0 * self.mass * r / (r * r + (self.spin * math.cos(theta)) ** 2)
        return b / math.sqrt(1.0 + b)

    def primitive(
        self, r: float, theta: float
    ) -> Optional[tuple[float, float, float, float]]:
        """(rho, v^r, v^phi, p) inside the disk; None outside (atmosphere region)."""
        if r < self.r_in:
            return None
        lnh = self._lnh(r, theta)
        if lnh <= 0.0:
            return None
        gm1 = self.gamma - 1.0
        rho = ((math.exp(lnh) - 1.0) * gm1 / (self.gamma * self.kk)) ** (1.0 / gm1)
        pre = self.kk * rho**self.gamma
        return (
            rho,
            self.radial_velocity(r, theta),
            self.azimuthal_velocity(r, theta),
            pre,
        )


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
        spin: float = 0.0,
        chart: str = "bl",
    ) -> None:
        # a stationary constant-l azimuthal flow needs a timelike LNRF (Delta > 0):
        # the equilibrium exists OUTSIDE the horizon only. the wedge must not
        # penetrate — unlike the infall problems, this state has no through-horizon
        # continuation.
        r_plus = mass + math.sqrt(max(mass * mass - spin * spin, 0.0))
        if bounds_r[0] <= r_plus:
            raise ValueError(
                f"the rotating equilibrium requires r_lo > r_plus = {r_plus:.3f}"
            )
        # place the potential's pressure maximum at r_max: with r_in = 0.9 r_max the
        # required kappa is (l(r_max)/l(r_in))^2 (both on the outer branch of the
        # eq. 3.8 circular-orbit relation at THIS spin — a probe instance supplies it).
        probe = FishboneMoncrief(
            mass=mass, r_in=0.9 * r_max, gamma=gamma, rho_max=1.0,
            kappa=1.01, spin=spin, chart=chart,
        )
        self.fm = FishboneMoncrief(
            mass=mass,
            r_in=0.9 * r_max,
            gamma=gamma,
            rho_max=1.0,
            kappa=(probe._ell_of(r_max) / probe._ell_of(0.9 * r_max)) ** 2,
            spin=spin,
            chart=chart,
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

    def primitive(self, r: float, theta: float) -> tuple[float, float, float, float]:
        """(rho, v^r, v^phi, p) of the equilibrium — defined everywhere in the domain."""
        gm1 = self.gamma - 1.0
        h = math.exp(self.fm._lnh_raw(r, theta) - self.cc)
        rho = ((h - 1.0) * gm1 / (self.gamma * self.kk)) ** (1.0 / gm1)
        return (
            rho,
            self.fm.radial_velocity(r, theta),
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
        if self.kerr_spin != 0.0:
            # spinning: the kerr spacetime on the horizon-penetrating chart, inner
            # boundary below r_plus = M + sqrt(M^2 - a^2) (supersonic through-horizon
            # inflow decouples the inner ghosts).
            self.spacetime = Spacetime.KERR
            mm = self.schwarzschild_mass
            r_plus = mm + math.sqrt(max(mm * mm - self.kerr_spin**2, 0.0))
            r_lo = 0.85 * r_plus
        else:
            r_lo = 3.0
        self.bounds = [
            (r_lo, 100.0),
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
            spin=self.kerr_spin,
            chart="ks" if self.kerr_spin != 0.0 else "bl",
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
        # equator this is sqrt(1 - 2M/r) at EVERY spin. no static observers exist
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
                    # PRESSURE-MATCHED surface: the corona replaces torus gas whose
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
