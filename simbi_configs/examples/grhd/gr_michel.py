# =============================================================================
# gr_michel.py
#
# michel (1972) steady transonic accretion onto a schwarzschild black hole — the
# EXACT GRHD solution, initialized directly on the grid. the flow is isentropic
# (p = K rho^gamma with one global K), so it solves the adiabatic GRHD equations,
# and it is STATIONARY: a correct scheme must hold the profile, with the residual
# shrinking at the truncation order under grid refinement. this makes the config
# the accuracy benchmark for the schwarzschild valencia path (flux densitization,
# geodesic sources, banyuls-font wave speeds), complementing the uniform-gas
# transient in gr_bondi.py which tests development, not accuracy.
#
# the solution (shapiro & teukolsky ch. 14, G = c = 1): two flow invariants,
#   baryon flux      r^2 rho u = jm            (u = |u^r|, proper radial velocity)
#   bernoulli        h^2 (1 - 2M/r + u^2) = h_inf^2
# with h = 1 + gamma/(gamma-1) K rho^(gamma-1). the transonic branch passes through
# the critical point
#   u_s^2 = M / (2 r_s),   a_s^2 = u_s^2 / (1 - 3 u_s^2)
# (a = sound speed), whose location follows from bernoulli evaluated there. inside
# r_s the flow is supersonic (u > u_s), outside subsonic; each radius has one root
# per branch, found by bisection bracketed against u_s.
#
# code variables at radius r (f = 1 - 2M/r): W = sqrt((f + u^2)/f), the valencia
# CONTRAVARIANT radial velocity v^r = u^r / W = -u sqrt(f) / sqrt(f + u^2), and
# p = K rho^gamma. the initial state samples the profile at each cell's
# volume-weighted centroid r_vw = (3/4)(rh^4 - rl^4)/(rh^3 - rl^3) — the same
# radius the backend evaluates the metric at, so the stored conserved state is the
# exact analytic profile at the scheme's own quadrature points.
#
# usage:
#   sol = MichelSolution(mass=1.0, gamma=4/3, rho_inf=1.0, p_inf=1e-2)
#   rho, v1, pre = sol.primitive(r)
#   simbi run gr_michel.py            (the held steady state; outflow both ends)
# =============================================================================

import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Spacetime,
)
from simbi.types.typing import GasStateGenerator, InitialStateType

# bisection iteration count: halves the bracket to ~1e-16 of its width, i.e. full
# double precision for the O(1) brackets used here.
_BISECT_ITERS = 200


def _bisect(func, lo: float, hi: float) -> float:
    """root of func on [lo, hi] by bisection; the bracket endpoints must straddle."""
    flo = func(lo)
    fhi = func(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError(
            f"bisection bracket does not straddle: f({lo}) = {flo}, f({hi}) = {fhi}"
        )
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        fmid = func(mid)
        if fmid == 0.0:
            return mid
        if flo * fmid < 0.0:
            hi = mid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


class MichelSolution:
    """the exact michel transonic accretion profile on schwarzschild.

    construction solves the critical point once; `primitive(r)` then solves the
    bernoulli root at any radius and returns the code primitives (rho, v^r, p)
    with v^r the valencia CONTRAVARIANT radial velocity (negative, inflow).
    """

    def __init__(
        self, mass: float, gamma: float, rho_inf: float, p_inf: float
    ) -> None:
        self.mass = mass
        self.gamma = gamma
        self.rho_inf = rho_inf
        # polytropic constant from the asymptotic state; one global K (isentropic).
        self.kk = p_inf / rho_inf**gamma
        h_inf = 1.0 + gamma / (gamma - 1.0) * p_inf / rho_inf
        self.h_inf_sq = h_inf * h_inf

        # ---- critical point: pin x = u_s^2 from bernoulli at the sonic radius ----
        # a_s^2 = x/(1-3x); h_s = 1/(1 - a_s^2/(gamma-1)); h_s^2 (1-3x) = h_inf^2.
        # the residual is negative as x -> 0 (h_s -> 1 < h_inf) and diverges to
        # +infinity at the enthalpy pole a_s^2 -> gamma-1, so the bracket straddles.
        gm1 = gamma - 1.0

        def sonic_residual(x: float) -> float:
            a_sq = x / (1.0 - 3.0 * x)
            h_s = 1.0 / (1.0 - a_sq / gm1)
            return h_s * h_s * (1.0 - 3.0 * x) - self.h_inf_sq

        x_pole = gm1 / (1.0 + 3.0 * gm1)  # a_s^2 = gamma-1 (h_s -> infinity)
        x = _bisect(sonic_residual, 1e-15, x_pole * (1.0 - 1e-12))

        self.u_sonic = math.sqrt(x)
        self.r_sonic = mass / (2.0 * x)
        a_sq = x / (1.0 - 3.0 * x)
        # invert a^2 = gamma y / (1 + gamma y / (gamma-1)) for y = K rho^(gamma-1).
        y_s = a_sq / (gamma * (1.0 - a_sq / gm1))
        self.rho_sonic = (y_s / self.kk) ** (1.0 / gm1)
        # the baryon-flux invariant per steradian: jm = r^2 rho u along the flow.
        self.jm = self.r_sonic**2 * self.rho_sonic * self.u_sonic

    def _enthalpy(self, rho: float) -> float:
        g = self.gamma
        return 1.0 + g / (g - 1.0) * self.kk * rho ** (g - 1.0)

    def proper_velocity(self, r: float) -> float:
        """u = |u^r| on the transonic branch at radius r."""
        f = 1.0 - 2.0 * self.mass / r

        def bernoulli_residual(u: float) -> float:
            rho = self.jm / (r * r * u)
            h = self._enthalpy(rho)
            return h * h * (f + u * u) - self.h_inf_sq

        if r >= self.r_sonic:
            # subsonic branch: u in (0, u_s). the residual diverges as u -> 0
            # (rho, h -> infinity) and is negative at u_s away from the sonic radius.
            lo, hi = 1e-14, self.u_sonic
        else:
            # supersonic branch: u in (u_s, u_max). at large u the density (and h)
            # drop toward 1 and the residual approaches f + u^2 - h_inf^2 > 0.
            lo, hi = self.u_sonic, math.sqrt(self.h_inf_sq) + 1.0
        # at the sonic radius itself the two roots merge at u_s; the residual there
        # is zero to roundoff and either bracket end returns it.
        if abs(bernoulli_residual(self.u_sonic)) < 1e-13 * self.h_inf_sq:
            return self.u_sonic
        return _bisect(bernoulli_residual, lo, hi)

    def primitive(self, r: float) -> tuple[float, float, float]:
        """(rho, v^r, p) at radius r; v^r is the valencia contravariant velocity."""
        u = self.proper_velocity(r)
        rho = self.jm / (r * r * u)
        f = 1.0 - 2.0 * self.mass / r
        v1 = -u * math.sqrt(f) / math.sqrt(f + u * u)
        return rho, v1, self.kk * rho**self.gamma


class GrMichel(SimbiProblem):
    """the exact michel steady transonic accretion profile, held on the grid."""

    # physics — mirrors gr_bondi.py; the ambient state sets the polytropic K and
    # the asymptotic bernoulli constant.
    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.SCHWARZSCHILD, description="background spacetime"),
    ]
    schwarzschild_mass: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1)"),
    ]
    rho_ambient: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="asymptotic rest-mass density"),
    ]
    p_ambient: Annotated[
        float,
        ProblemParam(
            1.0e-2,
            cli=True,
            description="asymptotic pressure (sets K and the sonic radius)",
        ),
    ]

    # domain — transonic: the sonic radius (~22.7 M at the defaults) is interior,
    # the inner boundary exit is supersonic, the outer boundary inflow subsonic.
    resolution: Annotated[
        int, ProblemParam(256, cli=True, description="radial grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(3.0, 100.0)], description="radial domain bounds (r > 2M)"),
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
            [BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW],
            description="boundary conditions (inner, outer)",
        ),
    ]

    end_time: Annotated[
        float,
        ProblemParam(
            10.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    def michel_solution(self) -> MichelSolution:
        return MichelSolution(
            mass=self.schwarzschild_mass,
            gamma=self.adiabatic_index,
            rho_inf=self.rho_ambient,
            p_inf=self.p_ambient,
        )

    def cell_centroids(self) -> list[float]:
        """volume-weighted cell centroids of the log-spaced radial grid — the same
        radii the backend evaluates the metric at when storing the conserved state."""
        (rmin, rmax) = self.bounds[0]
        n = self.resolution
        q = (rmax / rmin) ** (1.0 / n)
        out = []
        for ii in range(n):
            rl = rmin * q**ii
            rh = rl * q
            out.append(0.75 * (rh**4 - rl**4) / (rh**3 - rl**3))
        return out

    def initial_primitive_state(self) -> InitialStateType:
        sol = self.michel_solution()
        print(
            f"michel sonic radius r_s = {sol.r_sonic:.4f}, "
            f"u_s = {sol.u_sonic:.4f}, jm = r^2 rho u = {sol.jm:.4f}"
        )
        centroids = self.cell_centroids()

        def gas_state() -> GasStateGenerator:
            for r in centroids:
                yield sol.primitive(r)

        return gas_state
