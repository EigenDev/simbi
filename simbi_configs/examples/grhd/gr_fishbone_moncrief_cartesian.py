# =============================================================================
# gr_fishbone_moncrief_cartesian.py
#
# the fishbone-moncrief torus on the full 3D cartesian kerr-schild grid, at
# general spin: the same exact equilibrium as the spherical-chart version, but
# with no polar axis anywhere — the torus resolves its poles like any other
# direction, and the funnel region above the hole is ordinary grid. the
# horizon-penetrating chart carries the flow smoothly through r = r_+; the
# region inside the horizon is level-set-excised (r_ks < r_exc, the sphere at
# a = 0 and the oblate spheroid at spin — every excised cell is overwritten
# each step with a zero-gradient copy of its outward neighbor, numerical
# padding the exterior never sees).
#
# the cartesian kerr-schild position map: the KS radius solves the
# oblate-spheroidal quartic r^2 = (R^2 - a^2)/2 + sqrt(((R^2 - a^2)/2)^2 +
# a^2 z^2) (R = |x|; r = R at a = 0), theta = arccos(z/r). the valencia
# velocity is the KS-chart torus solution mapped by the chart jacobian
#   d x^i/d r   = ((r x + a y)/(r^2 + a^2), (r y - a x)/(r^2 + a^2), z/r)
#   d x^i/d phi = (-y, x, 0)
# (the r-column is exactly the kerr-schild null-vector direction l^i; a = 0
# reduces to x^i/r). outside the pressure-matched torus surface sits the
# quiescent power-law atmosphere (rho ~ r^{-3/2}, p ~ r^{-5/2}, the free-fall
# scalings), cold enough that the pressure match never eats into the torus.
#
# usage:
#   simbi run gr_fishbone_moncrief_cartesian --resolution 128,128,128
#   simbi run gr_fishbone_moncrief_cartesian --excision-radius 1.2
# =============================================================================
import math
from pathlib import Path
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver, Spacetime
from simbi.types.typing import GasStateGenerator, InitialStateType

from simbi_configs.helpers.fishbone_moncrief import FishboneMoncrief


class GrFishboneMoncriefCartesian(SimbiProblem):
    """the fishbone-moncrief torus on the pole-free 3d cartesian kerr-schild grid."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(
            Spacetime.SCHWARZSCHILD_KS,
            description="horizon-penetrating cartesian kerr-schild background; "
            "setup() promotes it to KERR when kerr_spin != 0",
        ),
    ]
    schwarzschild_mass: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1)"),
    ]
    kerr_spin: Annotated[
        float,
        ProblemParam(
            0.0,
            cli=True,
            description="black-hole spin a = J/M, |a| < M, about +z. nonzero spin "
            "selects the spinning cartesian kerr-schild metric, the FM torus at "
            "that spin, and the oblate-spheroidal excision surface",
        ),
    ]
    r_in: Annotated[
        float,
        ProblemParam(
            8.0,
            cli=True,
            description="torus inner edge (equator). 8M gives the compact torus "
            "(pressure maximum ~10.8M, density < 1e-2 rho_max beyond 13M) that fits "
            "the default 20M box with a wide corona margin; smaller r_in grows the "
            "torus rapidly (6M puts 0.85 rho_max ON the box face)",
        ),
    ]
    kappa: Annotated[
        float,
        ProblemParam(
            1.01,
            cli=True,
            description="FM angular-momentum parameter l = kappa^{1/2} l(r_in). the "
            "torus geometry is a strong function of (kappa, spin): 1.01 gives the "
            "compact a = 0 torus; at a ~ 0.9 use ~1.15 with r_in = 6 (the same "
            "kappa collapses to a sliver at spin) — setup() rejects unresolvable "
            "or box-overflowing tori loudly",
        ),
    ]
    rho_torus_max: Annotated[
        float, ProblemParam(1.0, description="torus density normalization")
    ]
    atm_rho: Annotated[
        float,
        ProblemParam(1.0e-4, description="corona rest-mass density at r_max"),
    ]
    atm_pre_frac: Annotated[
        float,
        ProblemParam(
            0.01,
            description="corona p/rho at r_max. the corona pressure MUST sit well "
            "below the torus peak pressure (the pressure-matched surface replaces "
            "any torus gas colder than the local corona — a warm corona SWALLOWS a "
            "near-marginal thin torus whole); setup() enforces the margin loudly",
        ),
    ]
    excision_radius: Annotated[
        float,
        ProblemParam(
            -1.0,
            cli=True,
            description="horizon-excision KS radius, strictly inside (M/2, r_+) "
            "with r_+ = M + sqrt(M^2 - a^2); the excised surface is the r_ks = "
            "r_exc level set (a sphere at a = 0, an oblate spheroid at spin). "
            "negative = auto (0.7 r_+, the recommended surface: a wider live "
            "band of near-vacuum supersonic infall between the surface and the "
            "horizon is the stiffest gas in the domain); 0 disables excision",
        ),
    ]
    half_width: Annotated[
        float,
        ProblemParam(
            20.0, cli=True, description="cube half-width in M (domain [-L, L]^3)"
        ),
    ]

    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((128, 128, 128), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]] | None,
        ProblemParam(None, description="domain bounds (computed from half_width)"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="GR hydro")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, description="solver")]
    boundary_conditions: Annotated[
        BoundaryCondition,
        ProblemParam(BoundaryCondition.OUTFLOW, description="boundary conditions"),
    ]
    cfl_number: Annotated[float, ProblemParam(0.3, description="cfl number")]

    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/grhd/fm_cartesian/"),
            cli=True,
            checkpoint_safe=True,
            description="output directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            300.0,
            cli=True,
            checkpoint_safe=True,
            description="end time (~1 orbit at the pressure maximum)",
        ),
    ]
    checkpoint_interval: Annotated[
        float,
        ProblemParam(
            25.0, cli=True, checkpoint_safe=True, description="checkpoint interval"
        ),
    ]

    def setup(self) -> None:
        super().setup()
        if self.bounds is None:
            ll = self.half_width
            self.bounds = [(-ll, ll), (-ll, ll), (-ll, ll)]
        # the spinning chart is a different metric family (non-diagonal gamma with
        # the frame-dragging swirl of l); the spacetime tag follows the spin knob.
        if self.kerr_spin != 0.0:
            self.spacetime = Spacetime.KERR_KS
        # auto excision surface: 0.7 r_+ keeps the live band between the surface
        # and the horizon narrow (the gas there is near-vacuum supersonic infall,
        # the stiffest cells in the domain; a wide band collapses dt).
        if self.excision_radius < 0.0:
            mm, aa = self.schwarzschild_mass, self.kerr_spin
            self.excision_radius = 0.7 * (mm + math.sqrt(max(mm * mm - aa * aa, 0.0)))
        # the pressure-matched surface replaces torus gas colder than the corona,
        # so a corona pressure approaching the torus peak pressure erases the torus
        # entirely (a near-marginal kappa ~ 1 torus is nearly pressureless: p_max =
        # (h_max - 1) rho_max (gamma-1)/gamma with h_max - 1 ~ 1e-4). fail loud.
        torus = self.torus()
        p_torus_max = torus.primitive(torus.r_max, math.pi / 2.0)[3]
        p_corona = self.atm_pre_frac * self.atm_rho
        if p_corona >= 0.5 * p_torus_max:
            raise ValueError(
                f"the corona pressure {p_corona:.3e} (atm_pre_frac * atm_rho) is not "
                f"well below the torus peak pressure {p_torus_max:.3e}: the "
                "pressure-matched surface would replace the torus with corona. "
                "lower atm_pre_frac/atm_rho or thicken the torus (raise kappa)."
            )
        # the torus geometry is a strong function of (r_in, kappa, spin): the same
        # kappa that gives a healthy zero-spin torus collapses to a sub-cell sliver
        # at high spin (r_out - r_in < dx), and a slightly larger kappa at a = 0
        # overflows the box. scan the equatorial outer edge and fail loud on both.
        r_out = torus.r_max
        rr = torus.r_max
        while rr < 3.0 * self.half_width:
            if torus.primitive(rr, math.pi / 2.0) is None:
                break
            r_out = rr
            rr += 0.05
        dx_min = 2.0 * self.half_width / max(self.resolution)
        if r_out - self.r_in < 4.0 * dx_min:
            raise ValueError(
                f"the torus annulus [{self.r_in:.2f}, {r_out:.2f}] spans "
                f"{(r_out - self.r_in) / dx_min:.1f} cells at this resolution — "
                "unresolvable. thicken it (raise kappa; at high spin the same kappa "
                "gives a much thinner torus) or refine."
            )
        if r_out > 0.9 * self.half_width:
            raise ValueError(
                f"the torus outer edge r_out = {r_out:.1f} reaches the box "
                f"(half_width {self.half_width}); shrink the torus (lower kappa or "
                "raise r_in) or widen the box — a torus surface on the outflow "
                "boundary drives spurious inflow."
            )

    def torus(self) -> FishboneMoncrief:
        # the torus solution is evaluated in the horizon-penetrating KS chart at
        # the configured spin (the drift v^r rides every radius).
        return FishboneMoncrief(
            mass=self.schwarzschild_mass,
            r_in=self.r_in,
            gamma=self.adiabatic_index,
            rho_max=self.rho_torus_max,
            kappa=self.kappa,
            spin=self.kerr_spin,
            chart="ks",
        )

    def ks_radius(self, x: float, y: float, z: float) -> float:
        # the kerr-schild radius at a cartesian point: the oblate-spheroidal
        # quartic root r^2 = (R^2 - a^2)/2 + sqrt(((R^2 - a^2)/2)^2 + a^2 z^2);
        # |x| at a = 0.
        a = self.kerr_spin
        rr2 = x * x + y * y + z * z
        d = 0.5 * (rr2 - a * a)
        return math.sqrt(max(d + math.sqrt(d * d + (a * z) ** 2), 1.0e-20))

    def atmosphere(self, r: float, r_ref: float) -> tuple[float, float]:
        """(rho, p) of the quiescent power-law atmosphere at radius r:
        rho = atm_rho (r/r_ref)^(-3/2), p = atm_pre_frac atm_rho (r/r_ref)^(-5/2),
        normalized at the torus pressure maximum. NOT an equilibrium — a cold
        atmosphere has no hydrostatic solution (the relativistic HSE invariant
        h alpha = const drops below h = 1 at finite radius when the reference
        enthalpy is small, and anchoring it further out piles up an enormous
        near-hole contrast); the standard quiescent medium simply free-falls
        into the excised hole within a dynamical time. the -3/2 / -5/2 slopes
        are the zero-energy free-fall scalings, so the infalling profile is
        roughly self-similar."""
        q = max(r, 2.0 * self.schwarzschild_mass) / r_ref
        rho = self.atm_rho * q ** (-1.5)
        return rho, self.atm_pre_frac * self.atm_rho * q ** (-2.5)

    def initial_primitive_state(self) -> InitialStateType:
        torus = self.torus()
        nx, ny, nz = self.resolution
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = self.bounds
        dx = (xhi - xlo) / nx
        dy = (yhi - ylo) / ny
        dz = (zhi - zlo) / nz

        aa = self.kerr_spin

        def gas_state() -> GasStateGenerator:
            for kk in range(nz):
                z = zlo + (kk + 0.5) * dz
                for jj in range(ny):
                    y = ylo + (jj + 0.5) * dy
                    for ii in range(nx):
                        x = xlo + (ii + 0.5) * dx
                        r = max(self.ks_radius(x, y, z), 1.0e-10)
                        theta = math.acos(max(-1.0, min(1.0, z / r)))
                        rho_a, pre_a = self.atmosphere(r, torus.r_max)
                        state = torus.primitive(r, theta)
                        # pressure-matched surface: the corona replaces torus gas
                        # whose polytropic pressure falls below the local corona
                        # pressure (a pure contact interface, no crush wave).
                        if state is None or state[3] < pre_a:
                            # the atmosphere free-falls in the KS chart; leaving it
                            # at rest is the standard quiescent start (the infall
                            # develops within a dynamical time and drains into the
                            # excised hole).
                            yield (rho_a, 0.0, 0.0, 0.0, pre_a)
                        else:
                            rho, v_r, vphi, pre = state
                            # jacobian map of the contravariant valencia components:
                            # dr -> ((r x + a y)/(r^2 + a^2), (r y - a x)/(r^2 + a^2), z/r)
                            # (the kerr-schild l direction; x/r at a = 0) and
                            # dphi -> (-y, x, 0).
                            den = 1.0 / (r * r + aa * aa)
                            vx = v_r * (r * x + aa * y) * den - y * vphi
                            vy = v_r * (r * y - aa * x) * den + x * vphi
                            vz = v_r * z / r
                            yield (rho, vx, vy, vz, pre)

        return gas_state
