# =============================================================================
# gr_fishbone_moncrief_cartesian.py
#
# the fishbone-moncrief torus on the FULL 3D CARTESIAN kerr-schild grid, at
# GENERAL SPIN: the same exact equilibrium as the spherical-chart version, but
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
# reduces to x^i/r). outside the pressure-matched torus surface sits the warm
# isentropic hydrostatic corona (h alpha = const), the same construction as
# the spherical chart.
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

from simbi_configs.examples.grmhd.gr_fishbone_moncrief import FishboneMoncrief


class GrFishboneMoncriefCartesian(SimbiProblem):
    """the fishbone-moncrief torus on the pole-free 3d cartesian kerr-schild grid."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(
            Spacetime.KERR_SCHILD,
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
            description="FM angular-momentum parameter l = kappa^{1/2} l(r_in)",
        ),
    ]
    rho_torus_max: Annotated[
        float, ProblemParam(1.0, description="torus density normalization")
    ]
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
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
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
            self.spacetime = Spacetime.KERR
        # auto excision surface: 0.7 r_+ keeps the live band between the surface
        # and the horizon narrow (the gas there is near-vacuum supersonic infall,
        # the stiffest cells in the domain; a wide band collapses dt).
        if self.excision_radius < 0.0:
            mm, aa = self.schwarzschild_mass, self.kerr_spin
            self.excision_radius = 0.7 * (mm + math.sqrt(max(mm * mm - aa * aa, 0.0)))

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
        """(rho, p) of the isentropic hydrostatic corona at radius r: the relativistic
        HSE invariant h(r) alpha(r) = const, normalized to (atm_rho, atm_pre_frac *
        atm_rho) at the torus pressure maximum. the static-observer redshift factor
        is floored inside the ergosphere/horizon — that gas free-falls (and is
        excised deep inside) regardless. the a = 0 redshift form is kept at spin:
        the corona is a pressure-matched heuristic and the spin correction to the
        static redshift is O(a^2 / r^2) at torus radii."""
        gm = self.adiabatic_index
        gm1 = gm - 1.0
        mm = self.schwarzschild_mass
        h_ref = 1.0 + gm / gm1 * self.atm_pre_frac
        redshift = lambda rr: math.sqrt(max(1.0 - 2.0 * mm / rr, 1.0e-2))
        hs = h_ref * redshift(r_ref)
        k_atm = gm1 * (h_ref - 1.0) / (gm * self.atm_rho**gm1)
        h = hs / redshift(r)
        rho = (gm1 * (h - 1.0) / (gm * k_atm)) ** (1.0 / gm1)
        return rho, k_atm * rho**gm

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
                        # PRESSURE-MATCHED surface: the corona replaces torus gas
                        # whose polytropic pressure falls below the local corona
                        # pressure (a pure contact interface, no crush wave).
                        if state is None or state[3] < pre_a:
                            # the corona free-falls in the KS chart at the eulerian
                            # drift; leaving it at rest is the standard quiescent
                            # start (the drift develops within a light-crossing).
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
