# =============================================================================
# gr_fishbone_moncrief_cartesian.py
#
# the fishbone-moncrief torus on the FULL 3D CARTESIAN kerr-schild grid: the
# same exact equilibrium as the spherical-chart version, but with no polar
# axis anywhere — the torus resolves its poles like any other direction, and
# the funnel region above the hole is ordinary grid. the horizon-penetrating
# chart carries the flow smoothly through r = 2M; the sphere inside the
# horizon is SDF-excised (every excised cell is overwritten each step with a
# zero-gradient copy of its outward neighbor — numerical padding the exterior
# never sees, since every characteristic inside the horizon points inward).
#
# the valencia velocity is the KS-chart torus solution mapped by the plain
# jacobian: with v^r = 2M/r / sqrt(1 + 2M/r) (the orbiter's drift against the
# infalling eulerian observers) and v^phi from the FM angular momentum,
#   v^x = v^r x/r - y v^phi,  v^y = v^r y/r + x v^phi,  v^z = v^r z/r.
# outside the pressure-matched torus surface sits the warm isentropic
# hydrostatic corona (h alpha = const), the same construction as the
# spherical chart.
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
            description="horizon-penetrating cartesian kerr-schild background (a = 0)",
        ),
    ]
    schwarzschild_mass: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1)"),
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
            1.0,
            cli=True,
            description="horizon-excision sphere radius, strictly inside (M/2, 2M)",
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

    def torus(self) -> FishboneMoncrief:
        # the cartesian chart is kerr-schild at a = 0, so the torus solution is
        # evaluated in the KS chart (the drift v^r rides every radius).
        return FishboneMoncrief(
            mass=self.schwarzschild_mass,
            r_in=self.r_in,
            gamma=self.adiabatic_index,
            rho_max=self.rho_torus_max,
            kappa=self.kappa,
            spin=0.0,
            chart="ks",
        )

    def atmosphere(self, r: float, r_ref: float) -> tuple[float, float]:
        """(rho, p) of the isentropic hydrostatic corona at radius r: the relativistic
        HSE invariant h(r) alpha(r) = const, normalized to (atm_rho, atm_pre_frac *
        atm_rho) at the torus pressure maximum. the static-observer redshift factor
        is floored inside the ergosphere/horizon — that gas free-falls (and is
        excised deep inside) regardless."""
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

        def gas_state() -> GasStateGenerator:
            for kk in range(nz):
                z = zlo + (kk + 0.5) * dz
                for jj in range(ny):
                    y = ylo + (jj + 0.5) * dy
                    for ii in range(nx):
                        x = xlo + (ii + 0.5) * dx
                        r = max(math.sqrt(x * x + y * y + z * z), 1.0e-10)
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
                            # dr -> x/r and dphi -> (-y, x, 0).
                            vx = v_r * x / r - y * vphi
                            vy = v_r * y / r + x * vphi
                            vz = v_r * z / r
                            yield (rho, vx, vy, vz, pre)

        return gas_state
