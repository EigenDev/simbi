# =============================================================================
# spb_torus.py
#
# the non-radiative viscous torus of Stone, Pringle & Begelman 1999 (MNRAS 310, 1002;
# astro-ph/9908185): a constant angular momentum polytropic torus about a newtonian point
# mass, GM = R_0 = 1, on the axisymmetric (r, theta) plane with the azimuthal velocity, driven
# by the anomalous azimuthal shear stress
#   T_r_phi = mu r d(v_phi / r)/dr,  T_theta_phi = (mu sin(theta) / r) d(v_phi / sin(theta))/dtheta,
# mu = rho nu, with its heating T^2 / mu in the energy. runs A through K of the paper's table 1:
#   A  nu = 1e-2 rho                64 / decade  R_0/R_B = 100  d = 1.125  gamma = 5/3
#   B  as A at 128 / decade
#   C  as A with R_0/R_B = 200      D  as A with R_0/R_B = 50
#   E  as A with gamma = 4/3, the torus mass of A
#   F  nu = 5e-3 rho, d = 1.042, the torus mass of A    G  as F with gamma = 4/3
#   H  nu = 1e-3 rho                I  nu = 1e-1 rho
#   J  nu = 1e-3                    K  nu = 1e-3 r^(1/2)
#   N  nu = 0 for two orbits: the paper's check that the torus holds its structure, angular
#      momentum and rotational energy without the stress (section 2.3)
# the grid runs from R_B to 4 R_0, logarithmic in radius. the paper refines its polar zones by
# a factor of four toward the equator over 88 zones; this grid is uniform in theta with
# `polar_zones` zones, 192 by default, which matches the paper's equatorial zone width. the
# ambient medium is rho_0 = 1e-4 at rest with p_0 = rho_0 / r. the radial walls take outflow
# conditions, the poles the axis condition; the stress passes no torque through the inner
# wall and the poles, and continues through the outer wall. time is in orbits at R_0, 2 pi.
# the paper's run I' (A restarted with a rising stress) is outside this table.
# `zones_per_decade` replaces the run's radial resolution, and `outward_outer` makes the outer
# wall pass outflow and refuse inflow: its face takes the riemann flux where mass leaves and the
# wall flux (pressure alone) where mass would enter. the paper's outer wall is a plain outflow
# condition, which admits inflow; a run with `outward_outer` departs from the paper there.
#
# usage:
#   simbi run simbi_configs/examples/newtonian/spb_torus.py --run A
#   simbi run simbi_configs/examples/newtonian/spb_torus.py --run K --orbits 4
#   simbi run simbi_configs/examples/newtonian/spb_torus.py --run N --orbits 3 \
#       --zones-per-decade 32 --polar-zones 44 --outward-outer
#   python scripts/spb_torus_analysis.py data/spb_torus/A --from-orbit 3.3
# =============================================================================
import math
from typing import Annotated

from pydantic import computed_field, model_validator

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime
from simbi.types.bodies import BodyCapability, GravitationalProperties, ImmersedBodyConfig
from simbi.types.typing import GasStateGenerator, InitialStateType

ORBIT = 2.0 * math.pi
OUTER_RADIUS = 4.0
AMBIENT_DENSITY = 1.0e-4
FIDUCIAL = ("A", 5.0 / 3.0, 1.125)

# per run: nu scale, nu density power, nu radius power, zones per decade, R_0 / R_B, d, gamma,
# the paper's final time in orbits, and whether the torus mass is held at run A's
RUNS = {
    "A": (1.0e-2, 1.0, 0.0, 64, 100.0, 1.125, 5.0 / 3.0, 4.325, False),
    "B": (1.0e-2, 1.0, 0.0, 128, 100.0, 1.125, 5.0 / 3.0, 2.274, False),
    "C": (1.0e-2, 1.0, 0.0, 64, 200.0, 1.125, 5.0 / 3.0, 1.947, False),
    "D": (1.0e-2, 1.0, 0.0, 64, 50.0, 1.125, 5.0 / 3.0, 4.000, False),
    "E": (1.0e-2, 1.0, 0.0, 64, 100.0, 1.125, 4.0 / 3.0, 2.155, True),
    "F": (5.0e-3, 1.0, 0.0, 64, 100.0, 1.042, 5.0 / 3.0, 3.539, True),
    "G": (5.0e-3, 1.0, 0.0, 64, 100.0, 1.042, 4.0 / 3.0, 1.391, True),
    "H": (1.0e-3, 1.0, 0.0, 64, 100.0, 1.125, 5.0 / 3.0, 20.389, False),
    "I": (1.0e-1, 1.0, 0.0, 64, 100.0, 1.125, 5.0 / 3.0, 0.593, False),
    "J": (1.0e-3, 0.0, 0.0, 64, 100.0, 1.125, 5.0 / 3.0, 5.479, False),
    "K": (1.0e-3, 0.0, 0.5, 64, 100.0, 1.125, 5.0 / 3.0, 13.6181, False),
    "N": (0.0, 0.0, 0.0, 64, 100.0, 1.125, 5.0 / 3.0, 2.0, False),
}


def torus_enthalpy(r: float, theta: float, distortion: float) -> float:
    """(n + 1) P / rho of the constant angular momentum torus, paper eq. 6 times n + 1"""
    big_r = r * math.sin(theta)
    return 1.0 / r - 0.5 / big_r**2 - 0.5 / distortion


def unit_mass(gamma: float, distortion: float) -> float:
    """the torus mass at polytropic constant A = 1: the mass at A scales as A^(-n)"""
    n = 1.0 / (gamma - 1.0)
    total = 0.0
    radii, angles = 1200, 1200
    lo, hi = 0.3, 3.0
    for ii in range(radii):
        r = lo + (ii + 0.5) * (hi - lo) / radii
        for jj in range(angles):
            theta = (jj + 0.5) * math.pi / angles
            w = torus_enthalpy(r, theta, distortion) / (n + 1.0)
            if w > 0.0:
                total += w**n * 2.0 * math.pi * r * r * math.sin(theta)
    return total * (hi - lo) / radii * math.pi / angles


def polytropic_constant(run: str) -> float:
    """A of P = A rho^gamma: rho_max = 1 (paper eq. 7), or run A's torus mass"""
    _, _, _, _, _, distortion, gamma, _, hold_mass = RUNS[run]
    n = 1.0 / (gamma - 1.0)
    if not hold_mass:
        return (1.0 / (n + 1.0)) * (distortion - 1.0) / (2.0 * distortion)
    _, gamma_a, distortion_a = FIDUCIAL
    n_a = 1.0 / (gamma_a - 1.0)
    a_fiducial = (1.0 / (n_a + 1.0)) * (distortion_a - 1.0) / (2.0 * distortion_a)
    target = a_fiducial ** (-n_a) * unit_mass(gamma_a, distortion_a)
    return (unit_mass(gamma, distortion) / target) ** (1.0 / n)


class SpbTorus(SimbiProblem):
    """the viscous non-radiative torus of Stone, Pringle & Begelman 1999"""

    run: Annotated[str, ProblemParam("A", cli=True, description="the paper's run, A through K, or N")]
    orbits: Annotated[float | None, ProblemParam(None, cli=True, description="final time in orbits at R_0; the paper's by default")]
    polar_zones: Annotated[int, ProblemParam(192, cli=True, description="uniform zones in theta")]
    snapshot_orbits: Annotated[float, ProblemParam(0.01, cli=True, description="checkpoint interval in orbits")]
    zones_per_decade: Annotated[int | None, ProblemParam(None, cli=True, description="radial zones per decade; the run's by default")]
    outward_outer: Annotated[bool, ProblemParam(False, cli=True, description="the outer wall passes outflow and refuses inflow")]
    # placeholders the run's validator replaces
    resolution: tuple[int, int] = (168, 192)
    bounds: list[tuple[float, float]] = [(0.01, OUTER_RADIUS), (0.0, math.pi)]
    adiabatic_index: float = 5.0 / 3.0
    end_time: float = 1.0
    coord_system: CoordSystem = CoordSystem.SPHERICAL
    regime: Regime = Regime.NEWTONIAN
    x1_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.LOG, description="log radius")]
    velocity_components: Annotated[int | None, ProblemParam(3, description="the azimuthal velocity")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW, BoundaryCondition.AXIS, BoundaryCondition.AXIS],
            description="outflow radial walls, axis poles",
        ),
    ]
    viscous_stress: Annotated[str | None, ProblemParam("azimuthal_shear", description="the azimuthal shear stress")]
    viscous_coefficient: Annotated[str, ProblemParam("kinematic", description="nu, with mu = rho nu")]
    viscous_boundaries: Annotated[
        list[str] | None,
        ProblemParam(
            ["zero_torque", "continued_shear", "zero_torque", "zero_torque"],
            description="no torque through the inner wall and the poles",
        ),
    ]

    @computed_field
    @property
    def inner_radius(self) -> float:
        return 1.0 / RUNS[self.run][4]

    @model_validator(mode="after")
    def follow_the_run(self) -> "SpbTorus":
        """the grid, the closure, the clock and the stress of the named run"""
        assert self.run in RUNS, f"run {self.run!r}: the paper's runs are {', '.join(RUNS)}"
        scale, density_power, radius_power, per_decade, _, _, gamma, orbits, _ = RUNS[self.run]
        per_decade = self.zones_per_decade if self.zones_per_decade is not None else per_decade
        # one zone past the rounded count per decade gives the paper's 168 and 334 radial zones
        self.resolution = (round(per_decade * math.log10(OUTER_RADIUS / self.inner_radius)) + 1, self.polar_zones)
        self.bounds = [(self.inner_radius, OUTER_RADIUS), (0.0, math.pi)]
        self.adiabatic_index = gamma
        self.end_time = ORBIT * (self.orbits if self.orbits is not None else orbits)
        self.checkpoint_interval = ORBIT * self.snapshot_orbits
        self.viscosity_density_power = density_power
        self.viscosity_radius_power = radius_power
        self.outward_only_walls = [False, True, False, False] if self.outward_outer else None
        if scale == 0.0:
            self.viscous_stress = None
            self.viscous_boundaries = None
        return self

    @computed_field
    @property
    def viscosity(self) -> float:
        return RUNS[self.run][0]

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        # the compact softening is the bare point mass outside its length, which lies inside R_B
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.GRAVITATIONAL,
                mass=1.0,
                velocity=(0.0, 0.0),
                position=(0.0, 0.0),
                radius=0.0,
                gravitational=GravitationalProperties(softening_length=0.9 * self.inner_radius, softening_kind="compact"),
            )
        ]

    def initial_primitive_state(self) -> InitialStateType:
        gamma = self.adiabatic_index
        distortion = RUNS[self.run][5]
        n = 1.0 / (gamma - 1.0)
        a = polytropic_constant(self.run)

        def gas_state() -> GasStateGenerator:
            nr, nt = self.resolution
            (rl, rh), (tl, th) = self.bounds
            for jj in range(nt):
                theta = tl + (jj + 0.5) * (th - tl) / nt
                for ii in range(nr):
                    lo, hi = rl * (rh / rl) ** (ii / nr), rl * (rh / rl) ** ((ii + 1) / nr)
                    r = 0.5 * (lo + hi)
                    w = torus_enthalpy(r, theta, distortion) / (n + 1.0)
                    if w > 0.0:
                        rho = (w / a) ** n
                        if rho > AMBIENT_DENSITY:
                            yield (rho, 0.0, 0.0, 1.0 / (r * math.sin(theta)), a * rho**gamma)
                            continue
                    yield (AMBIENT_DENSITY, 0.0, 0.0, 0.0, AMBIENT_DENSITY / r)

        return gas_state
