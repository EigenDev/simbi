# =============================================================================
# cold_surface_torus.py
#
# the inviscid adiabatic torus whose cold surface goes inadmissible: the polytropic constant
# angular momentum torus of Stone, Pringle & Begelman 1999 eq. 6, d = 1.125, rho_max = 1,
# gamma = 5/3, in hydrostatic ambient gas rho_0 = 1e-4, P_0 = rho_0 / r, on a log radial
# (r, theta) grid of 84 x 44 cells over [0.01, 4] x [0, pi]. the point mass is exactly newtonian
# on every cell: its compact softening 0.009 lies inside the inner radius.
#
# usage:
#   from simbi.simulation.tests.fixtures.cold_surface_torus import ColdSurfaceTorus
#   runner.run(ColdSurfaceTorus(data_directory=out), compute_mode="cpu")
# =============================================================================
import math
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime
from simbi.types.bodies import BodyCapability, GravitationalProperties, ImmersedBodyConfig

GAMMA = 5.0 / 3.0
DISTORTION = 1.125
AMBIENT_DENSITY = 1.0e-4
POLYTROPIC_INDEX = 1.0 / (GAMMA - 1.0)
ENTROPY = (1.0 / (POLYTROPIC_INDEX + 1.0)) * (DISTORTION - 1.0) / (2.0 * DISTORTION)


class ColdSurfaceTorus(SimbiProblem):
    """a constant angular momentum torus about a newtonian point mass on the (r, theta) plane"""

    resolution: Annotated[tuple[int, int], ProblemParam((84, 44), description="grid")]
    bounds: list[tuple[float, float]] = [(0.01, 4.0), (0.0, math.pi)]
    coord_system: CoordSystem = CoordSystem.SPHERICAL
    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = GAMMA
    x1_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.LOG, description="log radius")]
    end_time: Annotated[float, ProblemParam(0.02, cli=True, checkpoint_safe=True, description="end")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW, BoundaryCondition.AXIS, BoundaryCondition.AXIS], description="walls"),
    ]

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=1.0, velocity=(0.0, 0.0), position=(0.0, 0.0), radius=0.0,
                                   gravitational=GravitationalProperties(softening_length=0.009, softening_kind="compact"))]

    def initial_primitive_state(self):
        def gas_state():
            nr, nt = self.resolution
            (rl, rh), (tl, th) = self.bounds
            for jj in range(nt):
                theta = tl + (jj + 0.5) * (th - tl) / nt
                for ii in range(nr):
                    lo, hi = rl * (rh / rl) ** (ii / nr), rl * (rh / rl) ** ((ii + 1) / nr)
                    r = 0.5 * (lo + hi)
                    big_r = r * math.sin(theta)
                    w = (1.0 / (POLYTROPIC_INDEX + 1.0)) * (1.0 / r - 0.5 / big_r**2 - 0.5 / DISTORTION)
                    if w > 0.0:
                        rho = (w / ENTROPY) ** POLYTROPIC_INDEX
                        if rho > AMBIENT_DENSITY:
                            yield (rho, 0.0, 0.0, 1.0 / big_r, ENTROPY * rho**GAMMA)
                            continue
                    yield (AMBIENT_DENSITY, 0.0, 0.0, 0.0, AMBIENT_DENSITY / r)

        return gas_state
