# =============================================================================
# refined_blast_3d.py
#
# a spherical newtonian blast on a statically refined 3D mesh: a central
# overpressure of radius 0.1 at rest in a uniform medium, on the cube
# [-0.5, 0.5]^3 with outflow walls, one fine box of ratio two over
# [-0.25, 0.25]^3. the blast wave crosses the coarse-fine boundary, where the
# flux register refluxes the root and the fine level restricts into it.
#
# usage:
#  simbi run refined_blast_3d.py
#  simbi launch refined_blast_3d.py --layout layout.toml
# =============================================================================
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType

BOUND = 0.5


class RefinedBlast3D(SimbiProblem):
    """a spherical blast with a fine box around it."""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="adiabatic index")]
    p_blast: Annotated[float, ProblemParam(10.0, description="central overpressure")]
    r_blast: Annotated[float, ProblemParam(0.1, description="blast radius")]
    resolution: Annotated[tuple[int, int, int], ProblemParam((32, 32, 32), cli=True, description="root resolution")]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-BOUND, BOUND), (-BOUND, BOUND), (-BOUND, BOUND)], description="domain bounds"),
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")]
    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.OUTFLOW], description="boundary conditions")
    ]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="enable mesh refinement")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="coarse + 1 fine")]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam([[-0.25, 0.25, -0.25, 0.25, -0.25, 0.25]], description="one fine box around the blast"),
    ]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="coarse->fine refinement ratio per jump")]
    end_time: Annotated[float, ProblemParam(0.1, cli=True, checkpoint_safe=True, description="end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """central overpressure, at rest; primitive is (rho, vx, vy, vz, p)."""

        def gas_state() -> GasStateGenerator:
            nx, ny, nz = self.resolution
            (x0, x1), (y0, y1), (z0, z1) = self.bounds
            dx, dy, dz = (x1 - x0) / nx, (y1 - y0) / ny, (z1 - z0) / nz
            for kk in range(nz):
                z = z0 + (kk + 0.5) * dz
                for jj in range(ny):
                    y = y0 + (jj + 0.5) * dy
                    for ii in range(nx):
                        x = x0 + (ii + 0.5) * dx
                        r = (x * x + y * y + z * z) ** 0.5
                        yield (1.0, 0.0, 0.0, 0.0, self.p_blast if r <= self.r_blast else 0.1)

        return gas_state
