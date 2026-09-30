# =============================================================================
# viscous_shear_3d.py
#
# adiabatic newtonian hydro on a three-dimensional periodic box under a
# constant kinematic viscosity nu: a density wave rho = 1 + 0.2 sin(2 pi x) at
# uniform pressure, carried by a shear flow along every axis,
# v = 0.3 (sin(2 pi y), sin(2 pi z), sin(2 pi x)), so the stress couples all
# three axes and the dissipation heats the gas.
# =============================================================================
import math
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class ViscousShear3D(SimbiProblem):
    """a density wave in a three-axis shear flow under a constant viscosity, periodic."""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="adiabatic index")]
    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]
    resolution: Annotated[
        tuple[int, int, int], ProblemParam((32, 32, 32), cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")]
    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.PERIODIC], description="boundary conditions")
    ]
    end_time: Annotated[float, ProblemParam(0.5, cli=True, checkpoint_safe=True, description="simulation end time")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, vx, vy, vz, p) of the wave and the three-axis shear at unit pressure"""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            (x0, x1), (y0, y1), (z0, z1) = self.bounds
            dx, dy, dz = (x1 - x0) / ni, (y1 - y0) / nj, (z1 - z0) / nk
            for kk in range(nk):
                z = z0 + (kk + 0.5) * dz
                for jj in range(nj):
                    y = y0 + (jj + 0.5) * dy
                    for ii in range(ni):
                        x = x0 + (ii + 0.5) * dx
                        yield (
                            1.0 + 0.2 * math.sin(2.0 * math.pi * x),
                            0.3 * math.sin(2.0 * math.pi * y),
                            0.3 * math.sin(2.0 * math.pi * z),
                            0.3 * math.sin(2.0 * math.pi * x),
                            1.0,
                        )

        return gas_state
