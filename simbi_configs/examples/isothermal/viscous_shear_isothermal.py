# =============================================================================
# viscous_shear_isothermal.py
#
# a viscous shear layer in isothermal hydro on a periodic square: the transverse
# shear v_y = V0 sin(2 pi x) with a cross shear v_x = V0/2 sin(2 pi y), uniform
# density, under a constant kinematic viscosity nu. the closure p = cs^2 rho drops
# the energy equation, so the viscous stress acts on the momentum alone.
# =============================================================================
import math
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class ViscousShearIsothermal(SimbiProblem):
    """a viscous shear layer with a cross shear, isothermal, periodic."""

    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]
    v0: Annotated[float, ProblemParam(0.1, cli=True, description="shear amplitude")]
    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]
    resolution: Annotated[
        tuple[int, int], ProblemParam((64, 64), cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="domain boundaries")
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")]
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")]
    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.PERIODIC], description="boundary conditions")
    ]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="simulation end time")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, vx, vy) = (1, V0/2 sin(2 pi y), V0 sin(2 pi x))"""

        def gas_state() -> GasStateGenerator:
            ni, nj = self.resolution
            (x0, x1), (y0, y1) = self.bounds
            dx, dy = (x1 - x0) / ni, (y1 - y0) / nj
            for jj in range(nj):
                y = y0 + (jj + 0.5) * dy
                for ii in range(ni):
                    x = x0 + (ii + 0.5) * dx
                    yield (1.0, 0.5 * self.v0 * math.sin(2.0 * math.pi * y), self.v0 * math.sin(2.0 * math.pi * x))

        return gas_state
