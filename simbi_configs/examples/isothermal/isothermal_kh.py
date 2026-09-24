# =============================================================================
# isothermal_kh.py
#
# a kelvin-helmholtz shear layer in isothermal newtonian hydro on a periodic
# cartesian plane: a dense band moving right between light gas moving left,
# with a smooth, deterministic transverse perturbation seeded on the two
# interfaces, so every process that builds the state builds the same numbers.
#
# usage:
#   simbi run simbi_configs/examples/isothermal/isothermal_kh.py --resolution 64,64
# =============================================================================

import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class IsothermalKelvinHelmholtz(SimbiProblem):
    """isothermal kelvin-helmholtz shear layer on a periodic plane."""

    sound_speed: Annotated[
        float, ProblemParam(1.0, cli=True, description="isothermal sound speed")
    ]
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((128, 128), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.PERIODIC], description="boundary conditions"
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            2.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]
    shear_speed: Annotated[
        float, ProblemParam(0.5, cli=True, description="half the velocity jump")
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """a band of density 2 moving right between density 1 moving left, the
        transverse velocity perturbed on both interfaces by a smooth bump times
        a sinusoid. the primitive is (rho, vx, vy)."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            (xmin, xmax), (ymin, ymax) = self.bounds
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            u = self.shear_speed
            for jj in range(ny):
                y = ymin + (jj + 0.5) * dy
                inside = 0.25 <= y < 0.75
                rho = 2.0 if inside else 1.0
                vx = u if inside else -u
                bump = math.exp(-((y - 0.25) ** 2) / 0.005) + math.exp(
                    -((y - 0.75) ** 2) / 0.005
                )
                for ii in range(nx):
                    x = xmin + (ii + 0.5) * dx
                    vy = 0.01 * bump * math.sin(4.0 * math.pi * x)
                    yield (rho, vx, vy)

        return gas_state
