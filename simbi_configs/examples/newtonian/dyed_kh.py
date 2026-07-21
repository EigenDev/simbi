# =============================================================================
# dyed_kh.py
#
# kelvin-helmholtz instability with a passive-scalar dye: the central shear
# layer is painted chi = 1, the outer regions chi = 0. the dye advects with the
# mass flux, so the rolling billows carry sharp dye filaments into the mixing
# layer — the classic scalar-mixing visualization, and a live gate on scalar
# transport: chi stays in [0, 1] (donor-cell upwinding is monotone) and total
# rho*chi is exactly conserved on the periodic domain.
#
# usage:
#  simbi run dyed_kh                        # plot --field chi for the mixing dye
#  simbi run dyed_kh --resolution 512,512   # sharper filaments
# =============================================================================
from pathlib import Path
from typing import Annotated, Optional

import numpy as np

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.types.typing import GasStateGenerator, InitialStateType

SEED = 12345
rng = np.random.default_rng(SEED)
PEEK_TO_PEEK = 0.01


class DyedKelvinHelmholtz(SimbiProblem):
    """kelvin-helmholtz shear with a dyed central layer (passive scalar)."""

    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    rho_layer: Annotated[
        float, ProblemParam(2.0, description="density in the dyed central layer")
    ]
    rho_ambient: Annotated[
        float, ProblemParam(1.0, description="density outside the layer")
    ]
    shear_velocity: Annotated[
        float,
        ProblemParam(0.5, cli=True, description="half the velocity jump across the layer"),
    ]

    resolution: Annotated[
        tuple[int, int],
        ProblemParam((256, 256), cli=True, description="number of zones in x and y"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-0.5, 0.5), (-0.5, 0.5)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]
    boundary_conditions: Annotated[
        BoundaryCondition,
        ProblemParam(BoundaryCondition.PERIODIC, description="boundary conditions"),
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="numerical solver")
    ]

    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/dyed_kh/"),
            cli=True,
            checkpoint_safe=True,
            description="output data directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(10.0, cli=True, checkpoint_safe=True, description="end time"),
    ]
    checkpoint_interval: Annotated[
        float,
        ProblemParam(
            0.25, cli=True, checkpoint_safe=True, description="checkpoint interval"
        ),
    ]

    def _in_layer(self, y: float) -> bool:
        return abs(y) < 0.25

    def initial_primitive_state(self) -> InitialStateType:
        """shear layer with seeded noise; pressure-uniform."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            (_, _), (ymin, ymax) = self.bounds[0], self.bounds[1]
            dy = (ymax - ymin) / ny
            for jj in range(ny):
                y = ymin + jj * dy
                for _ii in range(nx):
                    vx_noise = PEEK_TO_PEEK * np.sin(2 * np.pi * rng.normal())
                    vy_noise = PEEK_TO_PEEK * np.sin(2 * np.pi * rng.normal())
                    if self._in_layer(y):
                        yield (
                            self.rho_layer,
                            self.shear_velocity + vx_noise,
                            vy_noise,
                            2.5,
                        )
                    else:
                        yield (
                            self.rho_ambient,
                            -self.shear_velocity + vx_noise,
                            vy_noise,
                            2.5,
                        )

        return gas_state

    def passive_scalar(self) -> Optional[GasStateGenerator]:
        """paint the central layer chi = 1, the ambient chi = 0."""

        def dye() -> GasStateGenerator:
            nx, ny = self.resolution
            ymin, ymax = self.bounds[1]
            dy = (ymax - ymin) / ny
            for jj in range(ny):
                y = ymin + jj * dy
                for _ii in range(nx):
                    yield 1.0 if self._in_layer(y) else 0.0

        return dye()
