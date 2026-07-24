# =============================================================================
# traced_kh.py
#
# kelvin-helmholtz instability seeded with LAGRANGIAN TRACER PARTICLES: massless
# points, seeded mass-weighted over the initial density (so the denser central
# layer starts with more of them), advected each step on the post-step gas
# velocity. as the shear rolls up, the tracers are wound into the billows and
# carried across the mixing layer, so their final positions map out where the
# layer's fluid ENDED UP -- the Lagrangian complement to the Eulerian dye in
# dyed_kh.py (same instability, particle view instead of field view).
#
# tracers land in the checkpoint `tracers` group (id, position, mass weight,
# provenance flags); on a periodic domain none escape, so the population is
# conserved and every checkpoint holds the full set.
#
# usage:
#  simbi run traced_kh                       # 2000 tracers by default
#  simbi run traced_kh --tracers 8000        # denser particle sampling
#  simbi run traced_kh --resolution 512,512  # sharper billows
#  simbi plot data/traced_kh/*.h5 --field rho --draw-tracers   # scatter the particles
# =============================================================================
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import computed_field

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


class TracedKelvinHelmholtz(SimbiProblem):
    """kelvin-helmholtz shear seeded with lagrangian tracer particles."""

    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    rho_layer: Annotated[
        float, ProblemParam(2.0, description="density in the central shear layer")
    ]
    rho_ambient: Annotated[
        float, ProblemParam(1.0, description="density outside the layer")
    ]
    shear_velocity: Annotated[
        float,
        ProblemParam(0.5, cli=True, description="half the velocity jump across the layer"),
    ]
    tracers: Annotated[
        int,
        ProblemParam(
            2000,
            cli=True,
            description="lagrangian tracer count (mass-weighted seeding; 0 = none)",
        ),
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
            Path("data/traced_kh/"),
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

    @computed_field
    @property
    def n_tracers(self) -> int:
        """the base problem reads tracer count here; expose it as the CLI-settable
        `tracers` field so `--tracers N` works while the backend contract is unchanged."""
        return self.tracers

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
