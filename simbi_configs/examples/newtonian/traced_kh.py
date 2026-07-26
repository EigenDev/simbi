# =============================================================================
# traced_kh.py
#
# kelvin-helmholtz instability seeded with fixed-mass transport tracers. the
# population is seeded in proportion to initial cell mass, so the denser
# central layer starts with more tracers. accepted finite-volume mass transfers
# move their authoritative cell owners through the same shear fluxes as the
# gas. the checkpoint position is the derived owner-cell centroid, giving a
# discrete lagrangian complement to the eulerian dye in dyed_kh.py.
#
# tracers land in the checkpoint `tracers` group with exact ids, cell or
# reservoir owners, derived positions, mass weight, immutable initial-material
# cohort, provenance flags, and deterministic spawning state. cohort 1 begins
# in the dense layer and cohort 0 in the ambient gas.
#
# usage:
#  simbi run traced_kh                       # 2000 tracers by default
#  simbi run traced_kh --tracers 8000        # denser particle sampling
#  simbi run traced_kh --resolution 512,512  # sharper billows
#  simbi plot data/traced_kh/*.h5 --field rho --draw-tracers   # scatter the particles
#  simbi plot data/traced_kh/*.h5 --tracers-only               # concentration map
#  simbi plot data/traced_kh/*.h5 --tracers-only --tracer-render scatter
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
        """expose the cli tracer count through the backend field."""
        return self.tracers

    def _in_layer(self, y: float) -> bool:
        return abs(y) < 0.25

    def tracer_cohort(self) -> GasStateGenerator:
        """label the dense layer independently of its later position."""

        nx, ny = self.resolution
        (_, _), (ymin, ymax) = self.bounds[0], self.bounds[1]
        dy = (ymax - ymin) / ny
        for jj in range(ny):
            y = ymin + (jj + 0.5) * dy
            for _ii in range(nx):
                yield 1 if self._in_layer(y) else 0

    def initial_primitive_state(self) -> InitialStateType:
        """shear layer with seeded noise; pressure-uniform."""

        def gas_state() -> GasStateGenerator:
            rng = np.random.default_rng(SEED)
            nx, ny = self.resolution
            (_, _), (ymin, ymax) = self.bounds[0], self.bounds[1]
            dy = (ymax - ymin) / ny
            for jj in range(ny):
                y = ymin + (jj + 0.5) * dy
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
