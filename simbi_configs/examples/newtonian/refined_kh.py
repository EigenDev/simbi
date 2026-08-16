# =============================================================================
# refined_kh.py
#
# the same kelvin-helmholtz problem as kh.py, on a statically refined mesh: one
# fine box over the central shear region where the vortex roll-ups live. the IC
# is byte-identical to kh.py (so a refined run and a uniform kh.py run are the
# same physics, only the grid differs). the hierarchy is checkpointed as level_0
# (coarse) + level_1 (fine); the fine interior is seeded by prolongation from the
# coarse IC.
#
# multi-gpu: this also exercises refinement x decomposition. run on N gpus (single
# node) with `--gpus N` -- the root grid is tiled, the fine box is tiled with it
# (its halos exchanged at the cuts), and the gathered checkpoint matches the
# single-gpu output. validate on one card: `--gpus 1` vs
# `SYMBI_GPU_OVERSUBSCRIBE=1 --gpus 2`, then diff the checkpoints.
# =============================================================================
from pathlib import Path
from typing import Annotated

import numpy as np

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.types.typing import GasStateGenerator, InitialStateType

# constants for initial conditions (identical to kh.py)
SEED = 12345
rng = np.random.default_rng(SEED)
PEEK_TO_PEEK = 0.01


class RefinedKelvinHelmholtz(SimbiProblem):
    """kelvin-helmholtz instability with one refined central box (cf. kh.py)."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    rhoL: Annotated[
        float, ProblemParam(2.0, description="density in the central layer")
    ]
    rhoR: Annotated[
        float, ProblemParam(1.0, description="density in the outer regions")
    ]
    vxT: Annotated[
        float, ProblemParam(0.5, description="x-velocity in the central layer")
    ]
    vxB: Annotated[
        float, ProblemParam(-0.5, description="x-velocity in the outer regions")
    ]
    pL: Annotated[
        float, ProblemParam(2.5, description="pressure in the central layer")
    ]
    pR: Annotated[
        float, ProblemParam(2.5, description="pressure in the outer regions")
    ]

    # domain
    resolution: Annotated[
        tuple[int, int],
        ProblemParam(
            (256, 256), cli=True, description="coarse base zones in x and y"
        ),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(-0.5, 0.5), (-0.5, 0.5)], description="domain boundaries"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(
            CellSpacing.LINEAR, description="grid spacing in x1 direction"
        ),
    ]

    # numerics
    boundary_conditions: Annotated[
        BoundaryCondition,
        ProblemParam(
            BoundaryCondition.PERIODIC, description="boundary conditions"
        ),
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="numerical solver")
    ]

    # ---- static mesh refinement: one fine box over the central shear region ----
    refinement_enabled: Annotated[
        bool, ProblemParam(True, description="enable mesh refinement")
    ]
    refinement_max_levels: Annotated[
        int, ProblemParam(2, description="coarse + 1 fine")
    ]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam(
            # one fine box over the shear layers (interfaces at y = +/-0.25, with
            # margin), wide in x but kept interior to the domain. [x_lo, x_hi, y_lo, y_hi].
            [[-0.4375, 0.4375, -0.375, 0.375]],
            description="one fine box over the central mixing region",
        ),
    ]
    refinement_ratios: Annotated[
        list[int],
        ProblemParam([2], description="coarse->fine refinement ratio per jump"),
    ]

    # simulation control
    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/refined_kh"),
            cli=True,
            checkpoint_safe=True,
            description="output data directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            20.0, cli=True, checkpoint_safe=True, description="end time"
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for kelvin-helmholtz instability."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            ymin, ymax = self.bounds[1]

            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny

            for jj in range(ny):
                y = ymin + jj * dy
                for ii in range(nx):
                    vx_noise = PEEK_TO_PEEK * np.sin(2 * np.pi * rng.normal())
                    vy_noise = PEEK_TO_PEEK * np.sin(2 * np.pi * rng.normal())

                    if abs(y) < 0.25:
                        rho = self.rhoL
                        vx = self.vxT + vx_noise
                        vy = 0.0 + vy_noise
                        p = self.pL
                    else:
                        rho = self.rhoR
                        vx = self.vxB + vx_noise
                        vy = 0.0 + vy_noise
                        p = self.pR

                    yield (rho, vx, vy, p)

        return gas_state
