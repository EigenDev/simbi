# =============================================================================
# refined_kh.py
#
# kelvin-helmholtz instability on a STATICALLY REFINED mesh -- the classic 2d
# shear-instability demo, with one fine box covering the shear layers where all
# the structure lives. a coarse base grid resolves the bulk flow cheaply while a
# refined central STRIP (full width, the central y-band) doubles the resolution
# across the two interfaces, where the vortex roll-ups are resolution-hungry.
#
# why this problem: KH vortices are notoriously resolution-dependent -- a coarse
# grid smears the roll-ups into a fuzzy band, a fine grid resolves the spiral
# cores. refining ONLY the central strip (static / SMR) captures the action at
# fine-grid quality for a fraction of the cells. the interfaces at y = +/-0.25
# sit well inside the strip; the coarse-fine boundaries (y = +/-0.375) lie in the
# quiet outer flow.
#
# multi-gpu: this is also the headline refinement x decomposition example. run it
# on N gpus with `--gpus N`; the root grid is split into tiles, the fine strip is
# split with them (its halos exchanged at the cuts), and the checkpoint is gathered
# back into one coarse + one fine level -- identical to the single-gpu output. to
# validate on one card: run `--gpus 1` and `SYMBI_GPU_OVERSUBSCRIBE=1 --gpus 2`
# and diff the checkpoints.
#
# the IC is a SMOOTH double shear layer (tanh ramps) with a single-mode vy seed
# localized at the interfaces -- reproducible and symmetric, so the refinement
# benefit is unambiguous rather than buried in grid-seeded noise.
# =============================================================================
from pathlib import Path
from typing import Annotated

import numpy as np

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType


class RefinedKelvinHelmholtz(SimbiProblem):
    """kelvin-helmholtz instability with one refined central strip."""

    # ---- physics ----
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    rho_in: Annotated[
        float, ProblemParam(2.0, description="density of the central (fast) layer")
    ]
    rho_out: Annotated[
        float, ProblemParam(1.0, description="density of the outer (slow) layers")
    ]
    v_shear: Annotated[
        float,
        ProblemParam(0.5, description="half the velocity jump across each interface"),
    ]
    pressure: Annotated[
        float, ProblemParam(2.5, description="uniform pressure (pressure balance)")
    ]
    shear_width: Annotated[
        float,
        ProblemParam(0.02, description="tanh shear-layer half-thickness"),
    ]
    seed_amp: Annotated[
        float,
        ProblemParam(0.01, description="single-mode vy perturbation amplitude"),
    ]
    seed_modes: Annotated[
        int,
        ProblemParam(2, description="number of vy perturbation wavelengths across x"),
    ]

    # ---- coarse base grid ----
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((128, 128, 1), cli=True, description="coarse (nx, ny, 1)"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-0.5, 0.5), (-0.5, 0.5)], description="domain bounds"),
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
        ProblemParam(BoundaryCondition.PERIODIC, description="periodic in x and y"),
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="numerical solver")
    ]

    # ---- static mesh refinement: one fine strip across the shear layers ----
    refinement_enabled: Annotated[
        bool, ProblemParam(True, description="enable mesh refinement")
    ]
    refinement_max_levels: Annotated[
        int, ProblemParam(2, description="coarse + 1 fine")
    ]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam(
            [[-0.5, 0.5, -0.375, 0.375]],
            description="one fine strip: full x, central y-band covering both interfaces",
        ),
    ]
    refinement_ratios: Annotated[
        list[int],
        ProblemParam([2], description="coarse->fine refinement ratio per jump"),
    ]

    # ---- simulation control ----
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
        ProblemParam(2.0, cli=True, checkpoint_safe=True, description="end time"),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """smooth double shear layer + single-mode vy seed; prim is (rho, vx, vy, p)."""

        def gas_state() -> GasStateGenerator:
            nx, ny, _ = self.resolution
            (xmin, xmax), (ymin, ymax) = self.bounds[0], self.bounds[1]
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            sigma = self.shear_width
            kx = 2.0 * np.pi * self.seed_modes / (xmax - xmin)

            for _kk in range(1):
                for jj in range(ny):
                    y = ymin + (jj + 0.5) * dy
                    # smooth top-hat: ~1 in the central band |y| < 0.25, ~0 outside,
                    # with tanh ramps of half-thickness sigma at y = +/-0.25.
                    band = 0.5 * (
                        np.tanh((y + 0.25) / sigma) - np.tanh((y - 0.25) / sigma)
                    )
                    rho = self.rho_out + (self.rho_in - self.rho_out) * band
                    vx = -self.v_shear + 2.0 * self.v_shear * band
                    # single-mode vy seed, localized at the two interfaces.
                    envelope = np.exp(-(((y - 0.25) / 0.05) ** 2)) + np.exp(
                        -(((y + 0.25) / 0.05) ** 2)
                    )
                    for ii in range(nx):
                        x = xmin + (ii + 0.5) * dx
                        vy = self.seed_amp * np.sin(kx * x) * envelope
                        yield (rho, vx, vy, self.pressure)

        return gas_state
