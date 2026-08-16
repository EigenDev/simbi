# =============================================================================
# test_refined_nonideal_plumbing.py
#
# the config-to-fine-level plumbing gate for the non-ideal knobs on refined
# runs. the hierarchy applies the viscous pass on the finest level only (the
# refined patch is where the resolved dynamics live), so a fine kernel set
# built without the viscosity coefficient makes the entire refined run
# silently inviscid — the base level never applies it, and the fine level has
# nu = 0. the resistive edge EMF likewise rides each level's own CT, so a fine
# set without eta runs the refined patch ideal. these gates pin the sharp
# symptom: a refined run with the knob on must evolve differently from the
# refined run with it off — a fine builder that drops the knob makes the two
# bit-identical.
# =============================================================================
import glob
import os
import tempfile
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest
from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.simulation import runner
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

RES = 32
L = 0.5


class _RefinedShear2D(SimbiProblem):
    """a smooth 2d shear whose velocity gradients live inside the central
    refined patch, so the finest-level viscous pass has real work to do."""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0)]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN)]
    resolution: Annotated[tuple[int, int], ProblemParam((RES, RES))]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(-L, L), (-L, L)])
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN)]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE)]
    boundary_conditions: Annotated[
        BoundaryCondition, ProblemParam(BoundaryCondition.PERIODIC)
    ]
    cfl_number: Annotated[float, ProblemParam(0.3)]
    end_time: Annotated[float, ProblemParam(0.1, checkpoint_safe=True)]
    nu: Annotated[float, ProblemParam(0.0, cli=True, description="kinematic viscosity")]
    refinement_enabled: Annotated[bool, ProblemParam(True)]
    refinement_max_levels: Annotated[int, ProblemParam(2)]
    refinement_regions: Annotated[
        list[list[float]], ProblemParam([[-0.25, 0.25, -0.25, 0.25]])
    ]
    refinement_ratios: Annotated[list[int], ProblemParam([2])]
    checkpoint_interval: Annotated[
        float, ProblemParam(1.0e30, cli=True, checkpoint_safe=True)
    ]
    data_directory: Annotated[
        Path, ProblemParam(Path("data/_test"), cli=True, checkpoint_safe=True)
    ]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu

    def initial_primitive_state(self) -> InitialStateType:
        nx, ny = self.resolution
        d = 2.0 * L / RES
        xc = lambda i: -L + (i + 0.5) * d
        two_pi = 2.0 * np.pi

        def gas_state() -> GasStateGenerator:
            for j in range(ny):
                for i in range(nx):
                    vx = 0.1 * np.sin(two_pi * xc(j) / (2 * L))
                    vy = 0.1 * np.sin(two_pi * xc(i) / (2 * L))
                    yield (1.0, vx, vy, 1.0)

        return gas_state


def _run_viscous(nu: float) -> np.ndarray:
    d = tempfile.mkdtemp() + "/"
    p = _RefinedShear2D(nu=nu, data_directory=Path(d))
    runner.run(p, compute_mode="cpu", max_steps=400)
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, f"refined viscous run (nu={nu}) crashed"
    with h5py.File(finals[0], "r") as h:
        # the fine level is where the viscous pass acts; read its velocity.
        prims = h["level_1/partition_0/hydro/primitives"]
        return prims["v1"][...]


@needs_backend
def test_refined_run_carries_viscosity_to_the_finest_level() -> None:
    ideal = _run_viscous(0.0)
    viscous = _run_viscous(0.05)
    dv = np.abs(ideal - viscous).max()
    vscale = np.abs(ideal).max()
    assert vscale > 1e-3, "the shear never developed; the comparison is vacuous"
    assert dv > 1e-6 * vscale, (
        f"viscosity = 0.05 left the refined run bit-near-identical to ideal ({dv:e}); "
        "the fine kernel-set builder is dropping the config viscosity — and since "
        "the hierarchy applies the viscous pass on the finest level only, the whole "
        "refined run is silently inviscid"
    )
