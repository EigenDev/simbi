# =============================================================================
# test_refined_body_shapes.py
#
# a declared body shape reaches the kernels only through a rigid wall, and a
# rigid wall on a refined staggered grid is refused: the wall would meet a
# coarse-fine face. the refusal is what keeps a refined magnetized run from
# replacing a declared shape with the sphere of its mask radius, so it holds by
# name, before any output, on the refined single grid, refined host tiles and
# a refined launch alike.
#
# run: pytest -m simulation simbi/simulation/tests/test_refined_body_shapes.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

STEPS = 4

PROBE = '''
import os
from functools import partial
from typing import Annotated, Any

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Shape, Solver
from simbi.types.bodies import BodyCapability, ImmersedBodyConfig, RigidProperties

N = 32


class ShapedObstacle(SimbiProblem):
    """a magnetized flow past a rigid body under one refined region"""

    resolution: Annotated[tuple[int, ...], ProblemParam((N, N, 1), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)], description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.NMHD, description="regime")]
    solver: Annotated[Solver, ProblemParam(Solver("hlle"), description="solver")]
    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    boundary_conditions: Annotated[list[Any], ProblemParam([BoundaryCondition("outflow")], description="walls")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    gpus: Annotated[int, ProblemParam(int(os.environ.get("SHAPE_GPUS", "1")), cli=True, description="devices")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[-0.25, 0.25, -0.25, 0.25]], description="boxes")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        # a square of half-extent 0.05 under a mask radius of 0.15: its shape differs from the
        # disc of that radius
        shape = Shape.box((0.0, 0.0, 0.0), (0.05, 0.05, 1.0))
        return [ImmersedBodyConfig(capability=BodyCapability.RIGID, mass=0.0, radius=0.15, position=(0.0, 0.0),
                                   velocity=(0.0, 0.0),
                                   rigid=RigidProperties(inertia=1.0, apply_no_slip=True, k_eta_n=50.0, k_eta_t=50.0, shape=shape))]

    def initial_primitive_state(self):
        def gas():
            for _ in range(N * N):
                yield (1.0, 0.4, 0.0, 0.0, 1.0)

        def faces(k):
            count = (N + (k == 0)) * (N + (k == 1)) * (1 + (k == 2))
            return (0.1 if k == 0 else 0.0 for _ in range(count))

        return gas, partial(faces, 0), partial(faces, 1), partial(faces, 2)
'''


@pytest.fixture
def probe(tmp_path: Path) -> Path:
    path = tmp_path / "shaped_obstacle.py"
    path.write_text(PROBE)
    return path


def _env(gpus: int = 1) -> dict:
    env = dict(os.environ, SHAPE_GPUS=str(gpus), SYMBI_GPU_OVERSUBSCRIBE="1")
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    return env


REASON = "a rigid body on a refined staggered grid is refused"


def _assert_refused(done: subprocess.CompletedProcess, out: Path, label: str) -> None:
    flat = " ".join((done.stdout + done.stderr).split())
    assert done.returncode != 0, f"{label}: a shaped rigid wall ran on a refined staggered grid"
    assert REASON in flat, f"{label}: refusal not named:\n{flat[-2000:]}"
    assert "panicked" not in flat, f"{label}: refused by a panic rather than at admission"
    assert not list(out.glob("*.h5")), f"{label}: a refused run published a checkpoint"


def _in_process(probe: Path, out: Path, gpus: int) -> subprocess.CompletedProcess:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(probe.parent)!r})\n"
        "from simbi.simulation import runner\n"
        "from shaped_obstacle import ShapedObstacle\n"
        f"runner.run(ShapedObstacle(data_directory=sys.argv[1], checkpoint_interval=1e9), compute_mode='cpu', max_steps={STEPS})\n"
    )
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600, env=_env(gpus))


def _launch(probe: Path, out: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(probe), "--layout", str(layout), "--max-steps", str(STEPS),
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_env())


def test_a_shaped_rigid_wall_on_a_refined_staggered_grid_is_refused_on_every_driver(tmp_path: Path, probe: Path) -> None:
    _assert_refused(_in_process(probe, tmp_path / "single", gpus=1), tmp_path / "single", "refined single grid")
    _assert_refused(_in_process(probe, tmp_path / "tiles", gpus=2), tmp_path / "tiles", "refined host tiles")
    _assert_refused(_launch(probe, tmp_path / "launched", tmp_path), tmp_path / "launched", "refined launch")
