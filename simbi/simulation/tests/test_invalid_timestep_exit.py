# =============================================================================
# test_invalid_timestep_exit.py
#
# a march that reaches a timestep it cannot take has halted, not completed: an
# inflow face of non-finite density poisons the ghost band, the wave-speed
# reduction answers a zero timestep, and the run stops. the single grid, flat
# host tiles and refined host tiles each publish the last state as crashed, no
# final checkpoint, a report naming the timestep, and a failing exit; the
# launched workers end with a failing exit and no final checkpoint.
#
# run: pytest -m simulation simbi/simulation/tests/test_invalid_timestep_exit.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

FIXTURE = '''
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi.types import BoundaryCondition
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

DRIVEN = [BoundaryCondition.DYNAMIC, BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW]


class PoisonedInflow(RefinedBlast):
    """the blast fed through its x1-inner face by an inflow of non-finite density"""

    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam(DRIVEN, description="driven")]
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> dict:
        _, x2 = expr.coords(2)
        return expr.boundary([expr.sqrt(x2 * 0.0 - 1.0), 0.3, 0.0, 1.0], dim=2)


class RefinedPoisonedInflow(PoisonedInflow):
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
'''


@pytest.fixture
def fixture(tmp_path: Path) -> Path:
    path = tmp_path / "poisoned.py"
    path.write_text(FIXTURE)
    return path


@pytest.mark.parametrize(("cls", "gpus"), [("PoisonedInflow", 1), ("PoisonedInflow", 2), ("RefinedPoisonedInflow", 1), ("RefinedPoisonedInflow", 2)])
def test_an_invalid_timestep_ends_an_in_process_run_with_a_failing_exit(tmp_path: Path, fixture: Path, cls: str, gpus: int) -> None:
    out = tmp_path / "out"
    snippet = (
        f"import sys\nsys.path.insert(0, {str(fixture.parent)!r})\nfrom poisoned import {cls} as P\nfrom simbi.simulation import runner\n"
        f"runner.run(P(resolution=(32, 32, 1), gpus={gpus}, data_directory=sys.argv[1], checkpoint_interval=1e9), compute_mode='cpu', max_steps=4)\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))
    label = f"{cls} on {gpus} device(s)"
    assert "cfl dt is non-positive" in " ".join((done.stdout + done.stderr).split()), f"{label}: the run never reached the invalid timestep; the gate is vacuous"
    assert done.returncode != 0, f"{label}: a run halted on an invalid timestep exited cleanly"
    assert list(out.glob("*crashed*.h5")), f"{label}: no crashed snapshot of the last state"
    assert not list(out.glob("*final*.h5")), f"{label}: a halted run published a final checkpoint"


@pytest.mark.parametrize("cls", ["PoisonedInflow", "RefinedPoisonedInflow"])
def test_an_invalid_timestep_ends_a_launch_with_a_failing_exit(tmp_path: Path, fixture: Path, cls: str) -> None:
    script = fixture.parent / "launched.py"
    script.write_text(f"from poisoned import {cls} as Base\n\n\nclass Launched(Base):\n    pass\n")
    env = {k: v for k, v in os.environ.items() if k != "SIMBI_LAUNCH_EXPERIMENTAL"}
    env["PYTHONPATH"] = os.pathsep.join([str(fixture.parent), os.getcwd(), env.get("PYTHONPATH", "")])
    out = tmp_path / "out"
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "4",
           "--resolution", "32,32,1", "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert "invalid CFL candidate" in " ".join((done.stdout + done.stderr).split()), f"{cls}: the launch never reached the invalid timestep; the gate is vacuous"
    assert done.returncode != 0, f"{cls}: a launch halted on an invalid timestep exited cleanly"
    assert not list(out.glob("*final*.h5")), f"{cls}: a halted launch published a final checkpoint"
