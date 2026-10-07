# =============================================================================
# test_crash_publication.py
#
# a run that halts on a state it cannot advance publishes that state as crashed
# and never as a completed run. a uniform force whose acceleration is sqrt(-1)
# turns a refined box's momentum non-finite on its first step; the first-order
# correction then freezes the same zone on every retried substage and the step
# halts. the single grid, refined host tiles and refined launched workers all
# end with a failing exit and no final checkpoint, and the in-process drivers
# write the `.crashed` snapshot of the last state.
#
# run: pytest -m simulation simbi/simulation/tests/test_crash_publication.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

PROBE = '''
import os
from typing import Annotated, Any

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver

REGIONS = [[0.25, 0.75, 0.25, 0.75]]


class Poisoned(SimbiProblem):
    """a refined quiescent box under a force whose acceleration is sqrt(-1)"""

    resolution: Annotated[tuple[int, ...], ProblemParam((16, 16), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0)] * 2, description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime("newtonian"), description="regime")]
    solver: Annotated[Solver, ProblemParam(Solver("hlle"), description="solver")]
    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    boundary_conditions: Annotated[list[Any], ProblemParam([BoundaryCondition("outflow")], description="walls")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    gpus: Annotated[int, ProblemParam(int(os.environ.get("POISON_GPUS", "1")), cli=True, description="devices")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam(REGIONS, description="boxes")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]

    def initial_primitive_state(self):
        def gas():
            for _ in range(16 * 16):
                yield (1.0, 0.0, 0.0, 1.0)
        return gas

    @property
    def source_expressions(self):
        return [{"kind": "force", "dim": 2, "outputs": [1, 2], "params": [],
                 "vocabulary": {"reads": [], "params": []},
                 "nodes": [{"op": "CONSTANT", "value": -1.0}, {"op": "SQRT", "left": 0},
                           {"op": "CONSTANT", "value": 0.0}]}]
'''


@pytest.fixture
def probe(tmp_path: Path) -> Path:
    path = tmp_path / "poisoned.py"
    path.write_text(PROBE)
    return path


def _env(gpus: int) -> dict:
    env = dict(os.environ, POISON_GPUS=str(gpus), SYMBI_GPU_OVERSUBSCRIBE="1")
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    return env


def _in_process(probe: Path, out: Path, gpus: int) -> subprocess.CompletedProcess:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(probe.parent)!r})\n"
        "from simbi.simulation import runner\n"
        "from poisoned import Poisoned\n"
        "runner.run(Poisoned(data_directory=sys.argv[1], checkpoint_interval=1e9), compute_mode='cpu', validate=False, max_steps=5)\n"
    )
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600, env=_env(gpus))


def _assert_crashed(done: subprocess.CompletedProcess, out: Path, label: str, snapshot: bool) -> None:
    flat = " ".join((done.stdout + done.stderr).split())
    assert "freeze fired on 16 consecutive substages" in flat, f"{label}: premise: the step never halted:\n{flat[-1500:]}"
    assert done.returncode != 0, f"{label}: a crashed run exited cleanly"
    assert not list(out.glob("*final*.h5")), f"{label}: a crashed run published a final checkpoint"
    if snapshot:
        assert list(out.glob("*crashed*.h5")), f"{label}: no crashed snapshot of the last state"


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "host-tiles"])
def test_a_crashed_refined_run_publishes_its_state_as_crashed(tmp_path: Path, probe: Path, gpus: int) -> None:
    out = tmp_path / "out"
    _assert_crashed(_in_process(probe, out, gpus), out, f"refined run on {gpus} device(s)", snapshot=True)


def test_a_crashed_refined_launch_publishes_no_final_checkpoint(tmp_path: Path, probe: Path) -> None:
    out = tmp_path / "launched"
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(probe), "--layout", str(layout), "--max-steps", "5",
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_env(1))
    _assert_crashed(done, out, "refined launch", snapshot=False)
