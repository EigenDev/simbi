# =============================================================================
# test_segment_reporting_decomposed.py
#
# the completion line of a decomposed run names the steps its own segment took
# and the root iteration it ended at, on every decomposed driver: host tiles and
# launched workers, flat and refined. a fresh run capped at 4 steps reports
# 4 steps to iteration 4; resumed from its final checkpoint and capped at 7, it
# reports 3 steps to iteration 7. the sustained rate divides by the same count.
#
# run: pytest -m simulation simbi/simulation/tests/test_segment_reporting_decomposed.py
# =============================================================================

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

PROBE = '''
import json
import os
from typing import Annotated, Any

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime

OV: dict[str, Any] = json.loads(os.environ["SEGMENT_PROBE"])


class Probe(SimbiProblem):
    """a 32 x 32 adiabatic box, refined over a central box when asked"""

    resolution: Annotated[tuple[int, int], ProblemParam((32, 32), description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="regime")]
    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    boundary_conditions: Annotated[list[Any], ProblemParam([BoundaryCondition.OUTFLOW], description="walls")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    gpus: Annotated[int, ProblemParam(OV.get("gpus", 1), cli=True, description="devices")]
    refinement_enabled: Annotated[bool, ProblemParam(OV.get("refined", False), description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2 if OV.get("refined") else 1, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75]] if OV.get("refined") else [], description="boxes")]
    refinement_ratios: Annotated[list[int], ProblemParam([2] if OV.get("refined") else [], description="ratios")]

    def initial_primitive_state(self):
        def gas():
            for jj in range(32):
                for ii in range(32):
                    yield (1.0 + 0.5 * (ii < 16), 0.0, 0.0, 1.0 + 0.5 * (jj < 16))

        return gas
'''

COMPLETION = re.compile(r"complete — (\d+) steps to iteration (\d+), t = ")


@pytest.fixture
def probe(tmp_path: Path) -> Path:
    path = tmp_path / "segment_probe.py"
    path.write_text(PROBE)
    return path


def _env(settings: dict, host_tiles: bool) -> dict:
    env = dict(os.environ, SEGMENT_PROBE=json.dumps(settings))
    env.pop("SYMBI_GPU_OVERSUBSCRIBE", None)
    if host_tiles:
        env["SYMBI_GPU_OVERSUBSCRIBE"] = "1"
    return env


def _tiles(probe: Path, settings: dict, out: Path, max_steps: int, checkpoint: Path | None) -> subprocess.CompletedProcess:
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint else ""
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(probe.parent)!r})\n"
        "from simbi.simulation import runner\n"
        "from segment_probe import Probe\n"
        f"runner.run(Probe(gpus=2, data_directory=sys.argv[1], checkpoint_interval=1e9{resume}), compute_mode='cpu', max_steps={max_steps})\n"
    )
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600, env=_env(settings, True))


def _launch(probe: Path, settings: dict, out: Path, layout: Path, max_steps: int, checkpoint: Path | None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(probe), "--layout", str(layout), "--max-steps", str(max_steps),
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint:
        cmd += ["--checkpoint", str(checkpoint)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_env(settings, False))


def _completion(done: subprocess.CompletedProcess, label: str) -> tuple[int, int]:
    text = done.stdout + done.stderr
    assert done.returncode == 0, f"{label}: the run failed:\n{text[-3000:]}"
    found = COMPLETION.findall(text)
    assert found, f"{label}: no completion line:\n{text[-3000:]}"
    steps, end = found[-1]
    return int(steps), int(end)


def _final(out: Path) -> Path:
    finals = sorted(out.glob("*final*.h5"))
    assert finals, f"no final checkpoint in {out}"
    return finals[0]


@pytest.mark.parametrize("refined", [False, True], ids=["flat", "refined"])
@pytest.mark.parametrize("driver", ["host-tiles", "launch"])
def test_a_resumed_decomposed_segment_counts_its_own_steps(tmp_path: Path, probe: Path, driver: str, refined: bool) -> None:
    settings = {"refined": refined}
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))

    def run(out: Path, max_steps: int, checkpoint: Path | None) -> tuple[int, int]:
        if driver == "host-tiles":
            done = _tiles(probe, settings, out, max_steps, checkpoint)
        else:
            done = _launch(probe, settings, out, layout, max_steps, checkpoint)
        return _completion(done, f"{driver} {'refined' if refined else 'flat'} cap {max_steps}")

    assert run(tmp_path / "fresh", 4, None) == (4, 4), "a fresh segment miscounted its steps"
    assert run(tmp_path / "resumed", 7, _final(tmp_path / "fresh")) == (3, 7), "a resumed segment counted steps it did not take"
