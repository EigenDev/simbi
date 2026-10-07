# =============================================================================
# test_checkpoint_cadence.py
#
# the checkpoint schedule is one schedule on every driver: the single grid, host
# tiles and launched workers publish the same files, by name, iteration and
# time, and the same state in each. two schedules on the blast, with and
# without its refined box:
# - logarithmic, anchored at a fixed reference (files named by index);
# - linear, the run starting off a multiple of the interval: the boundaries
#   sit at multiples of the interval from time zero, the same on a fresh run
#   and a resumed one.
# a launched run resumed from one of its own log checkpoints continues the
# uninterrupted schedule.
#
# run: pytest -m simulation simbi/simulation/tests/test_checkpoint_cadence.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_isothermal import _script_launch

pytestmark = pytest.mark.simulation

RESOLUTION = (32, 32, 1)
STEPS = 12
SCHEDULES = {
    # eight log intervals from 1e-4 to 1e-2: the blast's steps cross several boundaries apiece early on
    "log": [
        "log_checkpoints_tuple: Annotated[tuple[bool, int], ProblemParam((True, 8), description='log')]",
        "checkpoint_log_anchor: Annotated[float, ProblemParam(1e-4, description='anchor')]",
        "end_time: Annotated[float, ProblemParam(1e-2, description='end')]",
    ],
    # boundaries at multiples of 7e-4 from zero, the run starting at 2.5e-4
    "linear": [
        "checkpoint_interval: Annotated[float, ProblemParam(7e-4, description='interval')]",
        "start_time: Annotated[float, ProblemParam(2.5e-4, description='start')]",
    ],
}
REFINEMENT = {"refined": [], "flat": ["refinement_enabled: Annotated[bool, ProblemParam(False, description='flat')]"]}


def _module(tmp_path: Path, schedule: str, grid: str, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    lines = SCHEDULES[schedule] + REFINEMENT[grid]
    (tmp_path / "cadence.py").write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi_configs.examples.newtonian.refined_blast import RefinedBlast\n\n\n"
        "class Cadence(RefinedBlast):\n" + "\n".join(f"    {line}" for line in lines) + "\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return "cadence", str(tmp_path / "cadence.py")


def _runner(out: Path, gpus: int, steps: int = STEPS) -> None:
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from cadence import Cadence\n"
        f"runner.run(Cadence(resolution={RESOLUTION!r}, gpus={gpus}, data_directory=sys.argv[1]), compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"the run on {gpus} device(s) failed:\n{done.stderr[-3000:]}"


def _published(out: Path) -> dict[str, tuple[int, float]]:
    """every published file by name: its iteration and its time"""
    files = {}
    for path in sorted(out.glob("*.h5")):
        with h5py.File(path, "r") as f:
            files[path.name] = (int(f["level_0"].attrs["iteration"]), float(f["level_0"].attrs["time"]))
    return files


def _by_index(files: dict[str, tuple[int, float]]) -> dict[str, tuple[int, float]]:
    """log checkpoints keyed by their index value: a restart widens the index padding to fit
    the projected count past the resumed index, on the single grid as on every other driver"""
    out = {}
    for name, value in files.items():
        stem, _, tail = name.partition(".chkpt.")
        token = tail.removesuffix(".h5")
        out[f"{stem}.chkpt.{int(token) if token.isdigit() else token}.h5"] = value
    return out


def _levels(path: Path) -> dict[str, np.ndarray]:
    out = {}
    with h5py.File(path, "r") as f:
        for level in (k for k in f if k.startswith("level_")):
            ng = int(f[level]["mesh"].attrs["halo_width"])
            for name in ("conserved/den", "conserved/nrg"):
                if name in f[level]:
                    data = f[level][name][()]
                    out[f"{level}/{name}"] = data[ng:-ng, ng:-ng]
    return out


def _assert_same_schedule(got: Path, want: Path, label: str) -> None:
    a, b = _published(got), _published(want)
    assert len(b) > 3, f"{label}: the single grid published {sorted(b)}; the schedule is barely exercised"
    assert a == b, f"{label}: published files differ:\n{sorted(a.items())}\nagainst\n{sorted(b.items())}"
    for name in b:
        xs, ys = _levels(got / name), _levels(want / name)
        assert xs.keys() == ys.keys() and all(np.array_equal(x, ys[k]) for k, x in xs.items()), f"{label}: {name} holds another state"


@pytest.mark.parametrize("grid", list(REFINEMENT))
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_every_driver_walks_the_single_grid_schedule(tmp_path: Path, monkeypatch, schedule: str, grid: str) -> None:
    _, script = _module(tmp_path, schedule, grid, monkeypatch)
    _runner(tmp_path / "single", 1)
    _runner(tmp_path / "tiles", 2)
    _assert_same_schedule(tmp_path / "tiles", tmp_path / "single", f"{schedule} {grid}: host tiles")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(two), "--max-steps", str(STEPS),
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(tmp_path / "launched")]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"launch failed:\n{done.stderr[-3000:]}"
    _assert_same_schedule(tmp_path / "launched", tmp_path / "single", f"{schedule} {grid}: two launched workers")


@pytest.mark.parametrize("grid", list(REFINEMENT))
def test_a_launch_resumed_from_a_log_checkpoint_continues_the_schedule(tmp_path: Path, monkeypatch, grid: str) -> None:
    _, script = _module(tmp_path, "log", grid, monkeypatch)
    _runner(tmp_path / "single", 1)
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    base = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(two), "--resolution", ",".join(str(r) for r in RESOLUTION)]
    first = tmp_path / "first"
    done = subprocess.run(base + ["--max-steps", str(STEPS // 2), "--data-directory", str(first)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"launch failed:\n{done.stderr[-3000:]}"
    scheduled = {name: v for name, v in _published(first).items() if "final" not in name}
    resume_from = max(scheduled, key=lambda name: scheduled[name][0])
    written_at = scheduled[resume_from][0]
    assert 0 < written_at < STEPS, f"the first segment's last log checkpoint sits at iteration {written_at}"
    resumed = tmp_path / "resumed"
    # the step cap is absolute on the root iteration: the resumed run stops where the uninterrupted one does
    done = subprocess.run(base + ["--max-steps", str(STEPS),"--data-directory", str(resumed), "--checkpoint", str(first / resume_from)],
                          capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"resumed launch failed:\n{done.stderr[-3000:]}"
    union = _by_index({name: v for name, v in scheduled.items() if v[0] <= written_at})
    union.update(_by_index(_published(resumed)))
    want = _by_index(_published(tmp_path / "single"))
    assert union == want, f"log {grid}: the two segments published\n{sorted(union.items())}\nagainst\n{sorted(want.items())}"
