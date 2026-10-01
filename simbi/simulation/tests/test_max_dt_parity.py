# =============================================================================
# test_max_dt_parity.py
#
# the user timestep clamp (dt = min(dt_cfl, max_dt)) on every driver. the
# blast, with and without its refined box, runs a fixed number of clamped steps
# on the single grid, as two host tiles and as two launched workers; every
# driver lands on the single grid's clock and state bit for bit. the clamp
# binds on every step of the single grid (its clock is the step count times
# max_dt), and the unclamped single grid lands elsewhere.
#
# run: pytest -m simulation simbi/simulation/tests/test_max_dt_parity.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

RESOLUTION = (32, 32, 1)
STEPS = 8
MAX_DT = 1e-6
GRIDS = {"refined": [], "flat": ["refinement_enabled: Annotated[bool, ProblemParam(False, description='flat')]"]}


def _module(tmp_path: Path, grid: str, monkeypatch: pytest.MonkeyPatch) -> str:
    def cls(name: str, clamp: float) -> str:
        lines = [f"max_dt: Annotated[float, ProblemParam({clamp!r}, description='clamp')]", *GRIDS[grid]]
        return f"class {name}(RefinedBlast):\n" + "\n".join(f"    {line}" for line in lines) + "\n"

    for name, clamp in [("Clamped", MAX_DT), ("Free", 0.0)]:
        (tmp_path / f"{name.lower()}.py").write_text(
            "from typing import Annotated\n"
            "from simbi import ProblemParam\n"
            "from simbi_configs.examples.newtonian.refined_blast import RefinedBlast\n\n\n" + cls(name, clamp)
        )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return str(tmp_path / "clamped.py")


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _run(out: Path, gpus: int, problem: str = "Clamped") -> Path:
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {problem.lower()} import {problem}\n"
        f"runner.run({problem}(resolution={RESOLUTION!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]), "
        f"compute_mode='cpu', max_steps={STEPS})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"the run on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _launch(script: str, out: Path, layout: Path) -> Path:
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(layout), "--max-steps", str(STEPS),
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(out)]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"launch failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _clock(path: Path) -> tuple[int, float]:
    with h5py.File(path, "r") as f:
        return int(f["level_0"].attrs["iteration"]), float(f["level_0"].attrs["time"])


def _levels(path: Path) -> dict[str, np.ndarray]:
    out = {}
    with h5py.File(path, "r") as f:
        for level in (k for k in f if k.startswith("level_")):
            ng = int(f[level]["mesh"].attrs["halo_width"])
            for name in ("conserved/den", "conserved/nrg"):
                if name in f[level]:
                    out[f"{level}/{name}"] = f[level][name][()][ng:-ng, ng:-ng]
    return out


@pytest.mark.parametrize("grid", list(GRIDS))
def test_every_driver_takes_the_clamped_steps_of_the_single_grid(tmp_path: Path, monkeypatch, grid: str) -> None:
    script = _module(tmp_path, grid, monkeypatch)
    want = _run(tmp_path / "single", 1)
    iteration, time = _clock(want)
    assert iteration == STEPS
    # the clamp binds on every step: the clock is the sum of STEPS steps of max_dt
    assert time == pytest.approx(STEPS * MAX_DT, rel=1e-12), f"the clamp does not bind every step: t = {time:e}"
    free = _run(tmp_path / "free", 1, problem="Free")
    assert _clock(free)[1] != time, "the unclamped run takes the clamped steps; the gate is vacuous"
    expected = _levels(want)
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    for label, got in [("host tiles", _run(tmp_path / "tiles", 2)), ("two launched workers", _launch(script, tmp_path / "launched", two))]:
        assert _clock(got) == (iteration, time), f"{grid} {label}: clock {_clock(got)} against {(iteration, time)}"
        levels = _levels(got)
        assert levels.keys() == expected.keys(), f"{grid} {label}: levels {sorted(levels)} against {sorted(expected)}"
        for key, data in expected.items():
            assert np.array_equal(levels[key], data), f"{grid} {label}: {key} differs"
