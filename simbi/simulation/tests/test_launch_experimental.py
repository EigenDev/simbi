# =============================================================================
# test_launch_experimental.py
#
# a combination the launched tiles build as the single path does but that no
# distributed gate has validated is refused at startup by name, and runs only
# under the explicit experimental opt-in, which the launch announces. the
# magnetized gas on a one-dimensional spherical grid, which no distributed
# gate covers, stands for the class; the authorization rule itself is unit
# tested beside its definition.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_experimental.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import pytest

from simbi_configs.examples.newtonian.nmhd_spherical_field import NewtonianSphericalField

from simbi.simulation.tests.test_launch import (
    EXIT,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)

pytestmark = pytest.mark.simulation

SCRIPT = str(Path(NewtonianSphericalField.__module__.replace(".", "/") + ".py"))
UNGATED = "the nmhd regime on the 1-dimensional spherical grid"


def _command(tmp_path: Path, out: str) -> list[str]:
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2,))
    return [
        sys.executable, "-m", "simbi.cli", "launch", SCRIPT, "--layout", str(layout), "--max-steps", "1",
        "--resolution", "64,1,1", "--data-directory", str(tmp_path / out), "--checkpoint-interval", "1e9",
    ]


def test_an_ungated_combination_is_refused_by_name(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "SIMBI_LAUNCH_EXPERIMENTAL"}
    done = subprocess.run(_command(tmp_path, "refused"), capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert f"{UNGATED} is installed on the launched path without a distributed gate" in flat, done.stderr
    assert "SIMBI_LAUNCH_EXPERIMENTAL=1" in flat
    assert not (tmp_path / "refused").exists() or not list((tmp_path / "refused").glob("*.h5")), "a checkpoint was published"


def test_the_experimental_opt_in_runs_the_combination_and_says_so(tmp_path: Path) -> None:
    env = dict(os.environ, SIMBI_LAUNCH_EXPERIMENTAL="1")
    done = subprocess.run(_command(tmp_path, "experimental"), capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert exits == {0: 0, 1: 0}, f"worker exit codes {exits}"
    assert f"experimental combination without a distributed gate: {UNGATED}" in " ".join(done.stderr.split()), done.stderr
    assert len(list((tmp_path / "experimental").glob("*final*.h5"))) == 1
