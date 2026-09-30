# =============================================================================
# test_gradient_wall_defaults.py
#
# a neumann or robin wall that omits trailing velocity components runs with
# the documented defaults: a zero normal gradient for neumann and the dirichlet
# triple (1, 0, 0) for robin, bit for bit the run that spells them out. a wall
# giving more velocity components than the regime carries is refused by name.
#
# run: pytest -m simulation simbi/simulation/tests/test_gradient_wall_defaults.py
# =============================================================================

import subprocess
import sys
from pathlib import Path

import pytest

from simbi.simulation.tests.test_launch import _assert_same_state

pytestmark = pytest.mark.simulation

WALLS = {
    "omitted": "[Neumann(rho=0.2, pressure=-0.3), Robin(rho=(1.0, 0.2, 1.1), pressure=(1.0, 0.3, 1.2))]",
    "explicit": (
        "[Neumann(rho=0.2, velocity=(0.0, 0.0), pressure=-0.3), "
        "Robin(rho=(1.0, 0.2, 1.1), velocity=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), pressure=(1.0, 0.3, 1.2))]"
    ),
    "too_long": "[Neumann(rho=0.2, velocity=(0.0, 0.1, 0.0)), Neumann(rho=0.1)]",
}


def _run(out: Path, walls: str) -> subprocess.CompletedProcess:
    snippet = f"""
import sys
from simbi import Neumann, Robin
from simbi.simulation import runner
from simbi_configs.examples.newtonian.gradient_box import GradientBox
problem = GradientBox(resolution=(16, 16), data_directory=sys.argv[1], checkpoint_interval=1e9, boundary_conditions={walls})
runner.run(problem, compute_mode="cpu", max_steps=4)
"""
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600)


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def test_omitted_velocity_components_take_the_documented_defaults(tmp_path: Path) -> None:
    omitted = _run(tmp_path / "omitted", WALLS["omitted"])
    assert omitted.returncode == 0, f"the run with omitted components failed:\n{omitted.stderr[-2000:]}"
    explicit = _run(tmp_path / "explicit", WALLS["explicit"])
    assert explicit.returncode == 0, f"the run with explicit components failed:\n{explicit.stderr[-2000:]}"
    _assert_same_state(_final(tmp_path / "omitted"), _final(tmp_path / "explicit"), "omitted against explicit defaults")


def test_more_velocity_components_than_the_regime_carries_are_refused(tmp_path: Path) -> None:
    done = _run(tmp_path / "too_long", WALLS["too_long"])
    assert done.returncode != 0, "a wall with three velocity components ran on a 2d hydro grid"
    assert "a neumann wall gives 3 velocity components; the regime carries 2" in " ".join(done.stderr.split()), done.stderr[-2000:]
