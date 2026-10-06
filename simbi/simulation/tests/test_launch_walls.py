# =============================================================================
# test_launch_walls.py
#
# the domain walls through `simbi launch`. a tile at the domain edge fills its
# wall ghosts itself after the cut halos arrive, so the corner where a cut
# meets a wall holds the wall's image of the neighbor's cells. four workers on
# a two-by-two layout put such a corner on every tile: the rayleigh-taylor
# layer (periodic x, reflecting y, a gravity source), the MHD rotor (outflow
# walls, audited after every exchange), and the gradient box (a neumann wall
# on x, a robin wall on y) each equal the single-grid run, fresh and with the
# correction fired on a wall-corner cell; a gradient wall with a magnetized
# regime is refused by name, since the magnetized kernel set carries no
# gradient table.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_walls.py
# =============================================================================

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi_configs.examples.newtonian.gradient_box import GradientBox
from simbi_configs.examples.newtonian.nmhd_rotor import NewtonianRotor
from simbi_configs.examples.newtonian.rt import RayleighTaylor

from simbi.simulation.tests.test_launch import (
    MHD_STEPS,
    STEPS,
    _assert_same_state,
    _interior,
    _layout,
    _max_div_b,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference

pytestmark = pytest.mark.simulation


def _script_of(cls: type) -> str:
    return str(Path(cls.__module__.replace(".", "/") + ".py"))


RT_SCRIPT = _script_of(RayleighTaylor)
ROTOR_SCRIPT = _script_of(NewtonianRotor)
GRADIENT_SCRIPT = _script_of(GradientBox)


def _assert_flag_changed(clean: Path, flagged: Path) -> None:
    with h5py.File(clean, "r") as fc, h5py.File(flagged, "r") as fr:
        xs, ys = _interior(fc), _interior(fr)
        assert any(not np.array_equal(xs[name], ys[name]) for name in xs), "the flag left the one-worker run unchanged; the gate is vacuous"


def _wall_gates(tmp_path: Path, script: str, module: str, cls: str, resolution: tuple[int, ...], steps: int, trouble: str, env: dict[str, str] | None = None, magnetized: bool = False) -> None:
    reference = _script_reference(module, cls, resolution, tmp_path / "single", steps)
    four = _layout(tmp_path / "four.toml", workers=4, shape=(2, 2))
    launched = _script_launch(script, resolution, tmp_path / "four", four, steps, workers=4, env=env)
    _assert_same_state(launched, reference, "four workers, cuts meeting the walls")
    if magnetized:
        assert _max_div_b(launched) < 1e-10
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    inject = dict(env or {}, SIMBI_FABRIC_INJECT=trouble)
    clean = _script_launch(script, resolution, tmp_path / "clean", one, steps, workers=1, env=env)
    flagged = _script_launch(script, resolution, tmp_path / "flagged", one, steps, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(script, resolution, tmp_path / "corrected", four, steps, workers=4, env=inject)
    _assert_same_state(corrected, flagged, "four workers, the correction on a wall-corner cell")
    if magnetized:
        assert _max_div_b(corrected) < 1e-10


def test_reflecting_walls_with_a_gravity_source_match_the_single_grid_run(tmp_path: Path) -> None:
    # cuts at x = 8 and y = 24; the flagged cell (7, 0) sits against the x cut and the low
    # reflecting wall
    _wall_gates(tmp_path, RT_SCRIPT, RayleighTaylor.__module__, "RayleighTaylor", (16, 48), STEPS, "trouble:3:0:7:0")


def test_outflow_walls_of_the_magnetized_rotor_match_the_single_grid_run(tmp_path: Path) -> None:
    # cuts at x = 16 and y = 16; the flagged cell (15, 0) sits against the x cut and the low
    # outflow wall
    _wall_gates(
        tmp_path, ROTOR_SCRIPT, NewtonianRotor.__module__, "NewtonianRotor", (32, 32, 1), MHD_STEPS,
        "trouble:3:0:15:0", env={"SIMBI_FABRIC_AUDIT": "all"}, magnetized=True,
    )


def test_gradient_walls_match_the_single_grid_run(tmp_path: Path) -> None:
    # cuts at x = 16 and y = 16; the flagged cell (16, 31) sits against the x cut and the high
    # robin wall, with the flag in the corrector
    _wall_gates(tmp_path, GRADIENT_SCRIPT, GradientBox.__module__, "GradientBox", (32, 32), STEPS, "trouble:3:1:16:31")


def test_a_gradient_wall_with_a_magnetized_regime_is_refused_by_name(tmp_path: Path) -> None:
    script = tmp_path / "gradient_wall_vortex.py"
    script.write_text(
        "from typing import Annotated\n"
        "from simbi import Neumann, ProblemParam\n"
        "from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang\n"
        "\n"
        "class GradientWallVortex(NewtonianOrszagTang):\n"
        "    boundary_conditions: Annotated[list, ProblemParam([Neumann(rho=0.1)], description='boundary conditions')]\n"
    )
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1", "--resolution", "16,16,1", "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert "a neumann or robin wall with the nmhd regime: the magnetized substrates fill no gradient ghost" in flat, done.stderr
    assert not list((tmp_path / "out").glob("*.h5")), "a refused launch published a checkpoint"


def test_a_user_source_with_a_magnetized_regime_is_refused_by_name(tmp_path: Path) -> None:
    # the launched magnetized tiles install no user source, so a vortex with a source is refused
    # rather than run without it
    script = tmp_path / "sourced_vortex.py"
    script.write_text(
        "import simbi.expression as expr\n"
        "from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang\n"
        "\n"
        "class SourcedVortex(NewtonianOrszagTang):\n"
        "    @property\n"
        "    def source_expressions(self):\n"
        "        return [expr.force([0.0, -0.1, 0.0], dim=3)]\n"
    )
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1", "--resolution", "16,16,1", "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert "a user source with the nmhd regime is outside the distributed first release" in flat, done.stderr

