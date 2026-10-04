# =============================================================================
# test_decomposed_mhd.py
#
# magnetized gas on host tiles against the single grid. the newtonian,
# relativistic and isothermal orszag-tang vortices, whose field varies along
# both axes, run on three uneven tiles along x and on six uneven tiles cut along
# both axes, and publish the single grid's gas, cell field and face field bit
# for bit. the refined newtonian and relativistic magnetized blasts and the
# refined isothermal orszag-tang vortex, each with one fine box, run on two
# tiles, three uneven tiles and six tiles cut along both axes through the box,
# fresh and resumed from a checkpoint written on two tiles, and publish the
# single hierarchy's every level bit for bit. a locally isothermal refined run
# on tiles is refused by name before any output.
#
# run: pytest -m simulation simbi/simulation/tests/test_decomposed_mhd.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 4
VORTICES = {
    "NewtonianOrszagTang": "simbi_configs.examples.newtonian.nmhd_orszag_tang",
    "IsothermalOrszagTang": "simbi_configs.examples.isothermal.imhd_orszag_tang",
    "OrszagTang": "simbi_configs.examples.srmhd.rmhd_orszag_tang",
}
# uneven tiles of the 32 x 32 grid: three along x, and three by two
UNEVEN = {3: [[10, 21], []], 6: [[10, 21], [13]]}


REFINED = {
    "RefinedMhdBlast": "simbi_configs.examples.newtonian.refined_mhd_blast",
    "RefinedRmhdBlast": "simbi_configs.examples.srmhd.refined_rmhd_blast",
    "RefinedIsothermalOrszagTang": "simbi_configs.examples.isothermal.refined_imhd_orszag_tang",
}
# the fine box covers root cells 8..24 of 32 on both axes; every cut lies four or more cells from
# its faces
REFINED_LAYOUTS = {2: None, 3: [[12, 20], []], 6: [[12, 20], [14]]}


def _raw(problem: str, out: Path, gpus: int, decompose=None, steps: int = STEPS, checkpoint: Path | None = None) -> subprocess.CompletedProcess:
    extra = f", decompose={decompose!r}" if decompose is not None else ""
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
    module = {**VORTICES, **REFINED}[problem]
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {module} import {problem}\n"
        f"runner.run({problem}(resolution=(32, 32, 1), gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]{extra}{resume}), "
        f"compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)


def _run(problem: str, out: Path, gpus: int, decompose=None, steps: int = STEPS, checkpoint: Path | None = None) -> Path:
    done = _raw(problem, out, gpus, decompose, steps, checkpoint)
    assert done.returncode == 0, f"{problem} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    levels = sorted({name.split("/")[0] for name in ys if name.startswith("level_") and "/" in name})
    for level in levels:
        faces = [name for name in ys if name.startswith(f"{level}/") and "/magnetic/B" in name]
        assert len(faces) == 2 and f"{level}/conserved/b1" in ys, f"{label}: {level} publishes no face and cell field: {sorted(ys)}"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


@pytest.mark.parametrize("problem", list(VORTICES))
@pytest.mark.parametrize("gpus", list(UNEVEN))
def test_uneven_tiles_carry_the_single_grid_vortex(tmp_path: Path, problem: str, gpus: int) -> None:
    want = _run(problem, tmp_path / "single", 1)
    _assert_same(_run(problem, tmp_path / "tiles", gpus, UNEVEN[gpus]), want, f"{problem} on {gpus} uneven tiles")


@pytest.mark.parametrize("problem", list(REFINED))
def test_refined_tiles_carry_the_single_hierarchy(tmp_path: Path, problem: str) -> None:
    want = _run(problem, tmp_path / "single", 1)
    levels = {name.split("/")[0] for name in _all_levels(want)}
    assert "level_1" in levels, f"{problem}: the single hierarchy published no fine level: {sorted(levels)}"
    for gpus, decompose in REFINED_LAYOUTS.items():
        _assert_same(_run(problem, tmp_path / f"tiles{gpus}", gpus, decompose), want, f"{problem} on {gpus} refined tiles")


def test_refined_tiles_resume_onto_another_partition(tmp_path: Path) -> None:
    want = _run("RefinedMhdBlast", tmp_path / "single", 1)
    first = _run("RefinedMhdBlast", tmp_path / "first", 2, steps=STEPS // 2)
    # a resumed run on tiles takes `max_steps` further steps
    resumed = _run("RefinedMhdBlast", tmp_path / "resumed", 3, REFINED_LAYOUTS[3], steps=STEPS - STEPS // 2, checkpoint=first)
    _assert_same(resumed, want, "resumed onto three uneven refined tiles")


def test_a_locally_isothermal_refined_run_on_tiles_is_refused(tmp_path: Path) -> None:
    script = tmp_path / "local.py"
    script.write_text(
        "from typing import Annotated\n"
        "from pydantic import computed_field\n"
        "from simbi import ProblemParam\n"
        "from simbi_configs.examples.isothermal.refined_imhd_orszag_tang import RefinedIsothermalOrszagTang\n\n\n"
        "class LocalRefined(RefinedIsothermalOrszagTang):\n"
        "    @computed_field\n"
        "    @property\n"
        "    def locally_isothermal(self) -> bool:\n"
        "        return True\n\n"
        "    def initial_primitive_state(self):\n"
        "        gas, *fields = super().initial_primitive_state()\n\n"
        "        def local():\n"
        "            for row in gas():\n"
        "                yield (*row, 1.0)\n\n"
        "        return (local, *fields)\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1", PYTHONPATH=os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    out = tmp_path / "out"
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from local import LocalRefined\n"
        "runner.run(LocalRefined(resolution=(32, 32, 1), gpus=2, checkpoint_interval=1e9, data_directory=sys.argv[1]), compute_mode='cpu', max_steps=1)\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode != 0, "refined tiles ran a locally isothermal closure"
    flat = " ".join(done.stderr.split())
    assert "a locally isothermal closure with mesh refinement and gpus > 1 awaits the whole-patch closure" in flat, done.stderr[-1500:]
    assert not list(out.glob("*.h5")), "output was written before the refusal"
