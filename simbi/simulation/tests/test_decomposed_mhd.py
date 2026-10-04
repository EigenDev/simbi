# =============================================================================
# test_decomposed_mhd.py
#
# magnetized gas on host tiles against the single grid. the newtonian,
# relativistic and isothermal orszag-tang vortices, whose field varies along
# both axes, run on three uneven tiles along x and on six uneven tiles cut along
# both axes, and publish the single grid's gas, cell field and face field bit
# for bit.
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


def _run(problem: str, out: Path, gpus: int, decompose=None) -> Path:
    extra = f", decompose={decompose!r}" if decompose is not None else ""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {VORTICES[problem]} import {problem}\n"
        f"runner.run({problem}(resolution=(32, 32, 1), gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]{extra}), "
        f"compute_mode='cpu', max_steps={STEPS})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"{problem} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    faces = [name for name in ys if "/magnetic/B" in name]
    assert len(faces) == 2 and "level_0/conserved/b1" in ys, f"{label}: the reference publishes no face and cell field: {sorted(ys)}"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


@pytest.mark.parametrize("problem", list(VORTICES))
@pytest.mark.parametrize("gpus", list(UNEVEN))
def test_uneven_tiles_carry_the_single_grid_vortex(tmp_path: Path, problem: str, gpus: int) -> None:
    want = _run(problem, tmp_path / "single", 1)
    _assert_same(_run(problem, tmp_path / "tiles", gpus, UNEVEN[gpus]), want, f"{problem} on {gpus} uneven tiles")
