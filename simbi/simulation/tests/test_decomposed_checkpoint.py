# =============================================================================
# test_decomposed_checkpoint.py
#
# the decomposed run's checkpoint against the single grid's: the gathered output state
# carries the run's coordinate maps, so a logarithmic radial mesh records the same cell
# counts, axis order, spacing types, bounds, ratios and reconstructed coordinates from
# two tiles as from one; and a checkpoint that cannot be written fails the run with the
# target path in the error. both run the decomposed path as host tiles on the session's
# backend (parity_harness), each run a fresh process, through the same construction and
# writer a multi-gpu run uses. host tiles alone involve no launch, so every gate runs
# under the local placement.
#
# run: pytest -m simulation simbi/simulation/tests/test_decomposed_checkpoint.py
#      (SIMBI_PARITY_BACKEND selects the session, parity_harness.py)
# =============================================================================
import os
import re
import stat
from pathlib import Path

import numpy as np
import pytest

from simbi.reader import read_simulation
from simbi.simulation.tests import parity_harness as ph

pytestmark = pytest.mark.simulation

# eight radial zones per decade over one decade, log spaced, on a quarter sphere
SEDOV = (
    "import sys\n"
    "from simbi.simulation import runner\n"
    "from simbi_configs.examples.newtonian.sedov import SedovTaylor\n"
    "p = SedovTaylor(zpd=8, rinit=0.1, rend=1.0, gpus=int(sys.argv[2]), data_directory=sys.argv[1])\n"
    "runner.run(p, compute_mode=sys.argv[3], validate=True, max_steps=1)\n"
)


def _sedov(out: Path, gpus: int, check: bool = True):
    """the log-radial sedov blast to step one on the session's backend over `gpus` devices"""
    return ph.python(ph.snippet_args(SEDOV, str(out), str(gpus), ph.BACKEND), f"sedov on {gpus} device(s)", devices=gpus, check=check)


def _final_checkpoint(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _assert_same_mesh(b, a) -> None:
    assert b.mesh.shape == a.mesh.shape
    assert b.mesh.spacing_types == a.mesh.spacing_types
    # storage order runs x_n .. x1: the radial axis is the last entry.
    assert "log" in b.mesh.spacing_types[-1], b.mesh.spacing_types
    assert b.mesh.spacing_ratios == a.mesh.spacing_ratios
    assert b.mesh.bounds_min == a.mesh.bounds_min
    assert b.mesh.bounds_max == a.mesh.bounds_max
    np.testing.assert_array_equal(b.mesh.x1v, a.mesh.x1v)
    np.testing.assert_array_equal(b.mesh.x2v, a.mesh.x2v)
    assert b.metadata.x1_spacing == a.metadata.x1_spacing
    assert b.metadata.x2_spacing == a.metadata.x2_spacing


@ph.local_only
def test_two_host_tiles_record_the_log_radial_mesh_like_one(tmp_path: Path) -> None:
    one, two = tmp_path / "one", tmp_path / "two"
    _sedov(one, 1)
    _sedov(two, 2)
    a, b = read_simulation(str(_final_checkpoint(one))), read_simulation(str(_final_checkpoint(two)))
    ph.disagreement(_assert_same_mesh, b, a)


@ph.local_only
def test_a_failed_decomposed_checkpoint_write_fails_the_run(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("a read-only directory does not refuse root")
    out = tmp_path / "sealed"
    out.mkdir()
    out.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        done = _sedov(out, 2, check=False)
    finally:
        out.chmod(stat.S_IRWXU)
    assert done.returncode != 0, f"the run wrote into a read-only directory:\n{done.stdout[-1500:]}"
    assert re.search(r"RuntimeError: checkpoint write failed: .*sealed", done.stderr), done.stderr[-3000:]


REFINED = (
    "import sys\n"
    "from simbi.simulation import runner\n"
    "from simbi_configs.examples.newtonian.refined_blast import RefinedBlast\n"
    "p = RefinedBlast(resolution=(16, 16, 1), gpus=2, data_directory=sys.argv[1])\n"
    "runner.run(p, compute_mode=sys.argv[2], validate=True, max_steps=2)\n"
)


@ph.local_only
def test_a_refined_decomposed_run_reports_and_writes_its_final_checkpoint(tmp_path: Path) -> None:
    # the refined decomposed loop draws the same report as the single-level one: the setup
    # milestones, the checkpoint notices and the completion line, and its final snapshot lands.
    out = tmp_path / "refined"
    done = ph.python(ph.snippet_args(REFINED, str(out), ph.BACKEND), "refined blast on two host tiles", devices=2)
    text = done.stdout + done.stderr
    assert "tiles carry a fine level" in text
    assert "complete —" in text
    assert len(list(out.glob("*final*.h5"))) == 1
