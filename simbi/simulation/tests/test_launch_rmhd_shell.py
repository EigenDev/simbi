# =============================================================================
# test_launch_rmhd_shell.py
#
# special-relativistic MHD on the two-dimensional spherical shell through
# `simbi launch` and on host tiles: the magnetized monopole inflow on a polar
# wedge with log-spaced radius (the uct edge EMF), on two host tiles, on two
# launched workers and on four uneven workers cut along the radius, publishes
# the single grid's state bit for bit, every staggered face included, and the
# inflow moves off its initial state in the steps compared. every arm runs on
# the session's backend and placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_rmhd_shell.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

STEPS = 6
MODULE, CLS = "simbi_configs.examples.srmhd.flat_srmhd_monopole", "FlatSrmhdMonopole"
NR, NPOLAR = 64, 16
SCRIPT = MODULE.replace(".", "/") + ".py"


def _single(out: Path, gpus: int = 1) -> Path:
    """the shell on the session's backend, one device or `gpus` host tiles"""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {MODULE} import {CLS}\n"
        f"p = {CLS}(nr={NR}, npolar={NPOLAR}, gpus={gpus}, data_directory=sys.argv[1])\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={STEPS})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out)), f"shell on {gpus} device(s)", timeout=1800, devices=gpus)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _launch(out: Path, workers: int, cuts=None, shape=None) -> Path:
    """the shell launched on the session's backend and placement without the experimental opt-in"""
    # the shell derives its resolution from nr and npolar
    args = ["--max-steps", str(STEPS), "--nr", str(NR), "--npolar", str(NPOLAR), "--checkpoint-interval", "1e9"]
    return ph.launch(SCRIPT, out, workers, cuts=cuts, shape=shape, args=args, env=ph.environment(), timeout=1800).final


def test_the_relativistic_magnetized_shell_launches_as_the_single_grid_runs(tmp_path: Path) -> None:
    want = _single(tmp_path / "single")
    initial = [p for p in (tmp_path / "single").glob("*.h5") if "final" not in p.name]
    assert initial, "the reference wrote no initial checkpoint to measure the evolution against"
    with h5py.File(initial[0], "r") as f0, h5py.File(want, "r") as f1:
        assert not np.array_equal(f0["level_0/conserved/den"][()], f1["level_0/conserved/den"][()]), "the inflow never moved; the gate is vacuous"
    ph.same_checkpoint(_single(tmp_path / "tiles", gpus=2), want, "two host tiles")
    ph.same_checkpoint(_launch(tmp_path / "two_workers", 2, shape=(2, 1)), want, "two workers")
    ph.same_checkpoint(_launch(tmp_path / "four_uneven_workers", 4, cuts=[[13, 30, 47], []]), want, "four uneven workers")
