# =============================================================================
# test_launch_rmhd_shell.py
#
# special-relativistic MHD on the two-dimensional spherical shell through
# `simbi launch` and on host tiles: the magnetized monopole inflow on a polar
# wedge with log-spaced radius (the uct edge EMF), on two host tiles, on two
# launched workers and on four uneven workers cut along the radius, publishes
# the single grid's state bit for bit, every staggered face included, and the
# inflow moves off its initial state in the steps compared.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_rmhd_shell.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import EXIT, _assert_same_state, _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

STEPS = 6
MODULE, CLS = "simbi_configs.examples.srmhd.flat_srmhd_monopole", "FlatSrmhdMonopole"
NR, NPOLAR = 64, 16
SCRIPT = MODULE.replace(".", "/") + ".py"


def _single(out: Path, gpus: int = 1) -> Path:
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {MODULE} import {CLS}\n"
        f"p = {CLS}(nr={NR}, npolar={NPOLAR}, gpus={gpus}, data_directory=sys.argv[1])\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode='cpu', max_steps={STEPS})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, f"{gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _launch(out: Path, layout: Path, workers: int) -> Path:
    # the shell derives its resolution from nr and npolar
    cmd = [sys.executable, "-m", "simbi.cli", "launch", SCRIPT, "--layout", str(layout), "--max-steps", str(STEPS),
           "--nr", str(NR), "--npolar", str(NPOLAR), "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    env = {k: v for k, v in os.environ.items() if k != "SIMBI_LAUNCH_EXPERIMENTAL"}
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == list(range(workers)) and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def test_the_relativistic_magnetized_shell_launches_as_the_single_grid_runs(tmp_path: Path) -> None:
    want = _single(tmp_path / "single")
    initial = [p for p in (tmp_path / "single").glob("*.h5") if "final" not in p.name]
    assert initial, "the reference wrote no initial checkpoint to measure the evolution against"
    with h5py.File(initial[0], "r") as f0, h5py.File(want, "r") as f1:
        assert not np.array_equal(f0["level_0/conserved/den"][()], f1["level_0/conserved/den"][()]), "the inflow never moved; the gate is vacuous"
    _assert_same_state(_single(tmp_path / "tiles", gpus=2), want, "two host tiles")
    for label, workers, layout in [
        ("two workers", 2, _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))),
        ("four uneven workers", 4, _layout(tmp_path / "four.toml", workers=4, cuts=[[13, 30, 47], []])),
    ]:
        _assert_same_state(_launch(tmp_path / label.replace(" ", "_"), layout, workers), want, label)
