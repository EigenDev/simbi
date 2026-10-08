# =============================================================================
# test_launch_gr_plane.py
#
# the meridional plane of a spinning black hole through `simbi launch`. the
# fishbone-moncrief torus on the two-dimensional spherical kerr-schild chart
# carries the azimuthal momentum the frame dragging needs (three momentum
# components on a two-axis grid); two launched workers and three uneven
# workers cut along the radius publish the single grid's state bit for bit,
# and the torus moves off its initial state in the steps compared.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_gr_plane.py
# =============================================================================

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_isothermal import EXIT
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 6
MODULE, CLS = "simbi_configs.examples.grhd.gr_fishbone_moncrief", "GrFishboneMoncrief"
NR, NPOLAR = 48, 32
SCRIPT = MODULE.replace(".", "/") + ".py"


def _single(out: Path) -> Path:
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {MODULE} import {CLS}\n"
        f"p = {CLS}(nr={NR}, npolar={NPOLAR}, data_directory=sys.argv[1])\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode='cpu', max_steps={STEPS})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, f"the single grid failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _launch(out: Path, layout: Path, workers: int) -> Path:
    # the torus derives its resolution from nr and npolar
    cmd = [sys.executable, "-m", "simbi.cli", "launch", SCRIPT, "--layout", str(layout), "--max-steps", str(STEPS),
           "--nr", str(NR), "--npolar", str(NPOLAR), "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == list(range(workers)) and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert "level_0/conserved/den" in ys, f"{label}: the reference publishes no density; the comparison is vacuous"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def test_the_kerr_torus_plane_launches_as_the_single_grid_runs(tmp_path: Path) -> None:
    want = _single(tmp_path / "single")
    initial = [p for p in (tmp_path / "single").glob("*.h5") if "final" not in p.name]
    assert initial, "the reference wrote no initial checkpoint to measure the evolution against"
    with h5py.File(initial[0], "r") as f0, h5py.File(want, "r") as f1:
        assert not np.array_equal(f0["level_0/conserved/den"][()], f1["level_0/conserved/den"][()]), "the torus never moved; the gate is vacuous"
        assert f1["level_0/conserved"].keys() == f0["level_0/conserved"].keys()
    for label, workers, layout in [
        ("two workers", 2, _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))),
        ("three uneven workers", 3, _layout(tmp_path / "three.toml", workers=3, cuts=[[15, 33], []])),
    ]:
        got = _launch(tmp_path / label.replace(" ", "_"), layout, workers)
        _assert_same(got, want, label)
