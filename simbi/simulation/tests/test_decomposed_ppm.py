# =============================================================================
# test_decomposed_ppm.py
#
# a ppm run decomposed over several devices reaches the state the single-grid
# ppm run reaches, interior cell for interior cell: the cut-equivalence gate
# for the widened (-3..+2) exchange. the decomposed tile builder requests the
# reconstruction's ghost width, as the single-grid builder does; without it
# the ppm sweep trips the halo assertion on every tile, which is why the
# multi-device ppm request was refused at preflight until this gate existed.
# both arms run as fresh processes through the public command, the decomposed
# one as host tiles on a cpu build.
# =============================================================================
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.simulation

SCRIPT = "simbi_configs/examples/newtonian/kh.py"


def _run(out: Path, ngpus: int) -> Path:
    cmd = [
        sys.executable, "-m", "simbi.cli", "run", SCRIPT,
        "--mode", "cpu", "--ngpus", str(ngpus), "--reconstruction", "ppm",
        "--resolution", "32,32", "--end-time", "0.005", "--checkpoint-interval", "1e9",
        "--data-directory", str(out),
    ]
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"run with {ngpus} devices failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _interior(f: h5py.File) -> dict[str, np.ndarray]:
    level = f["level_0"]
    ng = int(level["mesh"].attrs["halo_width"])
    names: list[str] = []
    level.visit(lambda name: names.append(name) if isinstance(level[name], h5py.Dataset) else None)
    out = {}
    for name in names:
        if name.startswith("mesh/") or name.startswith("partition_0/owned") or "/domain/" in name:
            continue
        data = level[name][()]
        if data.ndim == 2:
            out[name] = data[ng:-ng, ng:-ng]
    assert out
    return out


def test_a_ppm_run_on_two_devices_matches_the_single_grid_ppm_run(tmp_path: Path) -> None:
    single = _run(tmp_path / "single", 1)
    with h5py.File(single, "r") as f:
        assert int(f["level_0/mesh"].attrs["halo_width"]) == 3, "the single-grid ppm run carries a three-cell halo"
    tiled = _run(tmp_path / "tiled", 2)
    with h5py.File(single, "r") as fs, h5py.File(tiled, "r") as ft:
        for key in ("time", "iteration"):
            assert fs["level_0"].attrs[key] == ft["level_0"].attrs[key], key
        xs, ys = _interior(fs), _interior(ft)
        assert xs.keys() == ys.keys()
        for name, x in xs.items():
            np.testing.assert_array_equal(x, ys[name], err_msg=name)
