# =============================================================================
# test_decomposed_checkpoint.py
#
# the decomposed run's checkpoint against the single grid's: the gathered output state
# carries the run's coordinate maps, so a logarithmic radial mesh records the same cell
# counts, axis order, spacing types, bounds, ratios and reconstructed coordinates from
# two tiles as from one. the decomposed path runs as host tiles on a cpu build, through
# the same construction and writer a multi-gpu run uses.
# =============================================================================
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from simbi.reader import read_simulation
from simbi.simulation import runner
from simbi_configs.examples.newtonian.sedov import SedovTaylor

pytestmark = pytest.mark.simulation


def _sedov(gpus: int, out: Path) -> SedovTaylor:
    # eight radial zones per decade over one decade, log spaced, on a quarter sphere.
    return SedovTaylor(zpd=8, rinit=0.1, rend=1.0, gpus=gpus, data_directory=out)


def _final_checkpoint(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def test_two_host_tiles_record_the_log_radial_mesh_like_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SYMBI_GPU_OVERSUBSCRIBE", "1")
    one, two = tmp_path / "one", tmp_path / "two"
    runner.run(_sedov(1, one), compute_mode="cpu", validate=True, max_steps=1)
    runner.run(_sedov(2, two), compute_mode="cpu", validate=True, max_steps=1)
    a, b = read_simulation(str(_final_checkpoint(one))), read_simulation(str(_final_checkpoint(two)))
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
