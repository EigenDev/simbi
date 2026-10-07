# =============================================================================
# test_retry_exclusion_propagation.py
#
# the runner configures one retry exclusion per run and carries it to every
# tile and level of every run shape: a cartesian black-hole run announces the
# legacy euclidean sphere of its outer horizon radius at startup, the same
# policy whether it runs on one grid, cut across two host tiles, or resumes
# from a checkpoint.
#
# run: pytest -m simulation simbi/simulation/tests/test_retry_exclusion_propagation.py
# =============================================================================
import re
from pathlib import Path

import pytest

from simbi.simulation import runner
from simbi_configs.examples.grhd.gr_bondi_cartesian import GrBondiCartesian

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(_BACKEND is None, reason="rust cpu_ext backend not built")

_ANNOUNCED = re.compile(r"SIMBI retry exclusion: (.*)")


def _announced(err: str) -> list[str]:
    return _ANNOUNCED.findall(err)


def _bondi(out: Path, gpus: int, **extra) -> GrBondiCartesian:
    return GrBondiCartesian(resolution=(16, 16, 16), gpus=gpus, data_directory=str(out) + "/", checkpoint_interval=1e9, **extra)


@needs_backend
def test_every_run_shape_carries_the_configured_policy(tmp_path: Path, monkeypatch, capfd) -> None:
    monkeypatch.setenv("SYMBI_GPU_OVERSUBSCRIBE", "1")
    runner.run(_bondi(tmp_path / "one", 1), compute_mode="cpu", max_steps=1)
    single = _announced(capfd.readouterr().err)
    runner.run(_bondi(tmp_path / "two", 2), compute_mode="cpu", max_steps=1)
    decomposed = _announced(capfd.readouterr().err)
    final = sorted((tmp_path / "one").glob("*final*.h5"))
    assert len(final) == 1, f"expected one final checkpoint, found {final}"
    # `max_steps` caps the absolute root iteration: the checkpoint sits at iteration 1, so the
    # resumed run takes one step
    runner.run(_bondi(tmp_path / "resumed", 1, checkpoint_file=str(final[0])), compute_mode="cpu", max_steps=1 + 1)
    resumed = _announced(capfd.readouterr().err)
    assert single and single[0].startswith("legacy euclidean sphere |x| < "), f"single-grid run announced {single}"
    assert decomposed == single, f"the decomposed run announced {decomposed}, the single-grid run {single}"
    assert resumed == single, f"the resumed run announced {resumed}, the single-grid run {single}"
