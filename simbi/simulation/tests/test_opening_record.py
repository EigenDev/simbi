# =============================================================================
# test_opening_record.py
#
# the opening record of a run: provenance, retry guard and retry exclusion.
# - a default run prints none of them on stderr and writes all three to the
#   run log, provenance naming the backend build and the config hash
# - a retry guard departing from the defaults reaches stderr as well
#
# run: pytest -m simulation simbi/simulation/tests/test_opening_record.py
# =============================================================================
from pathlib import Path

import pytest

from simbi.simulation import runner
from simbi_configs.examples.newtonian.sod import SodProblem

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(_BACKEND is None, reason="rust cpu_ext backend not built")

pytestmark = pytest.mark.simulation


def _sod(out: Path, **extra) -> SodProblem:
    return SodProblem(resolution=64, data_directory=str(out) + "/", checkpoint_interval=1e9, **extra)


@needs_backend
def test_a_default_run_keeps_its_opening_record_in_the_run_log(tmp_path: Path, capfd) -> None:
    runner.run(_sod(tmp_path), compute_mode="cpu", max_steps=2)
    err = capfd.readouterr().err
    for line in ("provenance:", "retry guard:", "retry exclusion:"):
        assert f"SIMBI {line}" not in err, f"a default run printed `{line}` on stderr:\n{err}"
    log = (tmp_path / "simbi.log").read_text()
    assert "provenance: backend=" in log and "config_sha256=" in log, f"the run log holds no provenance:\n{log}"
    assert "retry guard: at most 20 rejections" in log, f"the run log holds no retry guard:\n{log}"
    assert "retry exclusion: none" in log, f"the run log holds no retry exclusion:\n{log}"


@needs_backend
def test_a_retry_guard_off_the_defaults_reaches_stderr(tmp_path: Path, capfd) -> None:
    runner.run(_sod(tmp_path, retry_max_rejections=7), compute_mode="cpu", max_steps=2)
    err = capfd.readouterr().err
    assert "SIMBI retry guard: at most 7 rejections" in err, f"the configured guard was not announced:\n{err}"
    assert "SIMBI provenance:" not in err and "SIMBI retry exclusion:" not in err, err
