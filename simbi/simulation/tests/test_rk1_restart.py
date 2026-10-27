# =============================================================================
# test_rk1_restart.py
#
# a run under first-order time stepping resumes from its own checkpoint on one
# device: the checkpoint records the integrator as the backend names it
# ("euler"), and the restart reads it back as TimeStepping.RK1 and continues to
# the step cap under the same integrator.
#
# run: pytest -m simulation simbi/simulation/tests/test_rk1_restart.py
# =============================================================================
import subprocess
import sys
from pathlib import Path

import h5py
import pytest

pytestmark = pytest.mark.simulation

RUN = """
import sys
from simbi.simulation import runner
from simbi.types import TimeStepping
from simbi_configs.examples.newtonian.kh import KelvinHelmholtz
out, steps = sys.argv[1], int(sys.argv[2])
kwargs = dict(resolution=(32, 32), data_directory=out, timestepping=TimeStepping.RK1)
if len(sys.argv) > 3:
    kwargs["checkpoint_file"] = sys.argv[3]
p = KelvinHelmholtz(**kwargs)
p.checkpoint_interval = 1e9
runner.run(p, compute_mode="cpu", max_steps=steps)
"""


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"final checkpoints in {out}: {files}"
    return files[0]


def test_a_first_order_run_resumes_from_its_checkpoint(tmp_path: Path) -> None:
    first = subprocess.run([sys.executable, "-c", RUN, str(tmp_path / "first"), "2"], capture_output=True, text=True, timeout=600)
    assert first.returncode == 0, first.stderr[-2000:]
    with h5py.File(_final(tmp_path / "first"), "r") as f:
        assert str(f["metadata"].attrs["timestepping"]) == "euler", "the checkpoint names the first-order integrator otherwise"
    resumed = subprocess.run([sys.executable, "-c", RUN, str(tmp_path / "resumed"), "4", str(_final(tmp_path / "first"))],
                             capture_output=True, text=True, timeout=600)
    assert resumed.returncode == 0, resumed.stderr[-3000:]
    with h5py.File(_final(tmp_path / "resumed"), "r") as f:
        assert int(f["level_0"].attrs["iteration"]) == 4
        assert str(f["metadata"].attrs["timestepping"]) == "euler", "the resumed run left first-order time stepping"
