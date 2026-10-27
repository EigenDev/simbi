# =============================================================================
# test_launch_rmhd.py
#
# special-relativistic MHD through `simbi launch`: the relativistic
# orszag-tang vortex on a 2d cartesian grid, under both edge EMFs. on four
# workers the run equals the single-grid run bit for bit, every staggered face
# included, fresh and with the correction fired against a cut (the flag
# asserted to change the run on one worker), and a run checkpointed halfway on
# four workers resumes onto three unevenly owned tiles to the uninterrupted
# single-grid state under the same edge EMF. every arm runs on the session's
# backend and placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_rmhd.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

from pathlib import Path

import h5py
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import (
    MHD_STEPS,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_refined import _launch, _reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation

MODULE = "simbi_configs.examples.srmhd.rmhd_orszag_tang"
SCRIPT = MODULE.replace(".", "/") + ".py"
RESOLUTION = (32, 32, 1)
CUTS = [[16], [16]]
# the flag sits on the corner cell of the lower-left tile, against both cuts
TROUBLE = "trouble:3:0:15:15"
RESTART = [[10, 21], []]


@pytest.mark.parametrize("ct", ["contact", "uct"])
def test_an_rmhd_launch_matches_the_single_grid_run_fresh_and_corrected(tmp_path: Path, ct: str) -> None:
    flags = ["--ct-method", ct]
    reference = _reference(MODULE, "OrszagTang", RESOLUTION, tmp_path / "single", MHD_STEPS, fields=f"ct_method={ct!r}")
    launched = _launch(SCRIPT, RESOLUTION, tmp_path / "four", MHD_STEPS, 4, cuts=CUTS, flags=flags)
    ph.same_checkpoint(launched, reference, f"rmhd {ct}: four workers")
    one = [[], []]
    clean = _launch(SCRIPT, RESOLUTION, tmp_path / "clean", MHD_STEPS, 1, cuts=one, flags=flags)
    flagged = _launch(SCRIPT, RESOLUTION, tmp_path / "flagged", MHD_STEPS, 1, cuts=one, env=ph.environment(TROUBLE), flags=flags)
    _assert_flag_changed(clean, flagged)
    corrected = _launch(SCRIPT, RESOLUTION, tmp_path / "corrected", MHD_STEPS, 4, cuts=CUTS, env=ph.environment(TROUBLE), flags=flags)
    ph.same_checkpoint(corrected, flagged, f"rmhd {ct}: four workers, the correction against the cut")


@pytest.mark.parametrize("ct", ["contact", "uct"])
def test_an_rmhd_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path, ct: str) -> None:
    flags = ["--ct-method", ct]
    reference = _reference(MODULE, "OrszagTang", RESOLUTION, tmp_path / "single", MHD_STEPS, fields=f"ct_method={ct!r}")
    partial = _launch(SCRIPT, RESOLUTION, tmp_path / "first", MHD_STEPS // 2, 4, cuts=CUTS, flags=flags)
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < MHD_STEPS
    # the cap is the absolute root iteration: the checkpoint's steps and the rest
    resumed = _launch(SCRIPT, RESOLUTION, tmp_path / "three", MHD_STEPS, 3, cuts=RESTART, checkpoint=partial, flags=flags)
    ph.same_checkpoint(resumed, reference, f"rmhd {ct}: resumed onto three uneven workers against the uninterrupted run")
