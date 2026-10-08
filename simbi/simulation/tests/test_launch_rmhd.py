# =============================================================================
# test_launch_rmhd.py
#
# special-relativistic MHD through `simbi launch`: the relativistic
# orszag-tang vortex on a 2d cartesian grid, under both edge EMFs. on four
# workers the run equals the single-grid run bit for bit, every staggered face
# included, fresh and with the correction fired against a cut (the flag
# asserted to change the run on one worker), and a run checkpointed halfway on
# four workers resumes onto three unevenly owned tiles to the uninterrupted
# single-grid state under the same edge EMF.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_rmhd.py
# =============================================================================

from pathlib import Path

import h5py
import pytest

from simbi.simulation.tests.test_launch import (
    MHD_STEPS,
    _assert_same_state,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
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
    reference = _script_reference(MODULE, "OrszagTang", RESOLUTION, tmp_path / "single", MHD_STEPS, fields=f"ct_method={ct!r}")
    four = _layout(tmp_path / "four.toml", workers=4, cuts=CUTS)
    launched = _script_launch(SCRIPT, RESOLUTION, tmp_path / "four", four, MHD_STEPS, workers=4, flags=flags)
    _assert_same_state(launched, reference, f"rmhd {ct}: four workers")
    one = _layout(tmp_path / "one.toml", workers=1, cuts=[[], []])
    inject = {"SIMBI_FABRIC_INJECT": TROUBLE}
    clean = _script_launch(SCRIPT, RESOLUTION, tmp_path / "clean", one, MHD_STEPS, workers=1, flags=flags)
    flagged = _script_launch(SCRIPT, RESOLUTION, tmp_path / "flagged", one, MHD_STEPS, workers=1, env=inject, flags=flags)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(SCRIPT, RESOLUTION, tmp_path / "corrected", four, MHD_STEPS, workers=4, env=inject, flags=flags)
    _assert_same_state(corrected, flagged, f"rmhd {ct}: four workers, the correction against the cut")


@pytest.mark.parametrize("ct", ["contact", "uct"])
def test_an_rmhd_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path, ct: str) -> None:
    flags = ["--ct-method", ct]
    reference = _script_reference(MODULE, "OrszagTang", RESOLUTION, tmp_path / "single", MHD_STEPS, fields=f"ct_method={ct!r}")
    four = _layout(tmp_path / "four.toml", workers=4, cuts=CUTS)
    partial = _script_launch(SCRIPT, RESOLUTION, tmp_path / "first", four, MHD_STEPS // 2, workers=4, flags=flags)
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < MHD_STEPS
    three = _layout(tmp_path / "three.toml", workers=3, cuts=RESTART)
    # the cap is the absolute root iteration: the checkpoint's steps and the rest
    resumed = _script_launch(SCRIPT, RESOLUTION, tmp_path / "three", three, MHD_STEPS, workers=3, checkpoint=partial, flags=flags)
    _assert_same_state(resumed, reference, f"rmhd {ct}: resumed onto three uneven workers against the uninterrupted run")
