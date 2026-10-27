# =============================================================================
# test_launch_polar.py
#
# the polar seam through `simbi launch`: a spherical shell gridded over both
# poles with a periodic azimuth cut into tiles. the ghost band beyond a pole
# is the interior half a turn away on other tiles, which the exchange plan's
# antipodal transfers fill and the worker signs after every state exchange;
# the single grid fills the same band by its own half-turn. the shell on two
# workers cut in azimuth, on three unevenly cut, and on four with the polar
# axis cut as well equals the single-grid run fresh and with the correction
# fired on a cell at a pole against the azimuth cut.
# the single-grid references and the launches run on the session's backend
# and placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_polar.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

from pathlib import Path

import pytest

from simbi_configs.examples.newtonian.spherical_pulse_3d import SphericalPulse3D

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import (
    STEPS,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation

SCRIPT = str(Path(SphericalPulse3D.__module__.replace(".", "/") + ".py"))
RESOLUTION = (16, 16, 16)


def test_a_shell_over_both_poles_with_a_cut_azimuth_matches_the_single_grid_run(tmp_path: Path) -> None:
    reference = _script_reference(SphericalPulse3D.__module__, "SphericalPulse3D", RESOLUTION, tmp_path / "single", STEPS)
    two = _layout(tmp_path / "two-layout.toml", workers=2, shape=(1, 1, 2))
    launched = _script_launch(SCRIPT, RESOLUTION, tmp_path / "two", two, STEPS, workers=2)
    ph.same_checkpoint(launched, reference, "two workers, the azimuth cut in halves")
    three = _layout(tmp_path / "three-layout.toml", workers=3, cuts=[[], [], [6, 10]])
    launched = _script_launch(SCRIPT, RESOLUTION, tmp_path / "three", three, STEPS, workers=3)
    ph.same_checkpoint(launched, reference, "three workers, uneven azimuth slabs")
    four = _layout(tmp_path / "four-layout.toml", workers=4, shape=(1, 2, 2))
    launched = _script_launch(SCRIPT, RESOLUTION, tmp_path / "four", four, STEPS, workers=4)
    ph.same_checkpoint(launched, reference, "four workers, the polar axis and the azimuth cut")


def test_a_correction_at_the_pole_against_the_azimuth_cut_matches_one_worker(tmp_path: Path) -> None:
    # the flagged cell (5, 0, 7) sits at the theta = 0 pole against the azimuth cut at 8
    one = _layout(tmp_path / "one-layout.toml", workers=1, shape=(1, 1, 1))
    four = _layout(tmp_path / "four-layout.toml", workers=4, shape=(1, 2, 2))
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:0:5:0:7"}
    clean = _script_launch(SCRIPT, RESOLUTION, tmp_path / "clean", one, STEPS, workers=1)
    flagged = _script_launch(SCRIPT, RESOLUTION, tmp_path / "flagged", one, STEPS, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(SCRIPT, RESOLUTION, tmp_path / "corrected", four, STEPS, workers=4, env=inject)
    ph.same_checkpoint(corrected, flagged, "four workers, the correction at the pole against the azimuth cut")
