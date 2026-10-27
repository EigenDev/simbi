# =============================================================================
# test_launch_isothermal_curvilinear.py
#
# the isothermal regimes on the curvilinear charts through `simbi launch`, each
# on four workers against the single-grid run, bit for bit, fresh and with the
# correction fired against a cut, the flag's effect asserted on one worker:
# - isothermal hydro on a log-radial spherical shell spanning both poles, the
#   flagged cell against the radial cut and the theta = pi pole;
# - isothermal hydro on the (r, phi) disk plane with a periodic azimuth, the
#   flagged cell inside the rotating core's edge on the periodic seam;
# - isothermal MHD on the (r, phi) disk plane in a uniform out-of-plane field,
#   flagged at the same cell.
# the two hydro runs also resume onto three unevenly owned tiles.
#
# the single-grid references and the launches run on the session's backend and
# placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_isothermal_curvilinear.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

from pathlib import Path

import h5py
import pytest

from simbi.simulation.tests.test_launch import (
    MHD_STEPS,
    STEPS,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _launch, _same_state, _script_reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation

# (module, class, resolution, steps, the flag, the four-worker cuts)
# the core's edge at r = 0.3 falls in radial cell 7 of the 32 spanning [0.1, 1.0]; a radial cut
# there puts the flagged cell (6, 0) against the cut inside the core's gradient and on the seam
CASES = {
    "spherical": ("simbi_configs.examples.isothermal.isothermal_spherical_pulse", "IsothermalSphericalPulse", (32, 32), STEPS, "trouble:3:1:15:31", None),
    "disk": ("simbi_configs.examples.isothermal.isothermal_disk_cyl", "IsothermalDiskCyl", (32, 32), STEPS, "trouble:3:0:6:0", [[7], [16]]),
    "imhd_disk": ("simbi_configs.examples.isothermal.imhd_rotor_cyl", "IsothermalRotorCyl", (32, 32, 1), MHD_STEPS, "trouble:3:0:6:0", [[7], [16]]),
}


def _four(cuts: list[list[int]] | None) -> dict:
    return dict(workers=4, cuts=cuts) if cuts else dict(workers=4, shape=(2, 2))


@pytest.mark.parametrize("name", list(CASES))
def test_a_launch_matches_the_single_grid_run_fresh_and_corrected(tmp_path: Path, name: str) -> None:
    module, cls, resolution, steps, trouble, cuts = CASES[name]
    script = module.replace(".", "/") + ".py"
    reference = _script_reference(module, cls, resolution, tmp_path / "single", steps)
    four = _four(cuts)
    launched = _launch(script, resolution, tmp_path / "four", steps, **four)
    _same_state(launched, reference, f"{name}: four workers")
    one = dict(workers=1, shape=(1, 1))
    inject = {"SIMBI_FABRIC_INJECT": trouble}
    clean = _launch(script, resolution, tmp_path / "clean", steps, **one)
    flagged = _launch(script, resolution, tmp_path / "flagged", steps, **one, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _launch(script, resolution, tmp_path / "corrected", steps, **four, env=inject)
    _same_state(corrected, flagged, f"{name}: four workers, the correction against the cut")


@pytest.mark.parametrize("name", ["spherical", "disk"])
def test_a_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path, name: str) -> None:
    module, cls, resolution, steps, _, _ = CASES[name]
    script = module.replace(".", "/") + ".py"
    reference = _script_reference(module, cls, resolution, tmp_path / "single", steps)
    partial = _launch(script, resolution, tmp_path / "first", steps // 2, workers=2, shape=(2, 1))
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < steps
    # the cap is the absolute root iteration: the checkpoint's steps and the rest
    resumed = _launch(script, resolution, tmp_path / "three", steps, workers=3, cuts=[[10, 21], []], checkpoint=partial)
    _same_state(resumed, reference, f"{name}: resumed on three workers")
