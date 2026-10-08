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
# run: pytest -m simulation simbi/simulation/tests/test_launch_isothermal_curvilinear.py
# =============================================================================

from pathlib import Path

import h5py
import pytest

from simbi.simulation.tests.test_launch import (
    MHD_STEPS,
    STEPS,
    _assert_same_state,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
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


def _four(path: Path, cuts: list[list[int]] | None) -> Path:
    return _layout(path, workers=4, cuts=cuts) if cuts else _layout(path, workers=4, shape=(2, 2))


@pytest.mark.parametrize("name", list(CASES))
def test_a_launch_matches_the_single_grid_run_fresh_and_corrected(tmp_path: Path, name: str) -> None:
    module, cls, resolution, steps, trouble, cuts = CASES[name]
    script = module.replace(".", "/") + ".py"
    reference = _script_reference(module, cls, resolution, tmp_path / "single", steps)
    four = _four(tmp_path / "four.toml", cuts)
    launched = _script_launch(script, resolution, tmp_path / "four", four, steps, workers=4)
    _assert_same_state(launched, reference, f"{name}: four workers")
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    inject = {"SIMBI_FABRIC_INJECT": trouble}
    clean = _script_launch(script, resolution, tmp_path / "clean", one, steps, workers=1)
    flagged = _script_launch(script, resolution, tmp_path / "flagged", one, steps, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(script, resolution, tmp_path / "corrected", four, steps, workers=4, env=inject)
    _assert_same_state(corrected, flagged, f"{name}: four workers, the correction against the cut")


@pytest.mark.parametrize("name", ["spherical", "disk"])
def test_a_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path, name: str) -> None:
    module, cls, resolution, steps, _, _ = CASES[name]
    script = module.replace(".", "/") + ".py"
    reference = _script_reference(module, cls, resolution, tmp_path / "single", steps)
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    partial = _script_launch(script, resolution, tmp_path / "first", two, steps // 2, workers=2)
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < steps
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    # the cap is the absolute root iteration: the checkpoint's steps and the rest
    resumed = _script_launch(script, resolution, tmp_path / "three", three, steps, workers=3, checkpoint=partial)
    _assert_same_state(resumed, reference, f"{name}: resumed on three workers")
