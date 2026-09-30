# =============================================================================
# test_launch_isothermal_walls.py
#
# the isothermal regimes behind walls and under resistivity through `simbi
# launch`, each on four workers cut along both axes against the single-grid run,
# bit for bit, every staggered face included, fresh and with the correction fired
# on a cell against both cuts or, behind a wall, on the cell where a cut meets it:
# an isothermal channel (periodic along x, reflecting walls in y), an isothermal
# overdensity drifting toward outflow walls, the isothermal orszag-tang vortex
# with outflow walls, and the isothermal vortex under an ohmic resistivity, shown
# to act against the ideal launch.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_isothermal_walls.py
# =============================================================================

from pathlib import Path

import pytest

from simbi.simulation.tests.test_launch import (
    STEPS,
    _assert_same_state,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation

I = "simbi_configs.examples.isothermal."
# (module, class, resolution, the flagged cell, the flags that switch the physics under test off)
CASES = {
    "channel": (I + "isothermal_channel", "IsothermalChannel", (32, 32), "trouble:3:0:15:0", None),
    "outflow": (I + "isothermal_outflow_blob", "IsothermalOutflowBlob", (32, 32), "trouble:3:0:15:15", None),
    "imhd-walls": (I + "imhd_orszag_tang_walls", "IsothermalOrszagTangWalls", (32, 32, 1), "trouble:3:0:15:0", None),
    "imhd-resistive": (I + "resistive_imhd_orszag_tang", "ResistiveIsothermalOrszagTang", (32, 32, 1), "trouble:3:0:15:15", ["--eta", "0.0"]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_an_isothermal_launch_matches_the_single_grid_run(tmp_path: Path, name: str) -> None:
    module, cls, resolution, trouble, off = CASES[name]
    script = module.replace(".", "/") + ".py"
    planar = len(resolution) == 2
    reference = _script_reference(module, cls, resolution, tmp_path / "single", STEPS)
    four = _layout(tmp_path / "four.toml", workers=4, shape=(2, 2) if planar else (2, 2, 1))
    launched = _script_launch(script, resolution, tmp_path / "four", four, STEPS, workers=4)
    _assert_same_state(launched, reference, f"{name}: four workers")
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1) if planar else (1, 1, 1))
    if off:
        switched_off = _script_launch(script, resolution, tmp_path / "off", one, STEPS, workers=1, flags=off)
        _assert_flag_changed(switched_off, launched)
    inject = {"SIMBI_FABRIC_INJECT": trouble}
    clean = _script_launch(script, resolution, tmp_path / "clean", one, STEPS, workers=1)
    flagged = _script_launch(script, resolution, tmp_path / "flagged", one, STEPS, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(script, resolution, tmp_path / "corrected", four, STEPS, workers=4, env=inject)
    _assert_same_state(corrected, flagged, f"{name}: four workers, the correction")
