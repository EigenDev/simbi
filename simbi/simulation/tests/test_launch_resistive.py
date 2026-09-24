# =============================================================================
# test_launch_resistive.py
#
# the ohmic resistivity through `simbi launch`. the diffusive EMF eta*J reads
# the current at every edge from the faces around it, on a cut from the
# exchanged face halo, and its diffusion limit tightens the collective time
# step. the resistive orszag-tang vortex on two workers, audited after every
# exchange, equals the single-grid run fresh and with the correction fired
# against the cut; the launched run with the resistivity switched off differs
# from the resistive one, so the launch carries eta rather than dropping it.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_resistive.py
# =============================================================================

from pathlib import Path

import pytest

from simbi_configs.examples.newtonian.resistive_orszag_tang import ResistiveOrszagTang

from simbi.simulation.tests.test_launch import (
    MHD_STEPS,
    _assert_same_state,
    _layout,
    _max_div_b,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation

SCRIPT = str(Path(ResistiveOrszagTang.__module__.replace(".", "/") + ".py"))
RESOLUTION = (32, 32, 1)


def test_a_resistive_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    env = {"SIMBI_FABRIC_AUDIT": "all"}
    reference = _script_reference(ResistiveOrszagTang.__module__, "ResistiveOrszagTang", RESOLUTION, tmp_path / "single", MHD_STEPS)
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    launched = _script_launch(SCRIPT, RESOLUTION, tmp_path / "two", two, MHD_STEPS, workers=2, env=env)
    _assert_same_state(launched, reference, "two workers, resistive vortex")
    assert _max_div_b(launched) < 1e-10
    # the launch carries the resistivity: with eta switched off the run differs
    ideal = _script_launch(SCRIPT, RESOLUTION, tmp_path / "ideal", two, MHD_STEPS, workers=2, env=env, flags=["--eta", "0.0"])
    _assert_flag_changed(ideal, launched)


def test_a_resistive_correction_against_the_cut_matches_one_worker(tmp_path: Path) -> None:
    env = {"SIMBI_FABRIC_AUDIT": "all"}
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    inject = dict(env, SIMBI_FABRIC_INJECT="trouble:3:0:15:10")
    clean = _script_launch(SCRIPT, RESOLUTION, tmp_path / "clean", one, MHD_STEPS, workers=1, env=env)
    flagged = _script_launch(SCRIPT, RESOLUTION, tmp_path / "flagged", one, MHD_STEPS, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(SCRIPT, RESOLUTION, tmp_path / "corrected", two, MHD_STEPS, workers=2, env=inject)
    _assert_same_state(corrected, flagged, "two workers, the correction against the cut in the resistive vortex")
    assert _max_div_b(corrected) < 1e-10
