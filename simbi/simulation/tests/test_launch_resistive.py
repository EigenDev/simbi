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
# every arm runs on the session's backend and placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_resistive.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

from pathlib import Path

import pytest

from simbi_configs.examples.newtonian.resistive_orszag_tang import ResistiveOrszagTang

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import (
    MHD_STEPS,
    _max_div_b,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_refined import _launch, _reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation

SCRIPT = str(Path(ResistiveOrszagTang.__module__.replace(".", "/") + ".py"))
RESOLUTION = (32, 32, 1)


def _audited(inject: str | None = None) -> dict[str, str]:
    """the launch environment auditing every exchange, with the injected fault when given"""
    env = ph.environment(inject)
    env["SIMBI_FABRIC_AUDIT"] = "all"
    return env


def test_a_resistive_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    reference = _reference(ResistiveOrszagTang.__module__, "ResistiveOrszagTang", RESOLUTION, tmp_path / "single", MHD_STEPS)
    launched = _launch(SCRIPT, RESOLUTION, tmp_path / "two", MHD_STEPS, 2, shape=(2, 1), env=_audited())
    ph.same_checkpoint(launched, reference, "two workers, resistive vortex")
    assert _max_div_b(launched) < 1e-10
    # the launch carries the resistivity: with eta switched off the run differs
    ideal = _launch(SCRIPT, RESOLUTION, tmp_path / "ideal", MHD_STEPS, 2, shape=(2, 1), env=_audited(), flags=["--eta", "0.0"])
    _assert_flag_changed(ideal, launched)


def test_a_resistive_correction_against_the_cut_matches_one_worker(tmp_path: Path) -> None:
    trouble = "trouble:3:0:15:10"
    clean = _launch(SCRIPT, RESOLUTION, tmp_path / "clean", MHD_STEPS, 1, shape=(1, 1), env=_audited())
    flagged = _launch(SCRIPT, RESOLUTION, tmp_path / "flagged", MHD_STEPS, 1, shape=(1, 1), env=_audited(trouble))
    _assert_flag_changed(clean, flagged)
    corrected = _launch(SCRIPT, RESOLUTION, tmp_path / "corrected", MHD_STEPS, 2, shape=(2, 1), env=_audited(trouble))
    ph.same_checkpoint(corrected, flagged, "two workers, the correction against the cut in the resistive vortex")
    assert _max_div_b(corrected) < 1e-10
