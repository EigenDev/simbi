# =============================================================================
# test_launch_viscous_regimes.py
#
# the constant-nu viscous transport through `simbi launch` in the isothermal and
# the magnetized regimes: a shear layer with a cross shear on a periodic square
# (isothermal hydro, newtonian MHD and isothermal MHD under a uniform B_x), on four
# workers cut along both axes against the single-grid run, bit for bit, fresh and
# with the correction fired on a cell against both cuts; the viscosity is shown to
# act, the inviscid launch differing from the viscous one. a run checkpointed
# halfway resumes to the uninterrupted single-grid state, on one worker and onto
# three unevenly owned tiles from a four-worker checkpoint.
#
# the single-grid references and the launches run on the session's backend and
# placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_viscous_regimes.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

from pathlib import Path

import h5py
import pytest

from simbi.simulation.tests.test_launch import (
    STEPS,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _launch, _same_state, _script_reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation

CASES = {
    "isothermal": ("simbi_configs.examples.isothermal.viscous_shear_isothermal", "ViscousShearIsothermal", (32, 32)),
    "nmhd": ("simbi_configs.examples.newtonian.viscous_shear_mhd", "ViscousShearMhd", (32, 32, 1)),
    "imhd": ("simbi_configs.examples.isothermal.viscous_shear_imhd", "ViscousShearImhd", (32, 32, 1)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_a_viscous_launch_matches_the_single_grid_run(tmp_path: Path, name: str) -> None:
    module, cls, resolution = CASES[name]
    script = module.replace(".", "/") + ".py"
    planar = len(resolution) == 2
    reference = _script_reference(module, cls, resolution, tmp_path / "single", STEPS)
    four = dict(workers=4, shape=(2, 2) if planar else (2, 2, 1))
    launched = _launch(script, resolution, tmp_path / "four", STEPS, **four)
    _same_state(launched, reference, f"{name}: four workers")
    one = dict(workers=1, shape=(1, 1) if planar else (1, 1, 1))
    inviscid = _launch(script, resolution, tmp_path / "inviscid", STEPS, **one, flags=["--nu", "0.0"])
    _assert_flag_changed(inviscid, launched)
    # the flagged cell (15, 15) sits against both cuts at 16
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:0:15:15"}
    clean = _launch(script, resolution, tmp_path / "clean", STEPS, **one)
    flagged = _launch(script, resolution, tmp_path / "flagged", STEPS, **one, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _launch(script, resolution, tmp_path / "corrected", STEPS, **four, env=inject)
    _same_state(corrected, flagged, f"{name}: four workers, the correction against both cuts")


@pytest.mark.parametrize("name", list(CASES))
def test_a_viscous_launch_resumed_onto_another_partition_matches(tmp_path: Path, name: str) -> None:
    module, cls, resolution = CASES[name]
    script = module.replace(".", "/") + ".py"
    planar = len(resolution) == 2
    reference = _script_reference(module, cls, resolution, tmp_path / "single", STEPS)
    one = dict(workers=1, shape=(1, 1) if planar else (1, 1, 1))
    partial_one = _launch(script, resolution, tmp_path / "half", STEPS // 2, **one)
    with h5py.File(partial_one, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    # the cap is the absolute root iteration: the checkpoint's steps and the rest
    resumed_one = _launch(script, resolution, tmp_path / "resumed_one", STEPS, **one, checkpoint=partial_one)
    _same_state(resumed_one, reference, f"{name}: resumed on one worker against the uninterrupted run")
    partial = _launch(script, resolution, tmp_path / "first", STEPS // 2, workers=4, shape=(2, 2) if planar else (2, 2, 1))
    # a 2.5D magnetized problem partitions its two spatial axes; the cap is the absolute root
    # iteration: the checkpoint's steps and the rest
    resumed = _launch(script, resolution, tmp_path / "three", STEPS, workers=3, cuts=[[10, 21], []], checkpoint=partial)
    _same_state(resumed, reference, f"{name}: resumed onto three uneven workers against the uninterrupted run")
