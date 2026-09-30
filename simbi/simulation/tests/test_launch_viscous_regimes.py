# =============================================================================
# test_launch_viscous_regimes.py
#
# the constant-nu viscous transport through `simbi launch` in the isothermal and
# the magnetized regimes: a shear layer with a cross shear on a periodic square
# (isothermal hydro, newtonian MHD and isothermal MHD under a uniform B_x), on four
# workers cut along both axes against the single-grid run, bit for bit, fresh and
# with the correction fired on a cell against both cuts; the viscosity is shown to
# act, the inviscid launch differing from the viscous one.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_viscous_regimes.py
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
    four = _layout(tmp_path / "four.toml", workers=4, shape=(2, 2) if planar else (2, 2, 1))
    launched = _script_launch(script, resolution, tmp_path / "four", four, STEPS, workers=4)
    _assert_same_state(launched, reference, f"{name}: four workers")
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1) if planar else (1, 1, 1))
    inviscid = _script_launch(script, resolution, tmp_path / "inviscid", one, STEPS, workers=1, flags=["--nu", "0.0"])
    _assert_flag_changed(inviscid, launched)
    # the flagged cell (15, 15) sits against both cuts at 16
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:0:15:15"}
    clean = _script_launch(script, resolution, tmp_path / "clean", one, STEPS, workers=1)
    flagged = _script_launch(script, resolution, tmp_path / "flagged", one, STEPS, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(script, resolution, tmp_path / "corrected", four, STEPS, workers=4, env=inject)
    _assert_same_state(corrected, flagged, f"{name}: four workers, the correction against both cuts")
