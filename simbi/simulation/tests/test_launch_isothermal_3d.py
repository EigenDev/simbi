# =============================================================================
# test_launch_isothermal_3d.py
#
# the isothermal regimes on a three-dimensional cartesian grid through `simbi
# launch`: a density wave in a three-axis shear flow (isothermal hydro) and the
# orszag-tang vortex with a z flow (isothermal MHD, under the Contact and the UCT
# edge EMF), each on four workers cut along x and z against the single-grid run,
# bit for bit, every staggered face included, fresh and with the correction fired
# on a cell against both cuts.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_isothermal_3d.py
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

SHEAR = ("simbi_configs.examples.isothermal.isothermal_shear_3d", "IsothermalShear3D")
VORTEX = ("simbi_configs.examples.isothermal.imhd_orszag_tang_3d", "IsothermalOrszagTang3D")
# (problem, launch flags, reference fields); the flagged cell (7, 3, 7) sits against both cuts at 8
CASES = {
    "shear": (SHEAR, [], ""),
    "vortex-contact": (VORTEX, [], ""),
    "vortex-uct": (VORTEX, ["--ct-method", "uct"], "ct_method='uct'"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_an_isothermal_3d_launch_matches_the_single_grid_run(tmp_path: Path, name: str) -> None:
    (module, cls), flags, fields = CASES[name]
    script = module.replace(".", "/") + ".py"
    resolution = (16, 16, 16)
    reference = _script_reference(module, cls, resolution, tmp_path / "single", STEPS, fields)
    four = _layout(tmp_path / "four.toml", workers=4, shape=(2, 1, 2))
    launched = _script_launch(script, resolution, tmp_path / "four", four, STEPS, workers=4, flags=flags)
    _assert_same_state(launched, reference, f"{name}: four workers, cuts along x and z")
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1, 1))
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:0:7:3:7"}
    clean = _script_launch(script, resolution, tmp_path / "clean", one, STEPS, workers=1, flags=flags)
    flagged = _script_launch(script, resolution, tmp_path / "flagged", one, STEPS, workers=1, env=inject, flags=flags)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(script, resolution, tmp_path / "corrected", four, STEPS, workers=4, env=inject, flags=flags)
    _assert_same_state(corrected, flagged, f"{name}: four workers, the correction against both cuts")
