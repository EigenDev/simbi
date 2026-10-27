# =============================================================================
# test_launch_3d.py
#
# three-dimensional grids through `simbi launch`. the launched worker path is
# dimension-generic: tiles cut along one, two or three axes meet at edges and
# corners, and every cut halo is exchanged in axis order so the corners
# carry the neighbor's cells. the relativistic shock tube on two workers and
# on four (cut along x and z) equals the single-grid run, fresh and with the
# correction fired on the cell where the shock meets both cuts; the tabulated
# heating source, read at each cell's global position, equals the single
# grid on four workers fresh and with the correction in the heated core. the
# newtonian orszag-tang vortex with a z velocity, under the Contact and the UCT
# edge EMF, equals the single-grid run on four workers cut along x and z, every
# staggered face included, fresh and with the correction fired on a cell against
# both cuts.
# the single-grid references and the launches run on the session's backend
# and placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_3d.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

from pathlib import Path

import pytest

from simbi_configs.examples.newtonian.nmhd_orszag_tang_3d import NewtonianOrszagTang3D
from simbi_configs.examples.newtonian.tabulated_source_3d import TabulatedSource3D
from simbi_configs.examples.srhd.marti_muller_3d import MartiMuller3D

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import (
    STEPS,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation


def _script_of(cls: type) -> str:
    return str(Path(cls.__module__.replace(".", "/") + ".py"))


SHOCK_SCRIPT = _script_of(MartiMuller3D)
HEATING_SCRIPT = _script_of(TabulatedSource3D)
VORTEX_SCRIPT = _script_of(NewtonianOrszagTang3D)


def test_a_three_dimensional_shock_tube_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    resolution = (16, 16, 16)
    reference = _script_reference(MartiMuller3D.__module__, "MartiMuller3D", resolution, tmp_path / "single", STEPS)
    two = _layout(tmp_path / "two-layout.toml", workers=2, shape=(2, 1, 1))
    launched = _script_launch(SHOCK_SCRIPT, resolution, tmp_path / "two", two, STEPS, workers=2)
    ph.same_checkpoint(launched, reference, "two workers on the shock tube")
    four = _layout(tmp_path / "four-layout.toml", workers=4, shape=(2, 1, 2))
    launched = _script_launch(SHOCK_SCRIPT, resolution, tmp_path / "four", four, STEPS, workers=4)
    ph.same_checkpoint(launched, reference, "four workers, cuts along x and z")
    # the flagged cell (7, 3, 8) sits on the shock against the x cut at 8 and on the z cut at 8
    one = _layout(tmp_path / "one-layout.toml", workers=1, shape=(1, 1, 1))
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:0:7:3:8"}
    clean = _script_launch(SHOCK_SCRIPT, resolution, tmp_path / "clean", one, STEPS, workers=1)
    flagged = _script_launch(SHOCK_SCRIPT, resolution, tmp_path / "flagged", one, STEPS, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(SHOCK_SCRIPT, resolution, tmp_path / "corrected", four, STEPS, workers=4, env=inject)
    ph.same_checkpoint(corrected, flagged, "four workers, the correction where the shock meets both cuts")


def test_a_three_dimensional_tabulated_source_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    resolution = (12, 10, 8)
    reference = _script_reference(TabulatedSource3D.__module__, "TabulatedSource3D", resolution, tmp_path / "single", STEPS)
    four = _layout(tmp_path / "four-layout.toml", workers=4, shape=(2, 1, 2))
    launched = _script_launch(HEATING_SCRIPT, resolution, tmp_path / "four", four, STEPS, workers=4)
    ph.same_checkpoint(launched, reference, "four workers under the tabulated heating")
    # the flagged cell (5, 4, 3) sits in the heated core against the x cut at 6
    one = _layout(tmp_path / "one-layout.toml", workers=1, shape=(1, 1, 1))
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:1:5:4:3"}
    clean = _script_launch(HEATING_SCRIPT, resolution, tmp_path / "clean", one, STEPS, workers=1)
    flagged = _script_launch(HEATING_SCRIPT, resolution, tmp_path / "flagged", one, STEPS, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(HEATING_SCRIPT, resolution, tmp_path / "corrected", four, STEPS, workers=4, env=inject)
    ph.same_checkpoint(corrected, flagged, "four workers, the correction in the heated core")


@pytest.mark.parametrize(("flags", "fields"), [([], ""), (["--ct-method", "uct"], "ct_method='uct'")], ids=["contact", "uct"])
def test_a_three_dimensional_magnetized_launch_matches_the_single_grid_run(tmp_path: Path, flags: list[str], fields: str) -> None:
    resolution = (16, 16, 16)
    reference = _script_reference(NewtonianOrszagTang3D.__module__, "NewtonianOrszagTang3D", resolution, tmp_path / "single", STEPS, fields)
    four = _layout(tmp_path / "four-layout.toml", workers=4, shape=(2, 1, 2))
    launched = _script_launch(VORTEX_SCRIPT, resolution, tmp_path / "four", four, STEPS, workers=4, flags=flags)
    ph.same_checkpoint(launched, reference, "four workers on the 3d vortex, cuts along x and z")
    # the flagged cell (7, 3, 7) sits against the x cut at 8 and the z cut at 8
    one = _layout(tmp_path / "one-layout.toml", workers=1, shape=(1, 1, 1))
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:0:7:3:7"}
    clean = _script_launch(VORTEX_SCRIPT, resolution, tmp_path / "clean", one, STEPS, workers=1, flags=flags)
    flagged = _script_launch(VORTEX_SCRIPT, resolution, tmp_path / "flagged", one, STEPS, workers=1, env=inject, flags=flags)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(VORTEX_SCRIPT, resolution, tmp_path / "corrected", four, STEPS, workers=4, env=inject, flags=flags)
    ph.same_checkpoint(corrected, flagged, "four workers, the correction against both cuts of the 3d vortex")
