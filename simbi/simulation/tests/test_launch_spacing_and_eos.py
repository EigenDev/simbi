# =============================================================================
# test_launch_spacing_and_eos.py
#
# two grid and closure choices through `simbi launch`, each on four workers cut
# along both axes against the single-grid run, bit for bit, fresh and with the
# correction fired on a cell against both cuts:
# - geometric spacing: newtonian hydro on a 2D cartesian grid whose widths grow
#   geometrically on both axes, under a tabulated source read at each cell's
#   global position;
# - the synge perfect gas: a relativistic blast whose hot core walks the
#   effective adiabatic index toward 4/3 while the cold medium stays at 5/3.
# the single-grid references and the launches run on the session's backend
# and placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_spacing_and_eos.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

from pathlib import Path

import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import (
    STEPS,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation

# (module, class, resolution, the flagged cell against the cuts at the grid's center)
CASES = {
    "geometric": ("simbi_configs.examples.newtonian.decomposed_tabulated_geometric", "DecomposedTabulatedGeometric", (32, 16), "trouble:3:0:15:7"),
    "synge": ("simbi_configs.examples.srhd.synge_blast", "SyngeBlast", (32, 32), "trouble:3:0:15:15"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_a_launch_matches_the_single_grid_run_fresh_and_corrected(tmp_path: Path, name: str) -> None:
    module, cls, resolution, trouble = CASES[name]
    script = module.replace(".", "/") + ".py"
    reference = _script_reference(module, cls, resolution, tmp_path / "single", STEPS)
    four = _layout(tmp_path / "four-layout.toml", workers=4, shape=(2, 2))
    launched = _script_launch(script, resolution, tmp_path / "four", four, STEPS, workers=4)
    ph.same_checkpoint(launched, reference, f"{name}: four workers")
    one = _layout(tmp_path / "one-layout.toml", workers=1, shape=(1, 1))
    inject = {"SIMBI_FABRIC_INJECT": trouble}
    clean = _script_launch(script, resolution, tmp_path / "clean", one, STEPS, workers=1)
    flagged = _script_launch(script, resolution, tmp_path / "flagged", one, STEPS, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(script, resolution, tmp_path / "corrected", four, STEPS, workers=4, env=inject)
    ph.same_checkpoint(corrected, flagged, f"{name}: four workers, the correction against both cuts")
