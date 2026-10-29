# =============================================================================
# test_initial_recovery_report.py
#
# the report of a refined run whose initial conserved-to-primitive recovery
# rejects a cell: the 3D refined isothermal MHD vortex with one root cell at
# negative density stops before its first step, and the message names the
# cell's raw status code, the interior rule re-evaluated on the printed values
# (rejected here), the conserved, cell and face fields the recovery read, and
# the isothermal closure at the cell.
#
# run: pytest -m simulation simbi/simulation/tests/test_initial_recovery_report.py
#      (SIMBI_PARITY_BACKEND selects the session's backend, parity_harness.py)
# =============================================================================
from pathlib import Path

import pytest

from simbi.simulation.tests import parity_harness as ph

pytestmark = pytest.mark.simulation

VORTEX = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi_configs.examples.newtonian.refined_mhd_3d import RefinedMhd3D


class NegativeCell(RefinedMhd3D):
    """the isothermal MHD vortex at 24^3 under one box, one root cell, the second, at negative density (the frontend checks the first)"""

    resolution: Annotated[tuple, ProblemParam((24, 24, 24), description="grid")]
    regime: Annotated[Regime, ProblemParam(Regime.IMHD, description="isothermal mhd")]
    sound_speed: Annotated[float, ProblemParam(1.0, description="isothermal sound speed")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.25, 0.75]], description="box")]

    def initial_primitive_state(self):
        state = super().initial_primitive_state()
        gas, faces = (state[0], state[1:]) if isinstance(state, tuple) else (state, ())

        def isothermal():
            for index, row in enumerate(gas()):
                row = row[:-1]
                yield (-1.0, *row[1:]) if index == 1 else row

        return (isothermal, *faces) if faces else isothermal
'''


def test_a_rejected_initial_cell_reports_its_inputs_and_the_rule(tmp_path: Path) -> None:
    module = tmp_path / "negative_cell.py"
    module.write_text(VORTEX)
    snippet = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from simbi.simulation import runner\n"
        "from negative_cell import NegativeCell\n"
        "p = NegativeCell(data_directory=sys.argv[1])\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps=1)\n"
    )
    done = ph.python(ph.snippet_args(snippet, str(tmp_path / "out"), str(tmp_path)), "negative initial cell", check=False)
    flat = " ".join(done.stderr.split())
    assert done.returncode != 0, "the run with a negative-density cell succeeded"
    assert "c2p failed on initial conditions" in flat, flat[-2000:]
    for field in ("(code ", "interior rule on these values: rejected", "rho=-1.0", "bcell=Some(", "bface=Some(", "cs2=Some(1.0"):
        assert field in flat, f"the report carries no {field!r}: {flat[-2000:]}"
