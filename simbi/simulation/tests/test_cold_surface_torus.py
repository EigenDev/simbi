# =============================================================================
# test_cold_surface_torus.py
#
# the inviscid adiabatic torus through the frontend (fixtures/cold_surface_torus.py): the
# run reaches t = 0.02 and every interior cell of its final state holds a positive density
# and internal energy. the stage replay of the same state is
# crates/symbi/tests/cold_surface_stage_replay.rs.
#
# run: pytest -m simulation simbi/simulation/tests/test_cold_surface_torus.py
# =============================================================================
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph

pytestmark = pytest.mark.simulation


def test_the_cold_torus_surface_stays_admissible_to_the_end_time(tmp_path: Path) -> None:
    out = tmp_path / "run"
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from simbi.simulation.tests.fixtures.cold_surface_torus import ColdSurfaceTorus\n"
        f"runner.run(ColdSurfaceTorus(data_directory=sys.argv[1]), compute_mode={ph.BACKEND!r})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out)), "the cold surface torus")
    final = next(out.glob("*final*.h5"))
    with h5py.File(final, "r") as f:
        level = f["level_0"]
        assert float(level.attrs["time"]) >= 0.02, f"the run stopped at t = {level.attrs['time']}"
        ng = int(level["mesh"].attrs["halo_width"])
        den, m1, m2, m3, nrg = (level[f"conserved/{k}"][()][ng:-ng, ng:-ng] for k in ("den", "m1", "m2", "m3", "nrg"))
    internal = nrg - 0.5 * (m1**2 + m2**2 + m3**2) / den
    assert np.all(den > 0.0), f"the density reached {den.min()}"
    assert np.all(internal > 0.0), f"the internal energy reached {internal.min()}"
