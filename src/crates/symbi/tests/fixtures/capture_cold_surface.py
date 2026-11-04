# =============================================================================
# capture_cold_surface.py
#
# writes cold_surface_torus_84x44_step2000.bin: the interior conserved state of the cold surface
# torus (simbi/simulation/tests/fixtures/cold_surface_torus.py) at step 2000, as little-endian
# f64: the time, then den, m1, m2, m3, nrg, each ni * nj values with i fastest. the frontend's
# restored state is the exact state the run continues from, so a replay needs no history.
#
# usage:
#   python crates/symbi/tests/fixtures/capture_cold_surface.py <work dir>
# =============================================================================
import sys
from pathlib import Path

import h5py
import numpy as np

from simbi.simulation import runner
from simbi.simulation.tests.fixtures.cold_surface_torus import ColdSurfaceTorus

STEP = 2000
work = Path(sys.argv[1])
runner.run(ColdSurfaceTorus(data_directory=str(work)), compute_mode="cpu", max_steps=STEP)
with h5py.File(next(work.glob("*final*.h5")), "r") as f:
    level = f["level_0"]
    assert int(level.attrs["iteration"]) == STEP
    ng = int(level["mesh"].attrs["halo_width"])
    fields = [level[f"conserved/{k}"][()][ng:-ng, ng:-ng] for k in ("den", "m1", "m2", "m3", "nrg")]
    time = float(level.attrs["time"])
target = Path(__file__).with_name(f"cold_surface_torus_84x44_step{STEP}.bin")
target.write_bytes(np.concatenate([[time], *[a.ravel() for a in fields]]).astype("<f8").tobytes())
print(f"{target}: t = {time!r}")
