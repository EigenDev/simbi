# =============================================================================
# capture_outward_wall_surface.py
#
# writes outward_wall_surface_n84_step2061947.bin and its method record (.json): the state of
# the inviscid SPB torus (simbi_configs/examples/newtonian/spb_torus.py, run N, 84 x 44,
# outward_outer) at step 2061947, t = 9.600708849610989, 1043 steps before two surface cells
# reach negative internal energy. the binary is little-endian f64: the time, the iteration, the
# halo width, the stored rows and columns, then den, m1, m2, m3, nrg and rho, v1, v2, v3, pre,
# each with its halo, columns fastest. a restart file rebuilt from it continues the run bit for bit.
#
# usage:
#   python crates/symbi/tests/fixtures/capture_outward_wall_surface.py <checkpoint at step 2061947>
# =============================================================================
import json
import sys
from pathlib import Path

import h5py
import numpy as np

STEP = 2061947
CONSERVED = ("den", "m1", "m2", "m3", "nrg")
PRIMITIVE = ("rho", "v1", "v2", "v3", "pre")
METHOD = ("regime", "spacetime", "coord_system", "eos", "gamma", "solver", "reconstruction", "plm_theta", "timestepping", "cfl",
          "boundary_conditions", "x1_spacing", "wb_reconstruction", "viscosity", "dimensions", "build_git_sha")

with h5py.File(sys.argv[1], "r") as f:
    level = f["level_0"]
    assert int(level.attrs["iteration"]) == STEP, level.attrs["iteration"]
    ng = int(level["mesh"].attrs["halo_width"])
    arrays = [level[f"conserved/{k}"][()] for k in CONSERVED] + [level[f"partition_0/hydro/primitives/{k}"][()] for k in PRIMITIVE]
    rows, cols = arrays[0].shape
    header = [float(level.attrs["time"]), float(STEP), float(ng), float(rows), float(cols)]
    method = {k: str(f["metadata"].attrs[k]) for k in METHOD}
stem = Path(__file__).with_name(f"outward_wall_surface_n84_step{STEP}")
stem.with_suffix(".bin").write_bytes(np.concatenate([header, *[a.ravel() for a in arrays]]).astype("<f8").tobytes())
stem.with_suffix(".json").write_text(json.dumps(method, indent=1, sort_keys=True) + "\n")
print(f"{stem}.bin: t = {header[0]!r}, {rows} x {cols} with halo {ng}")
