# =============================================================================
# test_fmr_covered_restriction.py
#
# a refined run's checkpoint must show the covered coarse CONSERVED cells equal
# to the conservative restriction (2x2 average) of their fine children to
# round-off: the hierarchy restricts at the tail of every root step, and the
# writer serializes that synced state. this gates the whole chain — step-loop
# restriction, checkpoint writer, and the per-level halo metadata (the ghost
# width DIFFERS by level: reconstruct it from the array shape minus
# mesh/global_cells, never assume a uniform strip — a uniform strip shifts the
# fine array one cell diagonally and fakes a percent-level mismatch on every
# field). derived velocities are NOT compared: restrict(m)/restrict(rho)
# differs from restrict(m/rho) by the averaging nonlinearity, by design.
# =============================================================================
import glob
import tempfile

import h5py
import numpy as np

from simbi.simulation import runner
from simbi_configs.examples.isothermal.refined_locally_iso import (
    RefinedLocallyIso,
)


def _interior(h: h5py.File, lvl: int, path: str) -> np.ndarray:
    arr = h[f"level_{lvl}/{path}"][:]
    ncells = int(h[f"level_{lvl}/mesh/global_cells"][0])
    ng = int((arr.shape[0] - ncells) // 2)
    return arr[ng:-ng, ng:-ng]


def _restrict(fine: np.ndarray) -> np.ndarray:
    return 0.25 * (
        fine[0::2, 0::2] + fine[1::2, 0::2] + fine[0::2, 1::2] + fine[1::2, 1::2]
    )


def test_covered_coarse_equals_restricted_fine_in_checkpoint() -> None:
    d = tempfile.mkdtemp() + "/"
    p = RefinedLocallyIso(data_directory=d, checkpoint_interval=1.0e30)
    runner.run(p, compute_mode="cpu", max_steps=20)

    final = glob.glob(d + "*final*.h5")
    assert final, "no final checkpoint written"
    with h5py.File(final[0]) as h:
        # the refinement region [0.25, 0.75]^2 on the 64^2 base grid: coarse
        # cells [16, 48) are covered, and the 64^2 fine level spans exactly them.
        i0, i1 = 16, 48
        for nm in ("den", "m1", "m2"):
            coarse = _interior(h, 0, f"conserved/{nm}")[i0:i1, i0:i1]
            fine_r = _restrict(_interior(h, 1, f"conserved/{nm}"))
            scale = np.abs(fine_r).max()
            assert scale > 0.0, f"{nm}: fine level is identically zero; test is vacuous"
            rel = np.abs(coarse - fine_r).max() / scale
            assert rel < 1e-13, (
                f"covered coarse '{nm}' is not the restriction of its fine "
                f"children: max rel mismatch {rel:.3e}"
            )
