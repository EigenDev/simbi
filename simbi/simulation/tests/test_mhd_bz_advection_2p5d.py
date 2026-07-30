# =============================================================================
# test_mhd_bz_advection_2p5d.py
#
# the 2.5D out-of-plane predictor gate. the transverse Bz on a 2D
# cartesian grid has no staggered face (only Bx,By are CT) and is a cell-centered conserved
# variable evolved SOLELY by the out-of-plane cell-B flux predictor. a force-balanced Bz(x,y)
# advects rigidly at the flow velocity; after a HALF period the exact solution is the IC
# shifted by half the domain in each direction. a working predictor lands on that shift; a
# frozen predictor leaves Bz at the IC.
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np

from simbi.simulation import runner
from simbi.simulation.tests.fixtures.mhd_bz_advection_2p5d import MhdBzAdvection2p5d


def test_bz_advection_2p5d_out_of_plane_predictor() -> None:
    nx = ny = 64
    d = tempfile.mkdtemp() + "/"
    p = MhdBzAdvection2p5d.from_cli([])
    p.resolution = (nx, ny, 1)
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    p.end_time = 0.5  # half period -> the IC shifted by (0.5, 0.5)
    runner.run(p, compute_mode="cpu", max_steps=4000)
    assert not glob.glob(os.path.join(d, "*crashed*")), "2.5D Bz advection run crashed"
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    with h5py.File(final) as h:
        pr = h["level_0/partition_0/hydro/primitives"]
        rho = np.asarray(pr["rho"], dtype=np.float64)
        pre = np.asarray(pr["pre"], dtype=np.float64)
        bz = np.asarray(pr["b3"], dtype=np.float64)
    ny_g, nx_g = rho.shape
    ng = (ny_g - ny) // 2
    sl = (slice(ng, ny_g - ng), slice(ng, nx_g - ng))
    rho, pre, bz = rho[sl], pre[sl], bz[sl]

    # force balance holds and the state stays physical.
    assert np.all(np.isfinite(rho)) and np.all(np.isfinite(pre)) and np.all(np.isfinite(bz))
    assert np.all(pre > 0.0), "gas pressure went non-positive"
    assert np.max(np.abs(rho - 1.0)) < 5e-2, f"density drifted from the advecting equilibrium (max dev {np.max(np.abs(rho - 1.0)):.2e})"

    xc = (np.arange(nx) + 0.5) / nx
    yc = (np.arange(ny) + 0.5) / ny
    bz_ic = 1.0 + 0.3 * np.sin(2 * np.pi * xc)[None, :] + 0.2 * np.cos(2 * np.pi * yc)[:, None]
    bz_shift = 1.0 + 0.3 * np.sin(2 * np.pi * (xc - 0.5))[None, :] + 0.2 * np.cos(2 * np.pi * (yc - 0.5))[:, None]
    l1_shift = float(np.mean(np.abs(bz - bz_shift)))
    l1_ic = float(np.mean(np.abs(bz - bz_ic)))

    # the predictor advected Bz onto the analytic half-period shift (small truncation error).
    assert l1_shift < 1.5e-2, f"Bz did not advect to the analytic half-period shift: L1={l1_shift:.3e}"
    # and it genuinely MOVED — a frozen out-of-plane predictor would leave Bz at the IC (L1~0).
    assert l1_ic > 0.2, f"Bz did not move from its IC (L1={l1_ic:.3e}) — the out-of-plane predictor is frozen"
