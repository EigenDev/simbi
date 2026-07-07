# =============================================================================
# test_mhd_brio_wu_1p5d.py
#
# the reduced-dimension out-of-plane predictor gate (oop_predictor_spec.md). on a 1.5D
# newtonian-MHD Brio-Wu shock tube (D=1, DOF=3) the transverse By,Bz have NO staggered
# face and are cell-centered conserved variables evolved SOLELY by the out-of-plane cell-B
# flux predictor. before the predictor was restored, By was frozen at its sharp +-1 IC
# jump, which drove the gas pressure negative; the correct out-of-plane evolution develops
# the Brio-Wu compound wave (By reverses through intermediate values) and stays physical.
#
# asserts: Bx const to machine (the no-CT 1.5D property), physicality everywhere, the
# unshocked end states survive, the compound wave reverses By through intermediate values,
# and the rarefaction/contact drives the density well below the left state.
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np

from simbi.simulation import runner
from simbi.simulation.tests.fixtures.mhd_brio_wu_1p5d import MhdBrioWu1p5d


def _run(nx: int) -> dict:
    d = tempfile.mkdtemp() + "/"
    p = MhdBrioWu1p5d.from_cli([])
    p.resolution = (nx, 1, 1)
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    p.end_time = 0.1
    runner.run(p, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*")), "1.5D Brio-Wu run crashed"
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    with h5py.File(final) as h:
        pr = h["level_0/partition_0/hydro/primitives"]
        ng = (pr["rho"].shape[0] - nx) // 2
        sl = slice(ng, pr["rho"].shape[0] - ng)
        return {k: np.asarray(pr[k][sl], dtype=np.float64) for k in ("rho", "pre", "b1", "b2", "b3")}


def test_brio_wu_1p5d_out_of_plane_predictor() -> None:
    nx = 400
    f = _run(nx)
    rho, pre, bx, by, bz = f["rho"], f["pre"], f["b1"], f["b2"], f["b3"]

    # the crux of the no-CT 1.5D scheme: the normal field Bx is never curled -> exactly constant.
    assert np.max(np.abs(bx - 0.75)) < 1e-12, f"Bx drifted from 0.75 (max dev {np.max(np.abs(bx - 0.75)):.2e})"
    # out-of-plane Bz stays identically zero (no source seeds it).
    assert np.max(np.abs(bz)) < 1e-12, f"Bz grew from zero (max {np.max(np.abs(bz)):.2e})"

    # physicality everywhere — a frozen By drives p < 0 (the regression signature).
    assert np.all(np.isfinite(rho)) and np.all(rho > 0.05), "rho non-finite or unphysical"
    assert np.all(np.isfinite(pre)) and np.all(pre > 0.0), "pre non-finite or non-positive"

    # unshocked end states survive at the boundaries (waves have not reached them at t=0.1).
    assert abs(rho[3] - 1.0) < 0.05, f"left end rho={rho[3]:.4f} (expected ~1.0)"
    assert abs(rho[-4] - 0.125) < 0.02, f"right end rho={rho[-4]:.4f} (expected ~0.125)"
    assert abs(by[3] - 1.0) < 0.05, f"left end By={by[3]:.4f} (expected ~1.0)"
    assert abs(by[-4] + 1.0) < 0.05, f"right end By={by[-4]:.4f} (expected ~-1.0)"

    # the compound wave: By transitions +1 -> -1, so it changes sign in the interior AND passes
    # through intermediate values. a frozen By keeps the sharp +-1 step (no intermediate values).
    assert int(np.sum(by[:-1] * by[1:] < 0)) >= 1, "By never changed sign — compound-wave structure absent"
    assert np.any(np.abs(by) < 0.9), "By took no intermediate value — it is frozen at the +-1 IC jump"

    # the rarefaction/contact drove the density well below the left unshocked value.
    assert rho.min() < 0.6, f"solution did not develop the rarefaction/contact drop: rho_min={rho.min():.4f}"
