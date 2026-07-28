# =============================================================================
# test_mhd_oop_energy_conservation.py
#
# the reduced-dimension OUT-OF-PLANE energy-conservation gate (spec §6 / oop_predictor_spec.md).
# on a smooth periodic 1.5D relativistic-MHD flow whose transverse By is a cell-centered
# conserved variable (flux-evolved by the out-of-plane predictor), the
# total energy tau must hold to machine roundoff at EVERY resolution. this is the companion to
# test_mhd_energy_conservation (which covers the in-plane/CT energy with a zero out-of-plane
# field): it witnesses that the Poynting gas flux F_tau conserves the out-of-plane magnetic
# energy By^2/2 exactly, with NO magnetic-energy patch — the property that lets the delicate
# relativistic c2p recover a physical state (a drifting tau would desync from |B|^2 and fail).
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np

from simbi.simulation import runner
from simbi.simulation.tests.fixtures.mhd_oop_energy_1p5d import MhdOopEnergy1p5d


def _tau_sum(nx: int, end_time: float) -> float:
    d = tempfile.mkdtemp() + "/"
    p = MhdOopEnergy1p5d.from_cli([])
    p.resolution = (nx, 1, 1)
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    p.end_time = end_time
    runner.run(p, compute_mode="cpu", max_steps=400)
    assert not glob.glob(os.path.join(d, "*crashed*")), "run crashed"
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    with h5py.File(final) as h:
        nrg = np.asarray(h["level_0/conserved/nrg"][:], dtype=np.float64)
    ng = (nrg.shape[0] - nx) // 2
    return float(nrg[ng : nrg.shape[0] - ng].sum())


def test_out_of_plane_energy_conserved_and_resolution_independent() -> None:
    tol = 1e-11
    drifts = {}
    for nx in (64, 128, 256):
        t0 = _tau_sum(nx, 0.0)
        t1 = _tau_sum(nx, 1.0)
        drifts[nx] = abs(t1 - t0) / abs(t0)
    for nx, drift in drifts.items():
        assert drift < tol, (
            f"out-of-plane total energy drifted at nx={nx}: dtau/tau={drift:.3e} "
            f"(roundoff-conservation bound {tol:.0e}) — the Poynting flux is not conserving "
            "the cell-centered out-of-plane magnetic energy"
        )
    # a conservative scheme is flat in resolution (a non-conservative bookkeeping error would grow).
    assert drifts[256] < 10.0 * max(drifts[64], 1e-16), (
        f"energy drift grows with resolution ({drifts[64]:.2e} -> {drifts[256]:.2e})"
    )
