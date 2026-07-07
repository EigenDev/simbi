# =============================================================================
# test_mhd_energy_conservation.py
#
# the base-scheme CT energy-conservation gate (spec §6). on a periodic magnetized
# relativistic shock the total energy tau (conserved buffer, interior sum) must hold to
# machine roundoff — AND not grow with resolution. before the fix the magnetic-energy
# patch (applied outside the flux) drifted tau ~2e-4 at nx=256 and GREW to ~6e-4 at
# nx=512. after making cell B a derived quantity (interp of the CT face field) and
# deleting the patch, tau is conserved by the Poynting-carrying Godunov flux to roundoff.
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np

from simbi.simulation import runner
from simbi.simulation.tests.fixtures.mhd_energy_conservation import MhdEnergyConservation


def _tau_sum(nx: int, end_time: float) -> float:
    d = tempfile.mkdtemp() + "/"
    p = MhdEnergyConservation.from_cli([])
    p.resolution = (nx, 8, 1)
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    p.end_time = end_time
    runner.run(p, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*")), "run crashed"
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    with h5py.File(final) as h:
        nrg = np.asarray(h["level_0/conserved/nrg"][:], dtype=np.float64)
    ngi = (nrg.shape[1] - nx) // 2
    ngj = (nrg.shape[0] - 8) // 2
    return float(nrg[ngj : nrg.shape[0] - ngj, ngi : nrg.shape[1] - ngi].sum())


def test_mhd_total_energy_conserved_and_resolution_independent() -> None:
    tol = 1e-11
    drifts = {}
    # the fix must hold at EVERY resolution (the bug GREW with nx — the discriminating signature).
    for nx in (64, 128, 256):
        t0 = _tau_sum(nx, 0.0)
        t1 = _tau_sum(nx, 0.1)
        drifts[nx] = abs(t1 - t0) / abs(t0)
    for nx, d in drifts.items():
        assert d < tol, (
            f"total energy drifted at nx={nx}: dtau/tau={d:.3e} (roundoff-conservation bound "
            f"{tol:.0e}) — the CT magnetic-energy patch is non-conservative"
        )
    # and the drift must NOT grow with resolution (a conservative scheme is flat; the patch grew).
    assert drifts[256] < 10.0 * max(drifts[64], 1e-16), (
        f"energy drift grows with resolution ({drifts[64]:.2e} -> {drifts[256]:.2e}) — "
        "the hallmark of the non-conservative magnetic-energy patch, not truncation"
    )
