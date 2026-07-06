# =============================================================================
# test_fofc_conservation.py
#
# exact conservation of the first-order flux-correction fallback. on a periodic
# cartesian grid every face-telescoping finite-volume update conserves the total
# D, S, tau in the CONSERVED buffer to accumulated roundoff. the FOFC fallback
# must preserve that: the face-based redo splices the first-order flux onto only
# the faces adjacent to a flagged (fallback) cell and re-runs ONE godunov, so
# every face still carries a single flux and the sum telescopes. the superseded
# per-cell state replacement applied two different fluxes to the two sides of a
# flag-boundary face and created/destroyed conserved quantities there.
#
# the metric is the CONSERVED buffer (level_0/conserved), NOT the checkpoint's
# primitives: reconstructing D = rho*W from the stored (rho, v) amplifies the
# c2p round-trip error by dW/dv ~ W^3*v, which for the W ~ 7 collision here is
# ~1e-3 even for a perfectly conservative scheme — it measures c2p accuracy, not
# conservation. the conserved buffer is the quantity the finite-volume update
# actually telescopes.
#
# the fixture is tuned so FOFC fires on a few hundred substages but the
# first-order tier recovers every flagged cell (no freeze — the freeze tier is
# the one deliberately non-conservative FOFC operation, a bounded waiver), so a
# correct face-based redo holds the totals to roundoff.
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np

from simbi.simulation import runner
from simbi.simulation.tests.fixtures.fofc_periodic_blast import FofcPeriodicBlast


def _conserved_totals(end_time: float) -> tuple[float, float, float]:
    """run the fixture to end_time, return the interior sums of the conserved
    (D, S, tau) read straight from the checkpoint's conserved buffer."""
    d = tempfile.mkdtemp() + "/"
    prob = FofcPeriodicBlast.from_cli([])
    prob.data_directory = d
    prob.checkpoint_interval = 1.0e30
    prob.end_time = end_time
    runner.run(prob, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*")), "run crashed"
    final = glob.glob(os.path.join(d, "*final*.h5"))
    assert final, "no final checkpoint written"
    with h5py.File(final[0]) as h:
        c = h["level_0/conserved"]
        den = np.asarray(c["den"][:], dtype=np.float64)
        mom = np.asarray(c["m1"][:], dtype=np.float64)
        nrg = np.asarray(c["nrg"][:], dtype=np.float64)
    # the checkpoint stores the ALLOCATED grid (ghost halo included); the
    # conservation invariant holds over the interior only.
    ng = (len(den) - prob.resolution) // 2
    assert ng > 0, f"expected a ghost halo (got {len(den)} cells for {prob.resolution})"
    sl = slice(ng, -ng)
    return float(den[sl].sum()), float(mom[sl].sum()), float(nrg[sl].sum())


def test_fofc_periodic_blast_conserves_totals() -> None:
    # the code's own conserved buffer at t = 0 (its p2c of the initial primitives)
    # is the reference — reference-free of any analytic p2c mismatch.
    d0, s0, tau0 = _conserved_totals(0.0)
    d1, s1, tau1 = _conserved_totals(0.1)

    # scale-relative drifts; S starts near zero so it is scaled by the mass total.
    dd = abs(d1 - d0) / abs(d0)
    ds = abs(s1 - s0) / abs(d0)
    dtau = abs(tau1 - tau0) / abs(tau0)
    tol = 1e-11
    assert dd < tol and ds < tol and dtau < tol, (
        f"periodic conserved totals drifted: dD/D={dd:.3e}, dS/D={ds:.3e}, "
        f"dtau/tau={dtau:.3e} (face-telescoping conservation bound {tol:.0e})"
    )
