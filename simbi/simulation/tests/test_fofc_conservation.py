# =============================================================================
# test_fofc_conservation.py
#
# exact conservation of the first-order flux-correction fallback. on a periodic
# cartesian grid every face-telescoping finite-volume update conserves the total
# D, S, tau to accumulated roundoff (measured ~1e-14 relative over ~1e3 steps on
# a NON-firing strong blast). a fallback that replaces the STATE of a flagged
# cell (rather than the FLUX on its faces, re-updating both neighbors) applies
# two different fluxes to the two sides of every flag-boundary face and
# creates/destroys conserved quantities there; the freeze tier discards a
# cell's flux exchange entirely while its neighbors keep theirs. the colliding
# ultra-relativistic streams fire the fallback at the collision shocks, so the
# per-cell (non-face-consistent) select shows up as a total-conservation drift
# orders of magnitude above roundoff (measured ~2e-3 relative in D over
# t = 0.1 — a 0.2% mass loss).
#
# xfail(strict): the current fallback is per-cell state replacement, which is
# non-conservative at flag boundaries. the face-based redo (one flux per face,
# both neighbors re-updated from it) must flip this gate to green.
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np
import pytest

from simbi.simulation import runner
from simbi.simulation.tests.fixtures.fofc_periodic_blast import FofcPeriodicBlast

_GAMMA = 4.0 / 3.0


def _totals(rho: np.ndarray, v: np.ndarray, pre: np.ndarray) -> tuple[float, float, float]:
    """total conserved (D, S, tau) per unit cell volume from primitives (flat 1d)."""
    w = 1.0 / np.sqrt(1.0 - v * v)
    h = 1.0 + _GAMMA / (_GAMMA - 1.0) * pre / rho
    d = rho * w
    s = rho * h * w * w * v
    tau = rho * h * w * w - pre - d
    return float(d.sum()), float(s.sum()), float(tau.sum())


@pytest.mark.xfail(
    strict=True,
    reason="fofc per-cell select is non-conservative at flag boundaries; the "
    "face-based flux redo must restore exact telescoping",
)
def test_fofc_periodic_blast_conserves_totals() -> None:
    d = tempfile.mkdtemp() + "/"
    p = FofcPeriodicBlast.from_cli([])
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    runner.run(p, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*")), "run crashed"
    final = glob.glob(os.path.join(d, "*final*.h5"))
    assert final, "no final checkpoint written"
    with h5py.File(final[0]) as h:
        prim = h["level_0/partition_0/hydro/primitives"]
        rho = np.asarray(prim["rho"][:], dtype=np.float64)
        pre = np.asarray(prim["pre"][:], dtype=np.float64)
        vkey = "v1" if "v1" in prim else "v"
        v = np.asarray(prim[vkey][:], dtype=np.float64)
    # the checkpoint stores the ALLOCATED grid (ghost halo included); the conservation
    # invariant holds over the interior only.
    ng = (len(rho) - p.resolution) // 2
    assert ng > 0, f"expected a ghost halo (got {len(rho)} cells for resolution {p.resolution})"
    rho, v, pre = rho[ng:-ng], v[ng:-ng], pre[ng:-ng]

    d0, s0, tau0 = _totals(*(np.asarray(x, dtype=np.float64) for x in _initial_state()))
    d1, s1, tau1 = _totals(rho, v, pre)

    # scale-relative drifts; S starts at zero so it is scaled by the mass total.
    dd = abs(d1 - d0) / abs(d0)
    ds = abs(s1 - s0) / abs(d0)
    dtau = abs(tau1 - tau0) / abs(tau0)
    tol = 1e-11
    assert dd < tol and ds < tol and dtau < tol, (
        f"periodic totals drifted: dD/D={dd:.3e}, dS/D={ds:.3e}, dtau/tau={dtau:.3e} "
        f"(face-telescoping conservation bound {tol:.0e})"
    )


def _initial_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """the fixture's initial (rho, v, pre) arrays."""
    prob = FofcPeriodicBlast.from_cli([])
    gas = np.array(list(prob.initial_primitive_state()()), dtype=np.float64)
    return gas[:, 0], gas[:, 1], gas[:, 2]
