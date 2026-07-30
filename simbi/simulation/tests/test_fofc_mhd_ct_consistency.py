# =============================================================================
# test_fofc_mhd_ct_consistency.py
#
# a gas-only FOFC redo must leave the CELL B consistent with the (unchanged,
# high-order) staggered FACE field, bcell == interp(bface). the shared face-based redo
# re-advances cell B from the first-order induction flux; without the constrained-
# transport re-sync (restore the high-order induction flux for the cell-B predictor +
# re-run bcell_from_bface) that cell B diverges from interp(bface) by 2.75e-2 on a
# firing corrector, while the re-sync holds the two together to roundoff.
#
# flat cartesian makes interp(bface) the trivial 0.5 face-average (no metric weight),
# so the invariant is checked externally. the run is SHORT (deterministic; fires FOFC
# on a corrector without hitting the persistent-freeze halt).
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np

from simbi.simulation import runner
from simbi.simulation.tests.fixtures.fofc_mhd_ct_consistency import FofcMhdCtConsistency


def _read(fn: str):
    with h5py.File(fn) as h:
        c = h["level_0/conserved"]
        b1c = np.asarray(c["b1"][:], dtype=np.float64)  # cell Bx (allocated, +ghosts)
        b2c = np.asarray(c["b2"][:], dtype=np.float64)  # cell By
        den = np.asarray(c["den"][:], dtype=np.float64)
        mom = np.asarray(c["m1"][:], dtype=np.float64)
        B1 = np.asarray(h["level_0/partition_0/hydro/magnetic/B1/data"][:], dtype=np.float64)  # x-faces
        B2 = np.asarray(h["level_0/partition_0/hydro/magnetic/B2/data"][:], dtype=np.float64)  # y-faces
    return b1c, b2c, den, mom, B1, B2


def _run(end_time: float) -> tuple[FofcMhdCtConsistency, str]:
    d = tempfile.mkdtemp() + "/"
    p = FofcMhdCtConsistency.from_cli([])
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    p.end_time = end_time
    runner.run(p, compute_mode="cpu", max_steps=400)
    assert not glob.glob(os.path.join(d, "*crashed*")), "run crashed"
    final = glob.glob(os.path.join(d, "*final*.h5"))
    assert final, "no final checkpoint written"
    return p, final[0]


def test_fofc_mhd_cell_face_consistency() -> None:
    p, final = _run(0.03)
    ni, nj = p.resolution[0], p.resolution[1]
    b1c, b2c, _den, _mom, B1, B2 = _read(final)
    ngi = (b1c.shape[1] - ni) // 2
    ngj = (b1c.shape[0] - nj) // 2
    interior = (slice(ngj, b1c.shape[0] - ngj), slice(ngi, b1c.shape[1] - ngi))
    # interp(bface): flat-cartesian 0.5 average of the two bounding faces (no metric weight).
    bx_interp = 0.5 * (B1[:, :-1] + B1[:, 1:])
    by_interp = 0.5 * (B2[:-1, :] + B2[1:, :])
    incons_x = float(np.abs(b1c[interior] - bx_interp).max())
    incons_y = float(np.abs(b2c[interior] - by_interp).max())
    # the field must have actually evolved (else the run is a vacuous no-op that would pass trivially).
    assert float(np.abs(B2 - p.b0).max()) > 1e-6, "By did not evolve; fixture is not exercising CT/FOFC"
    tol = 1e-12
    assert incons_x < tol and incons_y < tol, (
        f"cell B inconsistent with interp(bface): max|bcell-interp|_x={incons_x:.3e}, "
        f"_y={incons_y:.3e} (CT re-sync bound {tol:.0e}) — the FOFC redo left cell B stale"
    )


def test_fofc_mhd_divb_preserved() -> None:
    # the CT redo splices a SINGLE-VALUED edge EMF and curls it, so the discrete div(B) of the
    # staggered face field stays at machine zero through the firing substages. gas D/S conservation
    # of the shared splice is asserted in the hydro FOFC conservation test; the MHD freeze tier
    # leaks a bounded amount of D/S, so conservation is not asserted here.
    p, final = _run(0.03)
    ni, nj = p.resolution[0], p.resolution[1]
    _b1c, _b2c, _den, _mom, B1, B2 = _read(final)
    (x0, x1), (y0, y1) = p.bounds[0], p.bounds[1]
    dx = (x1 - x0) / ni
    dy = (y1 - y0) / nj
    # flat-cartesian cell div(B) = (B1[i+1]-B1[i])/dx + (B2[j+1]-B2[j])/dy over the interior cells.
    div = (B1[:, 1:] - B1[:, :-1]) / dx + (B2[1:, :] - B2[:-1, :]) / dy
    scale = max(float(np.abs(B1).max()) / dx, 1.0)
    max_div = float(np.abs(div).max())
    assert max_div < 1e-10 * scale, f"div(B) broke: {max_div:.3e} (flux scale {scale:.3e})"
