# =============================================================================
# test_cartesian_ks_bh.py
#
# the first GRHD run in a NON-spherical chart — a schwarzschild black
# hole in CARTESIAN kerr-schild coordinates (SchwarzschildKSCartesian). the metric
# gamma_ij = delta_ij + 2M x_i x_j / r^3 and the shift beta^i = 2M x_i/(r^2(r+2M))
# are EXACTLY symmetric under the x <-> y coordinate swap, so a symmetric initial
# state on a square patch with symmetric boundaries must evolve symmetrically under
# TRANSPOSE to roundoff. this is the oracle-free correctness gate for the whole
# chart-generic GR chain: the metric-aware Valencia flux with the shift on EVERY
# sweep (not just radial), the metric-aware c2p, the covariant geodesic source, and
# the state-independent light-cone CFL. any coordinate-role bug — an axis treated as
# radial, a shift applied on one axis only — breaks the transpose symmetry exactly.
# also a stability check: horizon-penetrating, no floors, pressure stays positive.
# requires the built cpu_ext backend; skipped otherwise.
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np
import pytest

from simbi.simulation import runner

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)


def _run(res: int, data_dir: str):
    from simbi_configs.examples.grhd.gr_cartesian_ks_bh import GrCartesianKsBH

    p = GrCartesianKsBH.from_cli([])
    p.resolution = (res, res)
    p.end_time = 2.0
    p.data_directory = data_dir
    p.checkpoint_interval = 2.0  # initial + final only
    runner.run(p, compute_mode="cpu")

    finals = glob.glob(os.path.join(data_dir, "*.chkpt.final*.h5"))
    assert finals, f"cartesian KS BH run crashed at {res} zones"
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        # strip the equal halos on both axes -> the central res x res interior.
        halo = (prims["rho"].shape[0] - res) // 2
        sl = slice(halo, halo + res)
        return (
            prims["rho"][sl, sl],
            prims["pre"][sl, sl],
            prims["v1"][sl, sl],
            prims["v2"][sl, sl],
        )


@needs_backend
def test_cartesian_ks_bh_runs_stably() -> None:
    # GRHD in a NON-spherical chart runs end-to-end, horizon-penetrating, holding a
    # positive finite state with no floors — the metric-aware Valencia flux with the shift on every
    # axis, the covariant geodesic source, the metric-aware c2p, and the light-cone CFL all compose.
    with tempfile.TemporaryDirectory() as d:
        rho, pre, _v1, _v2 = _run(64, d + "/")
    assert np.isfinite(rho).all() and np.isfinite(pre).all(), "NaN/inf in the evolved state"
    assert pre.min() > 0.0, f"pressure went non-positive: {pre.min():.3e}"
    assert rho.min() > 0.0, f"density went non-positive: {rho.min():.3e}"


@needs_backend
def test_cartesian_ks_bh_preserves_x_y_symmetry() -> None:
    # the cartesian KS metric is EXACTLY symmetric under the x <-> y coordinate + index swap, so a
    # symmetric initial state on a square patch must evolve symmetrically under transpose to roundoff:
    # rho / pre transpose-symmetric, v_x at (i, j) == v_y at (j, i). the tolerance is roundoff
    # accumulated over the run and reflects no physics — any coordinate-role bug (an axis treated as radial, as
    # the densitization lapse once did with r = x-coordinate) breaks it far above this.
    with tempfile.TemporaryDirectory() as d:
        rho, pre, v1, v2 = _run(64, d + "/")

    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).max() / (np.abs(a).max() + 1e-300))

    assert rel(rho, rho.T) < 1e-10, f"rho not x<->y symmetric: {rel(rho, rho.T):.3e}"
    assert rel(pre, pre.T) < 1e-10, f"pre not x<->y symmetric: {rel(pre, pre.T):.3e}"
    assert rel(v1, v2.T) < 1e-10, f"v_x != v_y^T: {rel(v1, v2.T):.3e}"
