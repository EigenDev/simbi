# =============================================================================
# test_disk_ks_bh.py
#
# GRHD in the equatorial (R, phi) DISK chart — a schwarzschild BH in the
# diagonal cylindrical kerr-schild metric gamma = diag(1 + 2M/R, R^2) (the z = 0 razor-
# thin accretion-disk plane). the metric is AXISYMMETRIC (phi-independent), so a
# phi-uniform initial state must stay phi-uniform to roundoff — the oracle-free
# correctness gate for the disk chart, catching any azimuthal coordinate-role bug. also a
# stability check: horizon-penetrating, no floors. requires the built cpu_ext backend.
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

_NR, _NPHI = 48, 24  # distinct so the phi-axis (nphi) is identifiable by shape


def _run(data_dir: str):
    from simbi.simulation.tests.fixtures.gr_disk_ks_bh import GrDiskKsBH

    p = GrDiskKsBH.from_cli([])
    p.resolution = (_NR, _NPHI)
    p.end_time = 2.0
    p.data_directory = data_dir
    p.checkpoint_interval = 2.0
    runner.run(p, compute_mode="cpu")

    finals = glob.glob(os.path.join(data_dir, "*.chkpt.final*.h5"))
    assert finals, "disk KS BH run crashed"
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]

        def strip(name: str) -> np.ndarray:
            a = prims[name][...]
            # locate the phi-axis (interior size nphi) vs the R-axis (nR).
            phi_ax = 0 if abs(a.shape[0] - _NPHI) < abs(a.shape[0] - _NR) else 1
            r_ax = 1 - phi_ax
            hp = (a.shape[phi_ax] - _NPHI) // 2
            hr = (a.shape[r_ax] - _NR) // 2
            sl = [slice(None), slice(None)]
            sl[phi_ax] = slice(hp, hp + _NPHI)
            sl[r_ax] = slice(hr, hr + _NR)
            # return with phi as axis 0 for the uniformity reduction.
            out = a[tuple(sl)]
            return out if phi_ax == 0 else out.T

        return strip("rho"), strip("pre"), strip("v1")  # v1 = v_R


@needs_backend
def test_disk_ks_bh_runs_stably_and_is_axisymmetric() -> None:
    with tempfile.TemporaryDirectory() as d:
        rho, pre, v_r = _run(d + "/")

    # stability: horizon-penetrating, positive, finite (no floors).
    assert np.isfinite(rho).all() and np.isfinite(pre).all(), (
        "NaN/inf in the evolved state"
    )
    assert pre.min() > 0.0, f"pressure went non-positive: {pre.min():.3e}"
    assert rho.min() > 0.0, f"density went non-positive: {rho.min():.3e}"

    # axisymmetry: the metric never reads phi, so a phi-uniform state stays phi-uniform (each
    # R-ring constant along phi = axis 0). the radial infall varies with R (axis 1).
    def phi_var(a: np.ndarray) -> float:
        return float(
            np.abs(a - a.mean(axis=0, keepdims=True)).max() / (np.abs(a).max() + 1e-300)
        )

    assert phi_var(rho) < 1e-10, f"rho not phi-uniform: {phi_var(rho):.3e}"
    assert phi_var(pre) < 1e-10, f"pre not phi-uniform: {phi_var(pre):.3e}"
    assert phi_var(v_r) < 1e-10, f"v_R not phi-uniform: {phi_var(v_r):.3e}"
    # a sanity check that the flow actually developed (radial infall broke the R-uniformity).
    assert float(np.abs(rho - rho.mean(axis=1, keepdims=True)).max()) > 1e-6, (
        "no radial evolution"
    )
