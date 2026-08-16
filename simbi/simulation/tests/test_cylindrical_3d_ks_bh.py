# =============================================================================
# test_cylindrical_3d_ks_bh.py
#
# GRHD in the full 3D cylindrical kerr-schild chart — (R, phi, z) all
# gridded. the metric is axisymmetric (phi-independent), so a phi-uniform initial state
# must stay phi-uniform to roundoff even with the azimuth fully resolved — the correctness
# gate that the 3D path handles the gridded phi axis right (the radial + vertical infall
# develops along R and z). also a stability check: horizon-penetrating, no floors.
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

_RES = (
    20,
    12,
    16,
)  # (nR, nphi, nz) — distinct so the phi axis (nphi=12) is identifiable


def _run(data_dir: str):
    from simbi.simulation.tests.fixtures.gr_cylindrical_3d_ks_bh import (
        GrCylindrical3DKsBH,
    )

    p = GrCylindrical3DKsBH.from_cli([])
    p.resolution = _RES
    p.end_time = 0.5
    p.data_directory = data_dir
    p.checkpoint_interval = 0.5
    runner.run(p, compute_mode="cpu", max_steps=400)

    finals = glob.glob(os.path.join(data_dir, "*.chkpt.final*.h5"))
    assert finals, "3D cylindrical KS BH run crashed"
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]

        def strip(name: str) -> np.ndarray:
            a = prims[name][...]
            # uniform per-axis halo h: sum(shape) = sum(interior) + 2*ndim*h.
            h_ = (sum(a.shape) - sum(_RES)) // (2 * a.ndim)
            return a[tuple(slice(h_, s - h_) for s in a.shape)]

        return strip("rho"), strip("pre")


@needs_backend
def test_cylindrical_3d_ks_bh_runs_stably_and_is_axisymmetric() -> None:
    with tempfile.TemporaryDirectory() as d:
        rho, pre = _run(d + "/")

    # stability: horizon-penetrating, positive, finite (no floors).
    assert np.isfinite(rho).all() and np.isfinite(pre).all(), (
        "NaN/inf in the evolved state"
    )
    assert pre.min() > 0.0, f"pressure went non-positive: {pre.min():.3e}"
    assert rho.min() > 0.0, f"density went non-positive: {rho.min():.3e}"

    # per-axis variation of rho: the deviation from the axis-mean, normalized. the phi axis is
    # uniform (roundoff); R and z carry the developing infall.
    def axis_var(a: np.ndarray, ax: int) -> float:
        return float(
            np.abs(a - a.mean(axis=ax, keepdims=True)).max()
            / (np.abs(a).max() + 1e-300)
        )

    variations = [axis_var(rho, ax) for ax in range(3)]
    order = np.argsort(variations)
    phi_ax = int(order[0])  # the least-varying axis is the (uniform) azimuth

    assert variations[phi_ax] < 1e-10, f"no phi-axisymmetric axis: {variations}"
    assert rho.shape[phi_ax] == _RES[1], (
        f"the phi-uniform axis (size {rho.shape[phi_ax]}) is not the nphi={_RES[1]} axis: {rho.shape}"
    )
    # the other two axes (R, z) carry real structure — the flow actually developed.
    assert variations[int(order[1])] > 1e-6, (
        f"only one axis varies (flow static?): {variations}"
    )
