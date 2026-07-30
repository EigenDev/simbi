# =============================================================================
# test_cylindrical_ks_bh.py
#
# GRHD in the CYLINDRICAL kerr-schild chart — a schwarzschild BH on
# a 2.5D axisymmetric (R, z) grid with the azimuthal v_phi DOF. the metric is EXACTLY
# symmetric under z -> -z (r = sqrt(R^2 + z^2) is even in z; gamma_Rz and beta^z flip
# sign with the z-momentum), so a z-symmetric initial state on a grid symmetric about
# z = 0 must evolve z-REFLECTION symmetrically to roundoff: rho / pre EVEN in z, v_z ODD,
# v_R EVEN. this is the oracle-free correctness gate for the cylindrical chart (the analog
# of the cartesian x <-> y test), catching any coordinate-role or one-axis-shift bug — in
# particular the densitization-lapse "R as the radius" trap (the lapse must use the
# SPHERICAL r = sqrt(R^2 + z^2)). also a stability check: horizon-penetrating, no floors.
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

_NR, _NZ = 40, 30  # distinct so the z-axis (nz) is identifiable by shape


def _run(data_dir: str):
    from simbi_configs.examples.grhd.gr_cylindrical_ks_bh import GrCylindricalKsBH

    p = GrCylindricalKsBH.from_cli([])
    p.resolution = (_NR, _NZ)
    p.end_time = 2.0
    p.data_directory = data_dir
    p.checkpoint_interval = 2.0
    runner.run(p, compute_mode="cpu", max_steps=400)

    finals = glob.glob(os.path.join(data_dir, "*.chkpt.final*.h5"))
    assert finals, "cylindrical KS BH run crashed"
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]

        def strip(name: str) -> np.ndarray:
            a = prims[name][...]
            # the checkpoint stores 2D as [axis1][axis0] = [z][R]; per-axis halo strip.
            hz = (a.shape[0] - _NZ) // 2  # array axis 0 = z
            hr = (a.shape[1] - _NR) // 2  # array axis 1 = R
            return a[hz : hz + _NZ, hr : hr + _NR]

        # v1 = v_R, v2 = v_phi, v3 = v_z (cylindrical momentum order).
        return strip("rho"), strip("pre"), strip("v1"), strip("v3")


@needs_backend
def test_cylindrical_ks_bh_runs_stably_and_is_z_reflection_symmetric() -> None:
    with tempfile.TemporaryDirectory() as d:
        rho, pre, v_r, v_z = _run(d + "/")

    # the interior is (nz, nR); z is array axis 0 -> reflect with [::-1, :].
    assert rho.shape == (_NZ, _NR), f"unexpected interior shape {rho.shape}"

    # stability: horizon-penetrating, positive, finite (no floors).
    assert np.isfinite(rho).all() and np.isfinite(pre).all(), "NaN/inf in the evolved state"
    assert pre.min() > 0.0, f"pressure went non-positive: {pre.min():.3e}"
    assert rho.min() > 0.0, f"density went non-positive: {rho.min():.3e}"

    # z -> -z reflection: rho / pre EVEN, v_z ODD, v_R EVEN — to roundoff over the run.
    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).max() / (np.abs(a).max() + 1e-300))

    assert rel(rho, rho[::-1, :]) < 1e-10, f"rho not z-even: {rel(rho, rho[::-1, :]):.3e}"
    assert rel(pre, pre[::-1, :]) < 1e-10, f"pre not z-even: {rel(pre, pre[::-1, :]):.3e}"
    assert rel(v_r, v_r[::-1, :]) < 1e-10, f"v_R not z-even: {rel(v_r, v_r[::-1, :]):.3e}"
    assert rel(v_z, -v_z[::-1, :]) < 1e-10, f"v_z not z-odd: {rel(v_z, -v_z[::-1, :]):.3e}"
