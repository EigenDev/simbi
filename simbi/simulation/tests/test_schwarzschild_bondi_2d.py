# =============================================================================
# test_schwarzschild_bondi_2d.py
#
# 2D axisymmetric schwarzschild bondi accretion — the end-to-end gate for the
# multi-component Valencia covariant momentum (S_theta) on a curved background. the
# 2D metric-aware c2p (recovering v^r AND v^theta with gamma_{theta theta} = r^2), the
# per-sweep GR flux (radial + angular), and the uniform-lapse densitization of the
# angular momentum law all run for the first time here. the flow is purely radial
# (spherical symmetry -> no angular geodesic source), so every theta column develops
# the SAME transonic inflow: the solution stays theta-INDEPENDENT and the angular
# momentum stays ~0. the test asserts the run completes with NO floor (pressure
# strictly positive), the gas accretes (inner density rises above ambient), and the
# profile is axisymmetric. requires the built cpu_ext backend; skipped otherwise.
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

_END_TIME = 10.0
_NR = 128
_NPOLAR = 12
_RHO_AMBIENT = 1.0


def _bondi_2d_problem(data_dir: str):
    from simbi_configs.examples.grhd.gr_bondi_2d import GrBondi2D

    p = GrBondi2D.from_cli(["--nr", str(_NR), "--npolar", str(_NPOLAR)])
    p.end_time = _END_TIME
    p.data_directory = data_dir
    p.checkpoint_interval = _END_TIME  # initial + final only
    return p


def _read_interior(chkpt_path: str):
    """the interior (ghost-stripped) rho/pre as a (ntheta, nr) array. the primitives are stored as
    the padded 2D grid (axis 0 = theta, axis 1 = r); `owned_fin` is the per-axis INTERIOR size, so the
    symmetric ghost width is (padded - interior)/2."""
    with h5py.File(chkpt_path, "r") as h:
        part = h["level_0/partition_0"]
        n = [int(v) for v in part["owned_fin"][:]]  # [ntheta, nr]
        prims = part["hydro/primitives"]
        rho = np.asarray(prims["rho"][()])
        pre = np.asarray(prims["pre"][()])
    interior = tuple(
        slice((rho.shape[ax] - n[ax]) // 2, (rho.shape[ax] - n[ax]) // 2 + n[ax])
        for ax in range(rho.ndim)
    )
    return rho[interior], pre[interior]


@needs_backend
def test_bondi_2d_completes_positive_and_axisymmetric() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        runner.run(_bondi_2d_problem(d), compute_mode="cpu")

        finals = glob.glob(os.path.join(d, "*.chkpt.final*.h5"))
        assert finals, "2D schwarzschild bondi crashed before completion"

        rho, pre = _read_interior(finals[0])

        # NO FLOOR: pressure strictly positive everywhere.
        assert pre.min() > 0.0, f"pressure went non-positive: min = {pre.min():.3e}"

        # accretion: the innermost radial cell rises above ambient in every theta column.
        assert rho[:, 0].min() > 1.1 * _RHO_AMBIENT, (
            f"inner density did not rise: min over theta = {rho[:, 0].min():.3f}"
        )

        # axisymmetry: the radial profile must be theta-INDEPENDENT (the flow is purely
        # radial; any theta variation is a bug in the angular momentum / flux / c2p).
        col_spread = rho.std(axis=0) / (np.abs(rho.mean(axis=0)) + 1e-30)
        assert col_spread.max() < 1e-6, (
            f"profile not axisymmetric: max relative theta spread = {col_spread.max():.3e}"
        )
