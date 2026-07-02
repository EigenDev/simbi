# =============================================================================
# test_kerr_schild_bondi_horizon.py
#
# the ingoing-Kerr-Schild GRHD payoff: bondi accretion in a HORIZON-PENETRATING
# chart, with the inner boundary placed BELOW the horizon (r = 1.5 < 2M = 2). the
# schwarzschild-coordinate run (test_schwarzschild_bondi_transient) must hold its
# boundary OUTSIDE 2M and even so was fragile; the KS chart is regular across the
# horizon, so a uniform-at-rest gas develops transonic accretion and CROSSES r = 2M
# smoothly — no crash, no floor, density rising inward through the horizon. exercises
# the shift-advection flux kernel + KS densitization + KS sources end-to-end.
# requires the built cpu_ext backend; skipped otherwise.
# =============================================================================
import glob
import os
import tempfile

import h5py
import pytest

from simbi.simulation import runner

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

_END_TIME = 15.0
_RESOLUTION = 128
_HORIZON = 2.0  # 2M, M = 1


def _bondi_ks_problem(data_dir: str):
    from simbi_configs.examples.gr_bondi_ks import GrBondiKS

    p = GrBondiKS.from_cli(["--resolution", str(_RESOLUTION)])
    p.end_time = _END_TIME
    p.data_directory = data_dir
    p.checkpoint_interval = _END_TIME
    return p


def _read_interior(chkpt_path: str):
    # the stored arrays carry the ghost cells; the interior is the central
    # `_RESOLUTION` entries (owned_start/owned_fin are interior-relative and do
    # NOT index the ghost-padded arrays).
    with h5py.File(chkpt_path, "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - _RESOLUTION) // 2
        sl = slice(halo, halo + _RESOLUTION)
        return prims["rho"][sl], prims["pre"][sl]


@needs_backend
def test_ks_bondi_crosses_horizon_stable_and_positive() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        prob = _bondi_ks_problem(d)
        # the inner boundary must sit inside the horizon — that is the whole point.
        assert prob.bounds[0][0] < _HORIZON

        runner.run(prob, compute_mode="cpu")

        # a clean completion writes <res>.chkpt.final*.h5; a crash writes only crashed.h5.
        finals = glob.glob(os.path.join(d, "*.chkpt.final*.h5"))
        assert finals, "KS bondi crashed before completion (horizon-crossing regression)"

        rho, pre = _read_interior(finals[0])

        # NO FLOOR: pressure must stay strictly positive through and inside the horizon.
        assert pre.min() > 0.0, f"pressure went non-positive: min = {pre.min():.3e}"

        # accretion: the gas compresses toward the hole — inner density well above ambient (1.0).
        # rho[0] is the innermost interior cell (inside the horizon).
        assert rho[0] > 3.0, f"no inward compression: rho_inner = {rho[0]:.3f}"
        # and it rises monotonically inward across the whole profile (accretion, not depletion).
        assert rho[0] >= rho[-1], f"density does not rise inward: {rho[0]:.3f} vs {rho[-1]:.3f}"
