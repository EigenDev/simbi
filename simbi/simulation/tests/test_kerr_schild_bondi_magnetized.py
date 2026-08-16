# =============================================================================
# test_kerr_schild_bondi_magnetized.py
#
# the `_ks` GRMHD gates: magnetized through-horizon bondi accretion on the
# ingoing kerr-schild chart (gr_bondi_ks_magnetized.py). the load-bearing term is
# the shifted riemann fan's induction transpose `+(beta^i/alpha) B^n`: the true
# mag-row flux is (alpha v^n - beta^n) B^i - (alpha v^i - beta^i) B^n, so the
# fan's uniform -(beta^n/alpha) U subtraction alone would advect the radial field
# with the shift and B^r would drift through the accretion transient. the gates:
#
#   identity-class: the staggered B^r is bitwise static through horizon-crossing
#   accretion (measured 0.0 over t = 10 at b_ref = 0.5).
#
#   consistency-class: the radial monopole is force-free (radial B, radial v), so
#   the t = 10 transient must be field-independent at the truncation floor
#   (measured b = 0 vs b = 0.5 at 128: rho 3.7e-5, v1 2.8e-4, pre 8.1e-4
#   relative — the bound fan's dissipation shifts with c_ms, so the transient
#   separates at truncation, unlike the stationary michel hold's 4e-8), and the
#   gas must accrete (inner density rises well above ambient, p > 0, no floors).
#
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

_N = 128
_T_END = 10.0
# measured b=0 vs b=0.5 relative separation at t=10: 3.7e-5 (rho) .. 8.1e-4 (pre);
# ~3x margin on the largest.
_FIELD_INDEP_TOL = 3e-3


def _run(b_ref: float) -> dict:
    from simbi_configs.examples.grmhd.gr_bondi_ks_magnetized import GrBondiKsMagnetized

    d = tempfile.mkdtemp() + "/"
    p = GrBondiKsMagnetized.from_cli(
        ["--resolution", str(_N), "--b-ref", str(b_ref)]
    )
    p.end_time = _T_END
    p.data_directory = d
    p.checkpoint_interval = _T_END
    runner.run(p, compute_mode="cpu", max_steps=4000)
    crashed = glob.glob(os.path.join(d, "*crashed*.h5"))
    assert not crashed, f"ks magnetized bondi crashed at b_ref={b_ref}"
    first = sorted(glob.glob(os.path.join(d, "*chkpt.000_000*.h5")))[0]
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    out = {}
    for tag, fn in (("0", first), ("1", final)):
        with h5py.File(fn) as h:
            g = h["level_0/partition_0/hydro/primitives"]
            halo = (g["rho"].shape[0] - _N) // 2
            sl = slice(halo, halo + _N)
            for k in ("rho", "pre", "v1"):
                out[k + tag] = g[k][sl]
            out["B" + tag] = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
    return out


@needs_backend
def test_monopole_survives_horizon_crossing_bitwise() -> None:
    out = _run(0.5)
    db = float(np.abs(out["B1"] - out["B0"]).max())
    assert db == 0.0, f"staggered B^r moved through the shifted fan: {db:.3e}"
    assert out["pre1"].min() > 0.0, "pressure went non-positive"
    # transonic accretion developed: the near-horizon density rises well above ambient.
    assert out["rho1"][0] > 3.0, f"no through-horizon accretion: rho_in = {out['rho1'][0]:.3f}"


@needs_backend
def test_transient_is_field_independent() -> None:
    weak = _run(0.0)
    strong = _run(0.5)
    for k in ("rho", "v1", "pre"):
        scale = float(np.abs(weak[k + "1"]).max())
        e = float(np.abs(weak[k + "1"] - strong[k + "1"]).max()) / scale
        assert e < _FIELD_INDEP_TOL, f"{k} depends on the field: {e:.3e} relative"
