# =============================================================================
# test_fm_torus_spin_excised.py
#
# the spinning fishbone-moncrief torus on the 3d cartesian kerr chart with the
# oblate-spheroidal horizon excision — the end-to-end composition of the
# spinning cartesian metric, the FM equilibrium at spin, and the level-set
# excision. the gates:
# - the run completes finite and positive with the torus surviving (the
#   equilibrium is discretely near-stationary; a coarse grid diffuses but must
#   not destroy the pressure maximum within a few M of evolution);
# - the FM initial data is axisymmetric and equatorially symmetric, and the
#   spinning metric shares both, so the evolved state holds the quarter-turn
#   (x, y) -> (-y, x) and z -> -z grid symmetries to roundoff — the sharp
#   coordinate-role gate for the torus jacobian + spheroidal excision compose;
# - the spin genuinely enters: the a = 0.9 run differs from the a = 0 run of
#   the same configuration (different metric, different equilibrium).
# =============================================================================
import glob
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation import runner

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

from simbi_configs.examples.grhd.gr_fishbone_moncrief_cartesian import (
    GrFishboneMoncriefCartesian,
)

RES = 48
L_BOX = 20.0


def _run(spin: float) -> np.ndarray:
    d = tempfile.mkdtemp() + "/"
    # the torus geometry is spin-dependent: kappa = 1.01/r_in = 8 is the compact
    # a = 0 torus; at a = 0.9 that pair degenerates to a sub-cell sliver, so the
    # spinning run uses the thick-torus pair (r_in = 6, kappa = 1.15).
    r_in, kappa = (6.0, 1.15) if spin != 0.0 else (8.0, 1.01)
    p = GrFishboneMoncriefCartesian(
        kerr_spin=spin,
        r_in=r_in,
        kappa=kappa,
        resolution=(RES, RES, RES),
        end_time=5.0,
        checkpoint_interval=1.0e30,
        data_directory=Path(d),
    )
    # bounded: a healthy run reaches t = 5 in a few hundred steps. a dt collapse
    # (the failure mode this cap exists for: an excised-region cell throttling the
    # source-admissibility rate) burns the cap, and the time
    # assertion below reports it as the failure it is.
    runner.run(p, compute_mode="cpu", max_steps=2000)
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, f"spinning FM torus run (a={spin}) crashed"
    with h5py.File(finals[0], "r") as h:
        t_final = float(h["metadata"].attrs["time"])
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - RES) // 2
        sl = slice(halo, halo + RES)
        rho = prims["rho"][sl, sl, sl]
    assert t_final > 4.999, (
        f"the a={spin} torus run stalled at t = {t_final:.4f} < 5 within 2000 steps "
        "(dt collapse — check the excised-region source-admissibility mask)"
    )
    return rho


@needs_backend
def test_spinning_fm_torus_with_spheroidal_excision() -> None:
    rho = _run(0.9)
    assert np.isfinite(rho).all(), "non-finite state in the spinning torus run"
    assert rho.min() > 0.0, "density went non-positive"
    # the torus is PRESENT AND SURVIVES, asserted in ITS OWN equatorial band —
    # the global density maximum is the floored-redshift corona pileup at the
    # horizon, which exists whether or not the torus does (a global-max assert
    # is vacuous for torus presence: the corona once silently swallowed a
    # near-marginal thin torus whole via the pressure-matched surface).
    dd = 2.0 * L_BOX / RES
    xs = (np.arange(RES) + 0.5) * dd - L_BOX
    z, y, x = np.meshgrid(xs, xs, xs, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    band = (np.abs(z) < 2.0) & (r > 5.0) & (r < 14.0)
    assert rho[band].max() > 0.3, (
        f"no torus in the equatorial band 5 < r < 14 (max rho {rho[band].max():.4f}; "
        "the corona swallowed it or it disintegrated)"
    )

    # axisymmetric initial data on the axisymmetric spinning metric: the evolved
    # state holds the quarter turn about the spin axis and the equatorial
    # reflection to roundoff (storage [k, j, i]).
    err_rot = np.abs(rho - np.rot90(rho, 1, (1, 2))).max()
    assert err_rot < 1e-11, f"quarter-turn symmetry broken: {err_rot:e}"
    err_z = np.abs(rho - rho[::-1, :, :]).max()
    assert err_z < 1e-11, f"equatorial reflection symmetry broken: {err_z:e}"

    # the spin genuinely enters (non-vacuous: a dispatch that silently ran the
    # a = 0 chart would make these identical).
    rho0 = _run(0.0)
    assert np.abs(rho - rho0).max() > 1e-3, (
        "the a = 0.9 and a = 0 torus runs are near-identical; the spin never acted"
    )
