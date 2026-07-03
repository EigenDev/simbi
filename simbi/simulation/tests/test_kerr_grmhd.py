# =============================================================================
# test_kerr_grmhd.py
#
# the spinning-KERR GRMHD gate (design 44 phase C): the full kerr RMHD kernel
# path — the tetrad HLLD on the NON-DIAGONAL gamma_{r phi}, the moving-interface
# radial shift, the EM-stress covariant source with the azimuthal (swirl)
# momentum, the metric-aware c2p, and the kerr-wired constrained transport.
#
# the gate is the w-weighted div(B) machine-zero + stability on an advected
# poloidal loop at spin 0.9. this ALSO guards the python->rust dispatch: a Kerr
# MHD config that silently falls through to the flat path (the bug this test was
# written to catch) runs Minkowski kernels, and the KERR sqrt(gamma)-weighted
# div then drifts to O(1) — so a passing machine-zero here certifies the real
# kerr kernels actually ran. requires the built cpu_ext backend.
# =============================================================================
import glob
import math
import os
import tempfile

import h5py
import numpy as np
import pytest

from simbi.simulation import runner
from simbi.types import CtMethod, Solver

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)


def _kerr_wdiv(p, B1, B2):
    mm, a = p.schwarzschild_mass, p.kerr_spin
    nr, npolar = p.nr, p.npolar
    rf = np.array(p.radial_faces())
    tf = np.array(p.theta_faces())
    dr, dth = rf[1] - rf[0], tf[1] - tf[0]

    def sg(r, th):
        s = r * r + a * a * math.cos(th) ** 2
        return s * math.sin(th) * math.sqrt(1.0 + 2.0 * mm * r / s)

    rc = 0.5 * (rf[:-1] + rf[1:])
    tc = 0.5 * (tf[:-1] + tf[1:])
    md, sc = 0.0, 0.0
    for j in range(npolar):
        for i in range(nr):
            div = (
                sg(rf[i + 1], tc[j]) * dth * B1[j, i + 1]
                - sg(rf[i], tc[j]) * dth * B1[j, i]
                + sg(rc[i], tf[j + 1]) * dr * B2[j + 1, i]
                - sg(rc[i], tf[j]) * dr * B2[j, i]
            )
            md = max(md, abs(div))
            sc = max(sc, abs(sg(rf[i + 1], tc[j]) * dth * B1[j, i + 1]))
    return md / max(sc, 1e-30)


@needs_backend
@pytest.mark.parametrize("ct", [CtMethod.CONTACT, CtMethod.UCT])
def test_kerr_field_loop_divergence_free_and_stable(ct) -> None:
    from simbi_configs.examples.gr_kerr_field_loop import GrKerrFieldLoop

    d = tempfile.mkdtemp() + "/"
    p = GrKerrFieldLoop.from_cli(["--nr", "128", "--npolar", "64", "--kerr-spin", "0.9"])
    p.ct_method = ct
    p.solver = Solver.HLLE
    p.end_time = 4.0
    p.data_directory = d
    p.checkpoint_interval = 100.0
    runner.run(p, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*.h5")), f"kerr loop crashed at {ct}"
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    with h5py.File(final) as h:
        B1 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
        B2 = h["level_0/partition_0/hydro/magnetic/B2/data"][:]
        g = h["level_0/partition_0/hydro/primitives"]
        shp = g["rho"].shape
        halo = [(s - n) // 2 for s, n in zip(shp, (64, 128))]
        sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (64, 128)))
        pre, rho = g["pre"][sl], g["rho"][sl]
    assert pre.min() > 0.0, f"pressure went non-positive at {ct}"
    assert not np.isnan(rho).any(), f"NaN at {ct}"
    assert float(np.abs(B1).max()) < 1.0, "field blew up"
    ratio = _kerr_wdiv(p, B1, B2)
    assert ratio < 1e-10, f"kerr w-weighted div(B) not machine-zero at {ct}: {ratio:.3e}"
