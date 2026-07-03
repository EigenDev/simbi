# =============================================================================
# test_grmhd_uct.py
#
# the GR-UCT constrained-transport gates (design 44 GR-UCT): the upwind
# constrained-transport edge EMF (Del Zanna 2007 / M&DZ 2020 master form) on a
# curved background — the densitized corner EMF `Etilde_phi` built from the
# transport velocity `vtilde = alpha v - beta` and the SHIFTED Banyuls-Font bound
# speeds (materialized per cell, quartic-free), consumed by the same GR curl the
# contact EMF uses. UCT-HLL coefficients (the UCT-HLLD wave-sum is a further step).
#
# the gates are CORRECTNESS, not a quantitative checkerboard win (that needs the
# sharp UCT-HLLD solver — the HLLE gas flux here is already diffusive):
#
#   smooth-limit: the theta-uniform magnetized-michel monopole holds IDENTICALLY
#   under UCT and contact (E_phi = 0 pointwise, so the upwind master form and the
#   contact soft-blend agree) — B static, hydro hold == the 1D gate.
#
#   nontrivial 2D: a poloidal field loop advected through the wedge preserves the
#   w-weighted div(B) to machine precision (1.3e-18) and runs stably (p > 0, |B|
#   bounded, no crash) under the full curved-UCT machinery.
#
# requires the built cpu_ext backend; skipped otherwise.
# =============================================================================
import glob
import math
import os
import tempfile

import h5py
import numpy as np
import pytest

from simbi.simulation import runner
from simbi.types import CtMethod

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)


def _run_michel(ct: CtMethod):
    from simbi_configs.examples.gr_michel_magnetized_2d import GrMichelMagnetized2D

    d = tempfile.mkdtemp() + "/"
    p = GrMichelMagnetized2D.from_cli(
        ["--nr", "128", "--npolar", "16", "--b-ref", "0.5"]
    )
    p.ct_method = ct
    p.end_time = 10.0
    p.data_directory = d
    p.checkpoint_interval = 100.0
    runner.run(p, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*.h5")), f"michel-2d crashed at {ct}"
    first = sorted(glob.glob(os.path.join(d, "*chkpt.000_000*.h5")))[0]
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    return p, first, final


@needs_backend
def test_uct_holds_the_michel_monopole_like_contact() -> None:
    # the smooth field: the upwind UCT master form must give the SAME (zero) EMF as
    # the contact soft-blend, so B stays static and the hydro holds the michel profile.
    p, first, final = _run_michel(CtMethod.UCT)
    with h5py.File(final) as h:
        g = h["level_0/partition_0/hydro/primitives"]
        shp = g["rho"].shape
        halo = [(s - n) // 2 for s, n in zip(shp, (16, 128))]
        sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (16, 128)))
        rho, pre = g["rho"][sl], g["pre"][sl]
        b1_1 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
    with h5py.File(first) as h:
        b1_0 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
    assert pre.min() > 0.0, "pressure went non-positive under UCT"
    assert float(np.abs(b1_1 - b1_0).max()) < 1e-8, "staggered B drifted under UCT"
    sol = p.michel_solution()
    rc = np.array(p.cell_centroids())
    ref = np.array([sol.primitive(r)[0] for r in rc])
    interior = (slice(2, 14), slice(2, 126))
    l1 = float(np.abs(rho[interior] / ref[None, 2:126] - 1.0).mean())
    assert l1 < 3.6e-4, f"UCT michel hold L1 {l1:.3e} (the 1D gate is 1.19e-4)"


def _w_div_max(p, B1, B2) -> tuple[float, float]:
    mm = p.schwarzschild_mass
    nr, npolar = p.nr, p.npolar
    rf = np.array(p.radial_faces())
    tf = np.array(p.theta_faces())
    dr, dth = rf[1] - rf[0], tf[1] - tf[0]
    sg = lambda r, th: r * r * math.sin(th) / math.sqrt(1.0 - 2.0 * mm / r)
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
    return md, sc


@needs_backend
@pytest.mark.parametrize("ct", [CtMethod.CONTACT, CtMethod.UCT])
def test_field_loop_preserves_divergence_and_is_stable(ct) -> None:
    # the nontrivial 2D poloidal loop: the full curved-CT machinery (contact and
    # UCT) must hold the w-weighted div(B) at machine zero and run stably.
    from simbi_configs.examples.gr_field_loop import GrFieldLoop

    d = tempfile.mkdtemp() + "/"
    p = GrFieldLoop.from_cli(["--nr", "128", "--npolar", "64", "--inflow", "0.3"])
    p.ct_method = ct
    p.end_time = 6.0
    p.data_directory = d
    p.checkpoint_interval = 100.0
    runner.run(p, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*.h5")), f"field loop crashed at {ct}"
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
    md, sc = _w_div_max(p, B1, B2)
    assert md < 1e-12 * max(sc, 1.0), f"w-weighted div(B) broke at {ct}: {md:.3e} (scale {sc:.3e})"
