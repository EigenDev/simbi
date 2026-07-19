# =============================================================================
# test_cartesian_grmhd.py
#
# the cartesian kerr-schild GRMHD constrained-transport gate: a
# poloidal field loop on the non-spherical (x, y) chart, seeded div-free through the
# metric-weighted curl of A_z. the chart-generic densitized curl + the two-component-
# shift corner EMF must PRESERVE the w-weighted div(B) = sum sqrt(gamma)(face) x
# coordinate-length x B_face to machine precision as the gas free-falls, and run
# stably (p > 0, |B| bounded, finite — no floors, no crash). the gas flux is the
# fast-magnetosonic HLLE fan (the diagonal-metric HLLD wrapper does not apply to the
# non-diagonal cartesian metric). the same gate runs at BOTH the contact and the
# GR-UCT-HLL edge EMF. requires the built cpu_ext backend; skipped otherwise.
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

_NX, _NY = 96, 96


def _w_div_max(p, B1, B2) -> tuple[float, float]:
    # the coordinate divergence of the densitized face field Btilde = sqrt(gamma) B:
    # sum over the four cell faces of sqrt(gamma)(face) x transverse-length x B_face.
    # machine-zero for the CT-preserved field. sqrt(gamma) = sqrt(1 + 2M/r), r = |x|.
    mm = p.schwarzschild_mass
    nx, ny = p.nx, p.ny
    xf = np.array(p.x_faces())
    yf = np.array(p.y_faces())
    dx, dy = xf[1] - xf[0], yf[1] - yf[0]
    sg = lambda x, y: math.sqrt(1.0 + 2.0 * mm / math.hypot(x, y))
    xc = 0.5 * (xf[:-1] + xf[1:])
    yc = 0.5 * (yf[:-1] + yf[1:])
    md, sc = 0.0, 0.0
    for j in range(ny):
        for i in range(nx):
            div = (
                sg(xf[i + 1], yc[j]) * dy * B1[j, i + 1]
                - sg(xf[i], yc[j]) * dy * B1[j, i]
                + sg(xc[i], yf[j + 1]) * dx * B2[j + 1, i]
                - sg(xc[i], yf[j]) * dx * B2[j, i]
            )
            md = max(md, abs(div))
            sc = max(sc, abs(sg(xf[i + 1], yc[j]) * dy * B1[j, i + 1]))
    return md, sc


@needs_backend
@pytest.mark.parametrize("ct", [CtMethod.CONTACT, CtMethod.UCT])
def test_cartesian_field_loop_preserves_divergence_and_is_stable(ct) -> None:
    from simbi_configs.examples.grmhd.gr_cartesian_field_loop import GrCartesianFieldLoop

    d = tempfile.mkdtemp() + "/"
    p = GrCartesianFieldLoop.from_cli(["--nx", str(_NX), "--ny", str(_NY)])
    p.ct_method = ct
    p.end_time = 3.0
    p.data_directory = d
    p.checkpoint_interval = 100.0
    runner.run(p, compute_mode="cpu")

    assert not glob.glob(os.path.join(d, "*crashed*.h5")), f"cartesian field loop crashed at {ct}"
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    with h5py.File(final) as h:
        B1 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
        B2 = h["level_0/partition_0/hydro/magnetic/B2/data"][:]
        g = h["level_0/partition_0/hydro/primitives"]
        shp = g["rho"].shape
        halo = [(s - n) // 2 for s, n in zip(shp, (_NY, _NX))]
        sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (_NY, _NX)))
        pre, rho = g["pre"][sl], g["rho"][sl]

    # stability: positive, finite, bounded (no floors).
    assert np.isfinite(rho).all() and np.isfinite(pre).all(), f"NaN/inf at {ct}"
    assert pre.min() > 0.0, f"pressure went non-positive at {ct}: {pre.min():.3e}"
    assert rho.min() > 0.0, f"density went non-positive at {ct}: {rho.min():.3e}"
    assert float(np.abs(B1).max()) < 1.0, f"field blew up at {ct}"

    # the constrained-transport correctness gate: the w-weighted div(B) stays at roundoff.
    md, sc = _w_div_max(p, B1, B2)
    assert md < 1e-12 * max(sc, 1.0), f"w-weighted div(B) broke at {ct}: {md:.3e} (scale {sc:.3e})"


@needs_backend
@pytest.mark.parametrize("ct", [CtMethod.CONTACT, CtMethod.UCT])
def test_cartesian_hlld_gas_flux_runs_and_preserves_divergence(ct) -> None:
    # the tetrad-frame MUB09 HLLD gas flux on the NON-DIAGONAL cartesian kerr-schild metric:
    # orthonormal_basis(dir) Gram-Schmidts gamma_ij = delta + 2H x_i x_j / r^2 into the local flat
    # frame where the validated flat solver runs, and the flux maps back with the normal factor E_dd.
    # UCT additionally exercises the sharp UCT-HLLD wave-sum edge EMF (the tetrad states fan + the
    # multi-axis moving-interface shift beta^x, beta^y). the tetrad's exactness is pinned by the rust
    # unit gates; this is the pipeline gate that it bakes, dispatches, runs stably, and holds div(B) = 0.
    from simbi_configs.examples.grmhd.gr_cartesian_field_loop import GrCartesianFieldLoop

    d = tempfile.mkdtemp() + "/"
    p = GrCartesianFieldLoop.from_cli(["--nx", str(_NX), "--ny", str(_NY)])
    p.solver = Solver.HLLD
    p.ct_method = ct
    p.end_time = 3.0
    p.data_directory = d
    p.checkpoint_interval = 100.0
    runner.run(p, compute_mode="cpu")

    assert not glob.glob(os.path.join(d, "*crashed*.h5")), "cartesian HLLD run crashed"
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    with h5py.File(final) as h:
        B1 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
        B2 = h["level_0/partition_0/hydro/magnetic/B2/data"][:]
        g = h["level_0/partition_0/hydro/primitives"]
        shp = g["rho"].shape
        halo = [(s - n) // 2 for s, n in zip(shp, (_NY, _NX))]
        sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (_NY, _NX)))
        pre, rho = g["pre"][sl], g["rho"][sl]

    assert np.isfinite(rho).all() and np.isfinite(pre).all(), "NaN/inf under HLLD"
    assert pre.min() > 0.0, f"pressure went non-positive under HLLD: {pre.min():.3e}"
    assert rho.min() > 0.0, f"density went non-positive under HLLD: {rho.min():.3e}"
    assert float(np.abs(B1).max()) < 1.0, "field blew up under HLLD"
    md, sc = _w_div_max(p, B1, B2)
    assert md < 1e-12 * max(sc, 1.0), f"HLLD w-weighted div(B) broke: {md:.3e} (scale {sc:.3e})"
