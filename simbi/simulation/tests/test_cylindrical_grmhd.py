# =============================================================================
# test_cylindrical_grmhd.py
#
# the cylindrical kerr-schild GRMHD constrained-transport gates (design 45 GRMHD):
# a poloidal field loop on the 2.5D (R, z) plane and an in-plane loop on the (R, phi)
# equatorial DISK, each seeded div-free through the metric-weighted curl of a vector
# potential. the chart-generic densitized curl + the two-component-shift corner EMF
# must PRESERVE the w-weighted div(B) = sum sqrt(gamma)(face) x coordinate-length x
# B_face to machine precision as the gas free-falls, and run stably (p > 0, |B|
# bounded, finite — no floors). sqrt(det gamma) = R sqrt(1 + 2M/r), r = sqrt(R^2 + z^2)
# (the disk is the r = R equatorial slice). the gas flux is the HLLE fan (the diagonal-
# metric HLLD wrapper is a follow-on). both charts run at contact and UCT-HLL CT.
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
from simbi.types import CtMethod, Solver

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)


def _w_div_max(sg, rf, tf, B1, B2) -> tuple[float, float]:
    # coordinate divergence of the densitized face field Btilde = sqrt(gamma) B, over the
    # four cell faces: sum sqrt(gamma)(face) x transverse-length x B_face. rf are the R-faces
    # (the B1/B_R face axis), tf the transverse faces (z or phi). machine-zero for the
    # CT-preserved field. sg(r, t) = sqrt(det gamma) (t-independent for the disk).
    dr, dt = rf[1] - rf[0], tf[1] - tf[0]
    rc = 0.5 * (rf[:-1] + rf[1:])
    tc = 0.5 * (tf[:-1] + tf[1:])
    ni, nj = len(rf) - 1, len(tf) - 1
    md, sc = 0.0, 0.0
    for j in range(nj):
        for i in range(ni):
            div = (
                sg(rf[i + 1], tc[j]) * dt * B1[j, i + 1]
                - sg(rf[i], tc[j]) * dt * B1[j, i]
                + sg(rc[i], tf[j + 1]) * dr * B2[j + 1, i]
                - sg(rc[i], tf[j]) * dr * B2[j, i]
            )
            md = max(md, abs(div))
            sc = max(sc, abs(sg(rf[i + 1], tc[j]) * dt * B1[j, i + 1]))
    return md, sc


def _run(p, d, ct, end_time=2.5, solver=None):
    p.ct_method = ct
    if solver is not None:
        p.solver = solver
    p.end_time = end_time
    p.data_directory = d
    p.checkpoint_interval = 100.0
    runner.run(p, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*.h5")), f"run crashed at {ct}"
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    with h5py.File(final) as h:
        B1 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
        B2 = h["level_0/partition_0/hydro/magnetic/B2/data"][:]
        g = h["level_0/partition_0/hydro/primitives"]
        rho, pre = g["rho"][...], g["pre"][...]
    assert np.isfinite(rho).all() and np.isfinite(pre).all(), f"NaN/inf at {ct}"
    assert pre.min() > 0.0, f"pressure went non-positive at {ct}: {pre.min():.3e}"
    assert rho.min() > 0.0, f"density went non-positive at {ct}: {rho.min():.3e}"
    assert float(np.abs(B1).max()) < 1.0, f"field blew up at {ct}"
    return B1, B2


# the (R, z) 2.5D poloidal chart: the CT (contact + UCT-HLL) addresses the true in-plane axes via
# gr_ct_plane, so the GAPPED grid-axis set [0, 2] reconstructs the staggered field along the axis
# whose transverse halo it carries. t = 0.7 keeps the loop clean at BOTH CT methods — the poloidal
# free-fall converges and piles B up at the inner R boundary, and the sharp UCT does not diffuse it
# (a physical/BC effect the disk avoids, infalling only radially at fixed z). div(B) stays machine-zero.
@needs_backend
@pytest.mark.parametrize("ct", [CtMethod.CONTACT, CtMethod.UCT])
def test_cylindrical_rz_field_loop_preserves_divergence(ct) -> None:
    from simbi_configs.examples.grmhd.gr_cylindrical_rz_field_loop import (
        GrCylindricalRzFieldLoop,
    )

    d = tempfile.mkdtemp() + "/"
    p = GrCylindricalRzFieldLoop.from_cli(["--nr", "80", "--nz", "80"])
    B1, B2 = _run(p, d, ct, end_time=0.7)

    mm = p.schwarzschild_mass
    sg = lambda r, z: r * math.sqrt(1.0 + 2.0 * mm / math.hypot(r, z))
    rf = np.array(p.radial_faces())
    zf = np.array(p.z_faces())
    md, sc = _w_div_max(sg, rf, zf, B1, B2)
    assert md < 1e-12 * max(sc, 1.0), f"(R,z) w-div(B) broke at {ct}: {md:.3e} (scale {sc:.3e})"


# the sharp UCT-HLLD wave-sum edge EMF on the GAPPED (R, z) grid axes [0, 2]: the tetrad HLLD fan
# reads a full 3-vector prim + the world (R, z) metric, so the kernel assembles the prim in WORLD
# order (v[pc]=..) and solves along the world normal (dir = pc); gr_ct_plane maps the gapped axes.
# the two-component shift (beta^R, beta^z) rides both fans. div(B) machine-zero + stable at t = 0.7.
@needs_backend
def test_cylindrical_rz_uct_hlld_preserves_divergence() -> None:
    from simbi_configs.examples.grmhd.gr_cylindrical_rz_field_loop import (
        GrCylindricalRzFieldLoop,
    )

    d = tempfile.mkdtemp() + "/"
    p = GrCylindricalRzFieldLoop.from_cli(["--nr", "80", "--nz", "80"])
    B1, B2 = _run(p, d, CtMethod.UCT, end_time=0.7, solver=Solver.HLLD)

    mm = p.schwarzschild_mass
    sg = lambda r, z: r * math.sqrt(1.0 + 2.0 * mm / math.hypot(r, z))
    rf = np.array(p.radial_faces())
    zf = np.array(p.z_faces())
    md, sc = _w_div_max(sg, rf, zf, B1, B2)
    assert md < 1e-12 * max(sc, 1.0), f"(R,z) UCT-HLLD w-div(B) broke: {md:.3e} (scale {sc:.3e})"


@needs_backend
@pytest.mark.parametrize("ct", [CtMethod.CONTACT, CtMethod.UCT])
def test_disk_field_loop_preserves_divergence(ct) -> None:
    from simbi_configs.examples.grmhd.gr_disk_field_loop import GrDiskFieldLoop

    d = tempfile.mkdtemp() + "/"
    p = GrDiskFieldLoop.from_cli(["--nr", "80", "--nphi", "64"])
    B1, B2 = _run(p, d, ct)

    mm = p.schwarzschild_mass
    sg = lambda r, _phi: r * math.sqrt(1.0 + 2.0 * mm / r)  # equator: r = R, phi-independent
    rf = np.array(p.radial_faces())
    pf = np.array(p.phi_faces())
    md, sc = _w_div_max(sg, rf, pf, B1, B2)
    assert md < 1e-12 * max(sc, 1.0), f"disk w-div(B) broke at {ct}: {md:.3e} (scale {sc:.3e})"
