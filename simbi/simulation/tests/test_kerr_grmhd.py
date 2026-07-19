# =============================================================================
# test_kerr_grmhd.py
#
# the spinning-KERR GRMHD gate: the full kerr RMHD kernel
# path — the tetrad HLLD on the NON-DIAGONAL gamma_{r phi}, the moving-interface
# radial shift, the EM-stress covariant source with the azimuthal (swirl)
# momentum, the metric-aware c2p, and the kerr-wired constrained transport.
#
# the gate is the w-weighted div(B) machine-zero + stability on an advected
# poloidal loop at spin 0.9. this ALSO guards the python->rust dispatch: a Kerr
# MHD config that silently falls through to the flat path runs Minkowski
# kernels, and the KERR sqrt(gamma)-weighted
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
@pytest.mark.parametrize(
    "solver,ct",
    [
        (Solver.HLLE, CtMethod.CONTACT),
        (Solver.HLLE, CtMethod.UCT),
        (Solver.HLLD, CtMethod.UCT),  # the sharp tetrad HLLD flux + UCT-HLLD wave-sum EMF on kerr
    ],
)
def test_kerr_field_loop_divergence_free_and_stable(solver, ct) -> None:
    from simbi.simulation.tests.fixtures.gr_kerr_field_loop import GrKerrFieldLoop

    d = tempfile.mkdtemp() + "/"
    p = GrKerrFieldLoop.from_cli(["--nr", "128", "--npolar", "64", "--kerr-spin", "0.9"])
    p.ct_method = ct
    p.solver = solver
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


def _roteq_hold_l1(nr, npolar):
    from simbi.simulation.tests.fixtures.gr_rotating_equilibrium_mhd import GrRotatingEquilibriumMhd

    d = tempfile.mkdtemp() + "/"
    p = GrRotatingEquilibriumMhd.from_cli(
        ["--nr", str(nr), "--npolar", str(npolar), "--kerr-spin", "0.9"]
    )
    p.solver = Solver.HLLE
    p.ct_method = CtMethod.CONTACT
    p.end_time = 10.0
    p.data_directory = d
    p.checkpoint_interval = 100.0
    runner.run(p, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*.h5")), "rot-eq-mhd crashed"
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    with h5py.File(final) as h:
        g = h["level_0/partition_0/hydro/primitives"]
        shp = g["rho"].shape
        halo = [(s - n) // 2 for s, n in zip(shp, (npolar, nr))]
        sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (npolar, nr)))
        rho = g["rho"][sl]
    eq = p.equilibrium()
    (rmin, rmax) = p.bounds[0]
    (tmin, tmax) = p.bounds[1]
    qf = (rmax / rmin) ** (1.0 / nr)
    dth = (tmax - tmin) / npolar
    ref = np.zeros((npolar, nr))
    for jj in range(npolar):
        th = tmin + (jj + 0.5) * dth
        for ii in range(nr):
            rl = rmin * qf**ii
            rh = rl * qf
            rr = 0.75 * (rh**4 - rl**4) / (rh**3 - rl**3)
            ref[jj, ii] = eq.primitive(rr, th)[0]
    return float(np.abs(rho[:, 3:-3] / ref[:, 3:-3] - 1.0).mean())


@needs_backend
def test_kerr_rotating_equilibrium_holds_and_converges() -> None:
    # the frame-dragging ACCURACY oracle: the RMHD (B=0) constant-l orbit on the spinning-kerr
    # background must HOLD to truncation and CONVERGE under refinement. the v^phi w-reconstruction
    # keeps the flow on the zero-angular-momentum dragging manifold; a raw reconstruction generates
    # spurious S_phi, degrades the hold, and breaks convergence.
    l1_coarse = _roteq_hold_l1(64, 24)
    l1_fine = _roteq_hold_l1(128, 48)
    assert l1_coarse < 1e-3, f"rot-eq hold too large (coarse): {l1_coarse:.3e}"
    assert l1_fine < l1_coarse, "rot-eq hold did not decrease under refinement"
    assert l1_coarse / l1_fine > 2.0, (
        f"rot-eq hold not converging (ratio {l1_coarse / l1_fine:.2f}): the frame-dragging "
        "w-reconstruction is not holding the orbit"
    )


@needs_backend
def test_magnetized_fm_torus_seeds_divergence_free_and_stable() -> None:
    # the MRI initial condition: the fat FM torus threaded with a weak
    # beta-normalized poloidal seed field on the spinning-kerr RMHD path (tetrad HLLD + UCT-HLLD).
    # the seed must be div-free to machine zero, the torus core resolved, and the state stable.
    from simbi_configs.examples.grmhd.gr_fishbone_moncrief_mhd import GrFishboneMoncriefMhd

    d = tempfile.mkdtemp() + "/"
    p = GrFishboneMoncriefMhd.from_cli(
        ["--nr", "96", "--npolar", "48", "--kerr-spin", "0.9", "--target-beta", "100"]
    )
    p.end_time = 5.0
    p.data_directory = d
    p.checkpoint_interval = 100.0
    runner.run(p, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*.h5")), "fm torus crashed"
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    with h5py.File(final) as h:
        B1 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
        B2 = h["level_0/partition_0/hydro/magnetic/B2/data"][:]
        g = h["level_0/partition_0/hydro/primitives"]
        shp = g["rho"].shape
        halo = [(s - n) // 2 for s, n in zip(shp, (48, 96))]
        sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (48, 96)))
        pre, rho = g["pre"][sl], g["rho"][sl]
    assert pre.min() > 0.0, "pressure went non-positive"
    assert not np.isnan(rho).any(), "NaN in the torus"
    assert float(rho.max()) > 0.5, "torus core not resolved"
    assert float(np.abs(B1).max()) > 1e-4, "seed field vanished"
    mm, a = p.schwarzschild_mass, p.kerr_spin
    nr, npolar = p.resolution
    (rmin, rmax) = p.bounds[0]
    (tmin, tmax) = p.bounds[1]
    qf = (rmax / rmin) ** (1.0 / nr)
    dth = (tmax - tmin) / npolar
    rf = [rmin * qf**ii for ii in range(nr + 1)]
    tf = [tmin + jj * dth for jj in range(npolar + 1)]
    rc = [0.5 * (rf[i] + rf[i + 1]) for i in range(nr)]
    tc = [0.5 * (tf[j] + tf[j + 1]) for j in range(npolar)]

    def sg(r, th):
        s = r * r + a * a * math.cos(th) ** 2
        return s * math.sin(th) * math.sqrt(1.0 + 2.0 * mm * r / s)

    md = sc = 0.0
    for j in range(npolar):
        for i in range(nr):
            div = (
                sg(rf[i + 1], tc[j]) * dth * B1[j, i + 1]
                - sg(rf[i], tc[j]) * dth * B1[j, i]
                + sg(rc[i], tf[j + 1]) * (rf[i + 1] - rf[i]) * B2[j + 1, i]
                - sg(rc[i], tf[j]) * (rf[i + 1] - rf[i]) * B2[j, i]
            )
            md = max(md, abs(div))
            sc = max(sc, abs(sg(rf[i + 1], tc[j]) * dth * B1[j, i + 1]))
    assert md / max(sc, 1e-30) < 1e-10, f"fm-torus seed div(B) not machine-zero: {md / sc:.3e}"
