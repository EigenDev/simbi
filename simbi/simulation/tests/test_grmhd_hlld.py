# =============================================================================
# test_grmhd_hlld.py
#
# the GR-HLLD gates (design 44 GR-HLLD): the ORTHONORMAL-frame MUB09 five-wave
# relativistic-MHD Riemann solver on a curved background.
#
# the GR solve maps the diagonal spatial metric to the local orthonormal frame
# (V_hat^i = sqrt(g_i) v^i, B_hat^i = sqrt(g_i) B^i), runs the VALIDATED flat
# MUB09 solver there, and maps the flux back exactly (F_D /= sqrt(g_n),
# F_S_j *= sqrt(g_j)/sqrt(g_n), F_B^i /= sqrt(g_i) sqrt(g_n)). this reduces to
# F(U) in the smooth limit and to the SR solver at identity gamma (both proven in
# riemann/hlld.rs), so it holds the transonic magnetized-michel monopole exactly
# as HLLE does -- the coordinate-frame reconstruction did NOT (its transverse
# star fields carried spurious sqrt(gamma) factors that grew a theta-momentum
# mode). the same orthonormal map drives the GR-UCT-HLLD wave-sum edge EMF, whose
# Phi telescopes to the coordinate B_t flux.
#
# gates: (1) HLLD flux + UCT-HLLD EMF holds the michel monopole to the 1D L1 with
# the staggered B static (E_phi = 0 for the smooth field); (2) HLLE is unaffected.
# requires the built cpu_ext backend; skipped otherwise.
# =============================================================================
import glob
import os
import tempfile

import numpy as np
import h5py
import pytest

from simbi.simulation import runner
from simbi.types import CtMethod, Solver

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)


def _run_michel(solver: Solver, ct: CtMethod):
    from simbi_configs.examples.grmhd.gr_michel_magnetized_2d import GrMichelMagnetized2D

    d = tempfile.mkdtemp() + "/"
    p = GrMichelMagnetized2D.from_cli(["--nr", "128", "--npolar", "16", "--b-ref", "0.5"])
    p.ct_method = ct
    p.solver = solver
    p.end_time = 10.0
    p.data_directory = d
    p.checkpoint_interval = 100.0
    runner.run(p, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*.h5")), f"michel crashed at {solver}/{ct}"
    first = sorted(glob.glob(os.path.join(d, "*chkpt.000_000*.h5")))[0]
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    return p, first, final


@needs_backend
def test_gr_hlld_flux_uct_hlld_emf_holds_the_michel_monopole() -> None:
    # the ORTHONORMAL-frame GR HLLD gas flux with the GR-UCT-HLLD wave-sum EMF must hold the
    # transonic magnetized monopole to the 1D L1 gate, positive pressure, and keep the staggered
    # B static (the smooth-field EMF is zero). this is the profile the coordinate-frame HLLD grew.
    p, first, final = _run_michel(Solver.HLLD, CtMethod.UCT)
    with h5py.File(final) as h:
        g = h["level_0/partition_0/hydro/primitives"]
        shp = g["rho"].shape
        halo = [(s - n) // 2 for s, n in zip(shp, (16, 128))]
        sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (16, 128)))
        rho, pre = g["rho"][sl], g["pre"][sl]
        b1_1 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
    with h5py.File(first) as h:
        b1_0 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
    assert pre.min() > 0.0, "pressure went non-positive under GR HLLD"
    assert float(np.abs(b1_1 - b1_0).max()) < 1e-8, "staggered B drifted under GR-UCT-HLLD"
    sol = p.michel_solution()
    rc = np.array(p.cell_centroids())
    ref = np.array([sol.primitive(r)[0] for r in rc])
    l1 = float(np.abs(rho[2:14, 2:126] / ref[None, 2:126] - 1.0).mean())
    assert l1 < 3.6e-4, f"GR HLLD michel hold L1 {l1:.3e} (the 1D gate is 1.19e-4)"


@needs_backend
def test_gr_hlle_still_runs() -> None:
    # the HLLE GR path is unaffected and remains the robust default.
    from simbi_configs.examples.grmhd.gr_field_loop import GrFieldLoop

    d = tempfile.mkdtemp() + "/"
    p = GrFieldLoop.from_cli(["--nr", "64", "--npolar", "32"])
    p.ct_method = CtMethod.UCT
    p.solver = Solver.HLLE
    p.end_time = 0.5
    p.data_directory = d
    p.checkpoint_interval = 100.0
    runner.run(p, compute_mode="cpu")
    assert not glob.glob(os.path.join(d, "*crashed*.h5")), "GR HLLE loop crashed"
    assert glob.glob(os.path.join(d, "*final*.h5")), "GR HLLE loop produced no output"
