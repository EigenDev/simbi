# =============================================================================
# test_schwarzschild_michel_magnetized_2d.py
#
# the curved-CT instrument: the michel transonic profile on
# the (r, theta) wedge threaded by the theta-uniform radial monopole, run through
# the FULL 2D GRMHD machinery — the densitized corner EMF (contact assembly), the
# densitized-space curl, the metric-contracted face->cell interpolation, and the
# covariant bcell predictor. the gates:
#
#   identity-class: the w-weighted divergence of the staggered field (w =
#   sqrt(gamma)(face center) x coordinate length — the discrete divergence the
#   curl preserves by construction) stays MACHINE ZERO through the full t = 10
#   evolution (measured 6.3e-15 against a 0.29 flux scale over ~5000 steps); the
#   one-step b1/m3 rows are exactly 0.0 and m2/b2 sit at cancellation roundoff.
#
#   consistency-class: the hold L1 rho vs the michel solution EQUALS the 1D gate's
#   value (1.192e-4 at nr = 128 — the wedge adds nothing), the one-step
#   den/m1/nrg residuals equal the 1D values and converge (ratios 2.1-2.6), and
#   the staggered field drift is roundoff-accumulation only (measured 4.3e-10
#   over t = 10: the theta-momentum's exact-cancellation noise seeds
#   v_theta ~ 1e-11 which feeds the EMF; a wrong metric factor in the
#   EMF/curl/interpolation chain sits at truncation scale, 4+ orders above).
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

_NR, _NP = 128, 16
_HOLD_TIME = 10.0
_HOLD_TOL = 3.6e-4  # measured 1.192e-4 (== the 1D gate), ~3x margin
_DB_TOL = 1e-8      # measured 4.3e-10 (roundoff accumulation over t=10)
_DIV_TOL = 1e-12    # measured 6.3e-15 absolute vs a 0.29 flux scale
# measured one-step L1 at 128x16 == the 1D values: den 9.1e-5, m1 7.2e-4, nrg 3.3e-4
# (ratios 2.6/2.1/2.1 to 256x32); m2 4.9e-12, m3 0.0, b1 0.0, b2 1.6e-25.
_RESID_TOL = {"den": 2.7e-4, "m1": 2.2e-3, "nrg": 1.0e-3}
_RESID_CONV = 1.5
_SILENT_TOL = 1e-10


def _run(nr: int, npolar: int, t_end: float):
    from simbi.simulation.tests.fixtures.gr_michel_magnetized_2d import GrMichelMagnetized2D

    d = tempfile.mkdtemp() + "/"
    p = GrMichelMagnetized2D.from_cli(
        ["--nr", str(nr), "--npolar", str(npolar), "--b-ref", "0.5"]
    )
    p.end_time = t_end
    p.data_directory = d
    p.checkpoint_interval = max(t_end, 1.0)
    runner.run(p, compute_mode="cpu")
    crashed = glob.glob(os.path.join(d, "*crashed*.h5"))
    assert not crashed, f"2d magnetized michel crashed at {nr}x{npolar}"
    first = sorted(glob.glob(os.path.join(d, "*chkpt.000_000*.h5")))[0]
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    return p, first, final


def _w_div_max(p, fn: str) -> tuple[float, float]:
    """(max |w-weighted div(B)|, max face-flux scale) over the interior."""
    with h5py.File(fn) as h:
        b1 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
        b2 = h["level_0/partition_0/hydro/magnetic/B2/data"][:]
    mm = p.schwarzschild_mass
    faces = np.array(p.radial_faces())
    (tmin, tmax) = p.bounds[1]
    dth = (tmax - tmin) / p.npolar
    th_f = tmin + np.arange(p.npolar + 1) * dth
    sqrtg = lambda r, th: r * r * np.sin(th) / np.sqrt(1.0 - 2.0 * mm / r)
    r_c = 0.5 * (faces[:-1] + faces[1:])
    th_c = 0.5 * (th_f[:-1] + th_f[1:])
    max_div, scale = 0.0, 0.0
    for j in range(p.npolar):
        for i in range(p.nr):
            dr = faces[i + 1] - faces[i]
            div = (
                sqrtg(faces[i + 1], th_c[j]) * dth * b1[j, i + 1]
                - sqrtg(faces[i], th_c[j]) * dth * b1[j, i]
                + sqrtg(r_c[i], th_f[j + 1]) * dr * b2[j + 1, i]
                - sqrtg(r_c[i], th_f[j]) * dr * b2[j, i]
            )
            max_div = max(max_div, abs(div))
            scale = max(scale, abs(sqrtg(faces[i + 1], th_c[j]) * dth * b1[j, i + 1]))
    return max_div, scale


@needs_backend
def test_curved_ct_holds_michel_and_divergence() -> None:
    p, first, final = _run(_NR, _NP, _HOLD_TIME)
    with h5py.File(final) as h:
        g = h["level_0/partition_0/hydro/primitives"]
        shp = g["rho"].shape
        halo = [(s - n) // 2 for s, n in zip(shp, (_NP, _NR))]
        sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (_NP, _NR)))
        rho, pre, v2 = g["rho"][sl], g["pre"][sl], g["v2"][sl]
        b1_1 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
        b2_1 = h["level_0/partition_0/hydro/magnetic/B2/data"][:]
    with h5py.File(first) as h:
        b1_0 = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
        b2_0 = h["level_0/partition_0/hydro/magnetic/B2/data"][:]
    assert pre.min() > 0.0, "pressure went non-positive"
    # the staggered field: roundoff-accumulation drift only.
    db = max(float(np.abs(b1_1 - b1_0).max()), float(np.abs(b2_1 - b2_0).max()))
    assert db < _DB_TOL, f"staggered B drifted beyond roundoff: {db:.3e}"
    assert float(np.abs(v2).max()) < 1e-9, "theta flow beyond cancellation roundoff"
    # the hydro hold equals the 1D gate.
    sol = p.michel_solution()
    rc = np.array(p.cell_centroids())
    ref = np.array([sol.primitive(r)[0] for r in rc])
    interior = (slice(2, _NP - 2), slice(2, _NR - 2))
    l1 = float(np.abs(rho[interior] / ref[None, 2 : _NR - 2] - 1.0).mean())
    assert l1 < _HOLD_TOL, f"2d hold L1 {l1:.3e}"
    # the w-weighted divergence stays machine zero through the full evolution.
    max_div, scale = _w_div_max(p, final)
    assert max_div < _DIV_TOL * max(scale, 1.0), (
        f"w-weighted div(B) broke: {max_div:.3e} (flux scale {scale:.3e})"
    )


def _one_step_residual(nr: int, npolar: int) -> dict:
    dt = 1e-6
    _, first, final = _run(nr, npolar, dt)
    out = {}
    for tag, fn in (("0", first), ("1", final)):
        with h5py.File(fn) as h:
            c = h["level_0/conserved"]
            for k in ("den", "m1", "m2", "m3", "nrg", "b1", "b2"):
                out[k + tag] = c[k][:]
    sl = (slice(2, 2 + npolar), slice(2, 2 + nr))
    scale = out["nrg0"][sl] + out["den0"][sl]
    return {
        k: float((np.abs(out[k + "1"][sl] - out[k + "0"][sl]) / dt / scale).mean())
        for k in ("den", "m1", "m2", "m3", "nrg", "b1", "b2")
    }


@needs_backend
def test_one_step_residual_matches_1d_and_converges() -> None:
    r_lo = _one_step_residual(_NR, _NP)
    r_hi = _one_step_residual(2 * _NR, 2 * _NP)
    for k, tol in _RESID_TOL.items():
        assert r_lo[k] < tol, f"one-step {k} residual {r_lo[k]:.3e} exceeds {tol:.1e}"
        assert r_lo[k] / r_hi[k] > _RESID_CONV, (
            f"one-step {k} does not converge: {r_lo[k]:.3e} -> {r_hi[k]:.3e}"
        )
    # rows with NO generator on the theta-uniform radial monopole: the theta/azimuthal
    # momenta and both staggered induction rows.
    for k in ("m2", "m3", "b1", "b2"):
        assert r_lo[k] < _SILENT_TOL, f"one-step {k} generated: {r_lo[k]:.3e}"
