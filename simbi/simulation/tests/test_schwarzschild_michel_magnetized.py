# =============================================================================
# test_schwarzschild_michel_magnetized.py
#
# the GRMHD gates on the magnetized michel monopole:
# the exact michel transonic hydro profile threaded by the divergence-free radial
# field sqrt(gamma) B^r = const on the schwarzschild grid. a radial field aligned
# with a radial flow exerts zero lorentz force, so the stationary solution is the
# UNMAGNETIZED michel profile while every magnetic term in U, F, the covariant
# source, and the KKC recovery is engaged — the cancellations are the gate:
#
#   identity-class: the staggered B^r is BITWISE static (the 1D induction flux is
#   identically zero by the shared-face antisymmetry), and the azimuthal/polar
#   momentum rows generate nothing (measured 0.0 and 1.6e-23).
#
#   consistency-class: the hydro hold against the michel solution converges under
#   refinement (measured L1 rho 1.19e-4 at 128 -> 4.4e-5 at 256, ratio 2.7 — the
#   bound fan is mildly more diffusive than the RHD banyuls-font gate's 5.0e-5);
#   the hydro profile is FIELD-INDEPENDENT (b_ref 0 vs 2.0 differ by ~4e-8
#   relative over t=10 — the discrete force-free cancellation); the one-step
#   den/m1/nrg residuals converge (measured ratios 2.1-2.6; a wrong magnetic
#   term is a resolution-independent floor).
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

_HOLD_TIME = 10.0
# measured hold: L1 rho vs michel solution 1.19e-4 (128) -> 4.41e-5 (256), ratio 2.7;
# tolerances carry ~3x margin, 1.8 separates convergence from a wrong-term floor.
_HOLD_TOL_128 = 3.6e-4
_HOLD_CONV = 1.8
# measured one-step |dU/dt|/(tau+D) L1 at 128: den 9.1e-5, m1 7.2e-4, nrg 3.3e-4
# (ratios 2.6/2.1/2.1); m2 1.6e-23, m3 and b1 exactly 0.0.
_RESID_TOL = {"den": 2.7e-4, "m1": 2.2e-3, "nrg": 1.0e-3}
_RESID_CONV = 1.5
_SILENT_ROWS_TOL = 1e-12


def _run(n: int, b_ref: float, t_end: float):
    from simbi_configs.examples.grmhd.gr_michel_magnetized import GrMichelMagnetized

    d = tempfile.mkdtemp() + "/"
    p = GrMichelMagnetized.from_cli(
        ["--resolution", str(n), "--b-ref", str(b_ref)]
    )
    p.end_time = t_end
    p.data_directory = d
    p.checkpoint_interval = max(t_end, 1.0)
    runner.run(p, compute_mode="cpu")
    crashed = glob.glob(os.path.join(d, "*crashed*.h5"))
    assert not crashed, f"magnetized michel crashed at n={n}, b_ref={b_ref}"
    first = sorted(glob.glob(os.path.join(d, "*chkpt.000_000*.h5")))[0]
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    return p, first, final


def _prims(fn: str, n: int) -> dict:
    with h5py.File(fn) as h:
        g = h["level_0/partition_0/hydro/primitives"]
        halo = (g["rho"].shape[0] - n) // 2
        sl = slice(halo, halo + n)
        out = {k: g[k][sl] for k in ("rho", "pre", "v1")}
        out["bface"] = h["level_0/partition_0/hydro/magnetic/B1/data"][:]
    return out


def _hold(n: int, b_ref: float):
    p, first, final = _run(n, b_ref, _HOLD_TIME)
    f0, f1 = _prims(first, n), _prims(final, n)
    sol = p.michel_solution()
    rc = np.array(p.cell_centroids())
    ref = np.array([sol.primitive(r)[0] for r in rc])
    interior = slice(2, n - 2)
    assert f1["pre"][interior].min() > 0.0, "pressure went non-positive"
    l1 = float(np.abs(f1["rho"][interior] / ref[interior] - 1.0).mean())
    db = float(np.abs(f1["bface"] - f0["bface"]).max())
    return l1, db, f1


@needs_backend
def test_magnetized_michel_holds_and_field_is_bitwise_static() -> None:
    l1_lo, db_lo, _ = _hold(128, 0.5)
    l1_hi, db_hi, _ = _hold(256, 0.5)
    # identity: the 1D induction flux of B^r is exactly zero (the normal field is the
    # SHARED staggered face value on both riemann sides), so B^r must not move AT ALL.
    assert db_lo == 0.0, f"staggered B^r moved: {db_lo:.3e}"
    assert db_hi == 0.0, f"staggered B^r moved at 256: {db_hi:.3e}"
    # consistency: the hydro hold converges toward the michel solution.
    assert l1_lo < _HOLD_TOL_128, f"128 hold L1 {l1_lo:.3e}"
    assert l1_lo / l1_hi > _HOLD_CONV, (
        f"hold does not converge: {l1_lo:.3e} -> {l1_hi:.3e} "
        f"(ratio ~1 means a wrong magnetic term)"
    )


@needs_backend
def test_hydro_profile_is_field_independent() -> None:
    # radial B aligned with radial v: E = -v x B = 0, J = 0 — the lorentz force
    # vanishes, so the evolved hydro must not know the field strength. measured
    # b=0 vs b=2: rho 3.6e-8, v1 1.1e-8, pre 9.0e-7 relative over t=10.
    _, _, weak = _hold(128, 0.0)
    _, _, strong = _hold(128, 2.0)
    for k in ("rho", "v1", "pre"):
        scale = float(np.abs(weak[k]).max())
        e = float(np.abs(weak[k] - strong[k]).max()) / scale
        assert e < 1e-6, f"{k} depends on the field strength: {e:.3e} relative"


def _one_step_residual(n: int) -> dict:
    dt = 1e-6
    _, first, final = _run(n, 0.5, dt)
    out = {}
    for tag, fn in (("0", first), ("1", final)):
        with h5py.File(fn) as h:
            c = h["level_0/conserved"]
            for k in ("den", "m1", "m2", "m3", "nrg", "b1"):
                out[k + tag] = c[k][:]
    sl = slice(2, 2 + n)
    scale = out["nrg0"][sl] + out["den0"][sl]
    return {
        k: float((np.abs(out[k + "1"][sl] - out[k + "0"][sl]) / dt / scale).mean())
        for k in ("den", "m1", "m2", "m3", "nrg", "b1")
    }


@needs_backend
def test_one_step_residual_is_truncation_and_converges() -> None:
    r_lo = _one_step_residual(128)
    r_hi = _one_step_residual(256)
    for k, tol in _RESID_TOL.items():
        assert r_lo[k] < tol, f"one-step {k} residual {r_lo[k]:.3e} exceeds {tol:.1e}"
        assert r_lo[k] / r_hi[k] > _RESID_CONV, (
            f"one-step {k} does not converge: {r_lo[k]:.3e} -> {r_hi[k]:.3e}"
        )
    # the rows with NO generator: azimuthal/polar momentum (radial flow, radial B,
    # axisymmetric metric) and the radial induction row. roundoff or exactly zero.
    for k in ("m2", "m3", "b1"):
        assert r_lo[k] < _SILENT_ROWS_TOL, f"one-step {k} generated: {r_lo[k]:.3e}"
