# =============================================================================
# test_schwarzschild_rotating_equilibrium.py
#
# the rotating-balance precision gates on the surface-free constant-l equilibrium
# (gr_fishbone_moncrief.RotatingEquilibrium) with all four boundaries DRIVEN —
# ghost bands pinned to the analytic state (a theta-stratified equilibrium is
# mathematically incompatible with mirror/copy ghosts: they impose dp/dtheta = 0
# where the state requires the centrifugal-balancing gradient).
#
# one-step residual: the sharpest instrument. an exact stationary state's discrete
# time derivative is pure truncation — smooth and second-order — so every conserved
# component's one-step residual must CONVERGE under refinement. a wrong term shows
# up as a resolution-independent residual in its component: the covariant S_theta
# law once carried orthonormal (arc-length) angular face weights instead of the
# coordinate alpha sqrt(gamma) measure, leaving every theta-direction force short
# by a factor r — invisible to every radial-flow and theta-uniform gate, and an
# O(1e-2) non-converging m2 residual here.
#
# hold: the equilibrium is held over many dynamical times with drift shrinking
# under refinement. requires the built cpu_ext backend; skipped otherwise.
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

# measured one-step residual L1 (|dU/dt| / (tau + D)) at 96x32: den 2.1e-4, m1 2.9e-4,
# m2 9.5e-5, m3 7.3e-4, nrg 1.3e-4 — tolerances carry ~3x margin. the broken angular
# weights put m2 at 1.2e-2 with ratio 1.0.
_RESID_TOL = {"den": 7e-4, "m1": 1e-3, "m2": 3e-4, "m3": 2.2e-3, "nrg": 4e-4}
# measured 96->192 L1 ratios 3.9-7.1 (second order and better); 2.5 separates a
# converging truncation residual from a wrong term (ratio 1) with margin.
_RESID_CONV = 2.5
# measured t=100 hold: full-domain L1 rho drift 5.4e-2 (96x32) -> 1.7e-2 (192x64),
# ratio 3.2 (the state carries slow neutrally-stable constant-l modes; convergence
# is the teeth, the absolute bound the sanity rail).
_HOLD_TOL_96 = 1.5e-1
_HOLD_CONV = 2.0
_HOLD_TIME = 100.0


def _run(nr: int, npolar: int, t_end: float):
    from simbi_configs.examples.gr_rotating_equilibrium import GrRotatingEquilibrium

    d = tempfile.mkdtemp() + "/"
    p = GrRotatingEquilibrium.from_cli(["--nr", str(nr), "--npolar", str(npolar)])
    p.end_time = t_end
    p.data_directory = d
    p.checkpoint_interval = max(t_end, 1.0)
    runner.run(p, compute_mode="cpu")
    crashed = glob.glob(os.path.join(d, "*crashed*.h5"))
    assert not crashed, f"rotating equilibrium crashed at {nr}x{npolar}"
    first = sorted(glob.glob(os.path.join(d, "*chkpt.000_000*.h5")))[0]
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    return p, first, final


def _cons(fn: str) -> dict:
    with h5py.File(fn) as h:
        c = h["level_0/conserved"]
        return {k: c[k][:] for k in ("den", "m1", "m2", "m3", "nrg")}


def _one_step_residual(nr: int, npolar: int) -> dict:
    dt = 1e-6
    _, first, final = _run(nr, npolar, dt)
    c0, c1 = _cons(first), _cons(final)
    halo = 2
    sl = (slice(halo, halo + npolar), slice(halo, halo + nr))
    scale = c0["nrg"][sl] + c0["den"][sl]
    return {
        k: float((np.abs(c1[k][sl] - c0[k][sl]) / dt / scale).mean())
        for k in ("den", "m1", "m2", "m3", "nrg")
    }


@needs_backend
def test_stationary_one_step_residual_is_truncation_and_converges() -> None:
    r_lo = _one_step_residual(96, 32)
    r_hi = _one_step_residual(192, 64)
    for k, tol in _RESID_TOL.items():
        assert r_lo[k] < tol, f"one-step {k} residual {r_lo[k]:.3e} exceeds {tol:.1e}"
        ratio = r_lo[k] / r_hi[k]
        assert ratio > _RESID_CONV, (
            f"one-step {k} residual does not converge: {r_lo[k]:.3e} -> {r_hi[k]:.3e} "
            f"(ratio {ratio:.2f}; ~1 means a wrong term in that conservation law)"
        )


def _hold_drift(nr: int, npolar: int) -> float:
    p, _, final = _run(nr, npolar, _HOLD_TIME)
    with h5py.File(final) as h:
        g = h["level_0/partition_0/hydro/primitives"]
        shp = g["rho"].shape
        halo = [(s - n) // 2 for s, n in zip(shp, (npolar, nr))]
        sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (npolar, nr)))
        rho, pre = g["rho"][sl], g["pre"][sl]
    assert pre.min() > 0.0, f"pressure went non-positive: {pre.min():.3e}"
    eq = p.equilibrium()
    (rmin, rmax) = p.bounds[0]
    (tmin, tmax) = p.bounds[1]
    q = (rmax / rmin) ** (1.0 / nr)
    rl = rmin * q ** np.arange(nr)
    rh = rl * q
    rc = 0.75 * (rh**4 - rl**4) / (rh**3 - rl**3)
    th = tmin + (np.arange(npolar) + 0.5) * (tmax - tmin) / npolar
    ref = np.array([[eq.primitive(r, t)[0] for r in rc] for t in th])
    return float(np.abs(rho / ref - 1.0).mean())


@needs_backend
def test_rotating_equilibrium_is_held_and_converges() -> None:
    e_lo = _hold_drift(96, 32)
    e_hi = _hold_drift(192, 64)
    assert e_lo < _HOLD_TOL_96, f"96x32 hold drift {e_lo:.3e}"
    assert e_lo / e_hi > _HOLD_CONV, (
        f"hold drift does not converge: {e_lo:.3e} -> {e_hi:.3e}"
    )
