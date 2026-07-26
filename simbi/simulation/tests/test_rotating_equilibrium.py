# =============================================================================
# test_rotating_equilibrium.py
#
# the rotating-balance precision gates on the surface-free constant-l equilibrium
# (gr_fishbone_moncrief.RotatingEquilibrium) with all four boundaries DRIVEN —
# ghost bands pinned to the analytic state (a theta-stratified equilibrium is
# mathematically incompatible with mirror/copy ghosts: they impose dp/dtheta = 0
# where the state requires the centrifugal-balancing gradient). parametrized over
# spin: a = 0 is the schwarzschild chart (zero shift); a = 0.9 is the spinning
# kerr ingoing kerr-schild chart, where the state carries the orbiter's radial
# drift v^r = beta^r/alpha and the flux's radial-shift riemann fan is load-bearing.
#
# one-step residual: the sharpest instrument. an exact stationary state's discrete
# time derivative is pure truncation — smooth and second-order — so every conserved
# component's one-step residual must CONVERGE under refinement. a wrong term shows
# up as a resolution-independent residual in its component: the covariant S_theta
# law's correct angular face weight is the coordinate alpha sqrt(gamma) measure;
# an orthonormal (arc-length) weight gives an O(1e-2) non-converging m2 residual,
# and the kerr-schild charts once dropped the shift-advection `-beta^r U` from the
# face flux entirely (an O(1e-1) non-converging residual in every transported
# component; the exact stationary state has zero transport velocity
# alpha v^r - beta^r = 0, so its den residual is analytically zero) — both
# invisible to every kernel-vs-kernel and qualitative-infall gate.
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

# measured one-step residual L1 (|dU/dt| / (tau + D)) at 96x32 — tolerances carry
# ~3x margin. a = 0: den 2.1e-4, m1 2.9e-4, m2 9.5e-5, m3 7.3e-4, nrg 1.3e-4 (the
# broken angular weights put m2 at 1.2e-2 with ratio 1.0). a = 0.9: den 4.5e-5,
# m1 2.3e-5, m2 8.2e-5, m3 1.7e-4, nrg 1.8e-5 (the dropped shift advection put
# den/m1/m3/nrg at 4e-3 .. 2e-1 with ratio 1.0-2.0).
_RESID_TOL = {
    0.0: {"den": 7e-4, "m1": 1e-3, "m2": 3e-4, "m3": 2.2e-3, "nrg": 4e-4},
    0.9: {"den": 1.5e-4, "m1": 7e-5, "m2": 3e-4, "m3": 5e-4, "nrg": 6e-5},
}
# measured 96->192 L1 ratios 3.9-7.1 (a = 0) and 3.6-3.9 (a = 0.9); 2.5 separates
# a converging truncation residual from a wrong term (ratio ~1) with margin.
_RESID_CONV = 2.5
# measured t=100 hold: full-domain L1 rho drift (96x32 -> 192x64) a = 0: 5.4e-2 ->
# 1.7e-2 (ratio 3.2); a = 0.9: 2.5e-3 -> 7.0e-4 (ratio 3.6). the state carries slow
# neutrally-stable constant-l modes; convergence is the teeth, the absolute bound
# the sanity rail. tolerances carry ~3x margin on the measured drift.
_HOLD_TOL_96 = {0.0: 1.5e-1, 0.9: 8e-3}
_HOLD_CONV = 2.0
_HOLD_TIME = 100.0
_SPINS = [0.0, 0.9]


def _run(nr: int, npolar: int, t_end: float, spin: float):
    from simbi.simulation.tests.fixtures.gr_rotating_equilibrium import (
        GrRotatingEquilibrium,
    )

    d = tempfile.mkdtemp() + "/"
    p = GrRotatingEquilibrium.from_cli(
        ["--nr", str(nr), "--npolar", str(npolar), "--kerr-spin", str(spin)]
    )
    p.end_time = t_end
    p.data_directory = d
    p.checkpoint_interval = max(t_end, 1.0)
    runner.run(p, compute_mode="cpu")
    # GUARD-ACTIVATION CENSUS: this is an EXACT stationary solution, smooth and warm — no limiter
    # has any physical business firing on it, at either spin. a nonzero count would mean the
    # equilibrium is being held by a floor rather than by the scheme, which would also silently
    # contaminate the one-step residual this file measures. the count covers the whole defensive
    # surface: the admissible-boundary projection and the first-order redo run INSIDE
    # `fofc_orchestrate` (which early-returns when nothing is flagged), and the relativistic
    # velocity ceiling only binds an out-of-cone state — exactly what sets the flag.
    fallback, freeze, _, _ = _BACKEND.guard_census()
    assert (fallback, freeze) == (0, 0), (
        f"rotating equilibrium tripped a limiter at {nr}x{npolar} spin={spin}: {fallback} "
        f"first-order fallback cell-steps, {freeze} freezes — an exact stationary state must be "
        "held by the scheme, and a guard firing here also taints the one-step residual"
    )
    crashed = glob.glob(os.path.join(d, "*crashed*.h5"))
    assert not crashed, f"rotating equilibrium crashed at {nr}x{npolar}"
    first = sorted(glob.glob(os.path.join(d, "*chkpt.000_000*.h5")))[0]
    final = glob.glob(os.path.join(d, "*final*.h5"))[0]
    return p, first, final


def _cons(fn: str) -> dict:
    with h5py.File(fn) as h:
        c = h["level_0/conserved"]
        return {k: c[k][:] for k in ("den", "m1", "m2", "m3", "nrg")}


def _one_step_residual(nr: int, npolar: int, spin: float) -> dict:
    dt = 1e-6
    _, first, final = _run(nr, npolar, dt, spin)
    c0, c1 = _cons(first), _cons(final)
    halo = 2
    sl = (slice(halo, halo + npolar), slice(halo, halo + nr))
    scale = c0["nrg"][sl] + c0["den"][sl]
    return {
        k: float((np.abs(c1[k][sl] - c0[k][sl]) / dt / scale).mean())
        for k in ("den", "m1", "m2", "m3", "nrg")
    }


@needs_backend
@pytest.mark.parametrize("spin", _SPINS)
def test_stationary_one_step_residual_is_truncation_and_converges(spin) -> None:
    r_lo = _one_step_residual(96, 32, spin)
    r_hi = _one_step_residual(192, 64, spin)
    for k, tol in _RESID_TOL[spin].items():
        assert r_lo[k] < tol, f"one-step {k} residual {r_lo[k]:.3e} exceeds {tol:.1e}"
        ratio = r_lo[k] / r_hi[k]
        assert ratio > _RESID_CONV, (
            f"one-step {k} residual does not converge: {r_lo[k]:.3e} -> {r_hi[k]:.3e} "
            f"(ratio {ratio:.2f}; ~1 means a wrong term in that conservation law)"
        )


def _hold_drift(nr: int, npolar: int, spin: float) -> float:
    p, _, final = _run(nr, npolar, _HOLD_TIME, spin)
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
@pytest.mark.parametrize("spin", _SPINS)
def test_rotating_equilibrium_is_held_and_converges(spin) -> None:
    e_lo = _hold_drift(96, 32, spin)
    e_hi = _hold_drift(192, 64, spin)
    assert e_lo < _HOLD_TOL_96[spin], f"96x32 hold drift {e_lo:.3e}"
    assert e_lo / e_hi > _HOLD_CONV, (
        f"hold drift does not converge: {e_lo:.3e} -> {e_hi:.3e}"
    )
