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
#   consistency-class: the hydro hold against the michel solution CONVERGES under
#   refinement, the hydro profile is FIELD-INDEPENDENT (a radial field along a
#   radial flow exerts no force, so the evolved gas must not know the field
#   strength), and the one-step den/m1/nrg residuals converge. a wrong magnetic
#   term is a resolution-independent floor, so the ORDER is what discriminates.
#
# every consistency assertion is stated WITHOUT a grid in it -- a measured order and
# an extrapolated error constant (see convergence.py) rather than an absolute
# tolerance at one resolution. an absolute bound encodes the resolution AND the
# scheme's dissipation, so a sharper wave-speed estimate reads as a physics failure.
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
from simbi_configs.examples.grhd.gr_michel import michel_chart
from simbi.simulation.tests.convergence import (
    assert_converges,
    assert_structurally_silent,
    convergence,
)

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

# the hold is measured on the HORIZON-PENETRATING chart, at a time where the state has
# actually evolved.
#
# t = 0.1 is too early to measure a convergence order: the state has barely departed
# from the seeded solution, so the residual is the initial condition's own
# representation rather than evolution truncation, and it does not fall under
# refinement (measured p = -0.01 at L1 ~ 5.6e-7). by t = 0.5 the flow has evolved and
# the truncation error dominates.
_HOLD_TIME = 0.5
# measured in kerr-schild at t = 0.5: L1 rho vs michel 1.076e-5 (128) -> 3.353e-6 (256),
# order p = 1.68, constant C = 3.78e-2, with dt at 0.31 / 0.71 of the light-crossing
# step and ZERO limiter activations at either resolution. the same hold in schwarzschild
# coordinates converged at only p = 0.53 and, past t ~ 0.2, stopped converging at all as
# the timestep collapsed -- the chart, not the scheme.
#
# the bounds are stated grid-free (see convergence.py): MIN_ORDER separates convergence
# from a resolution-independent floor with ~1.7x margin on the measurement, and
# MAX_CONSTANT carries ~3x, making it a smoke alarm for a uniformly more diffusive
# scheme.
_HOLD_MIN_ORDER = 1.0
_HOLD_MAX_CONSTANT = 1.2e-1
# a smooth stationary solution needs no limiter, and its step must stay within reach of
# the light-crossing step. both fail FIRST if the timestep collapse ever migrates in.
_HOLD_MIN_DT_FRACTION = 0.05
# the one-step residual gate, stated WITHOUT reference to a grid.
#
# an absolute tolerance on |dU/dt| pins the resolution into the number, so it reports a
# physics failure whenever the discretization's error CONSTANT moves for a legitimate
# reason: a sharper wave-speed estimate changes the HLLE dissipation and lifts every row
# by about a factor of three at unchanged convergence order, which a fixed absolute bound
# reports as a wrong magnetic term.
#
# what is asserted instead is the pair that does not mention a grid (see
# convergence.py): the measured order p, and the extrapolated constant C = E N^p.
#
# measured 128 -> 256:  den p=1.59 C=7.30e-1 | m1 p=0.82 C=1.37e-1 | nrg p=0.93 C=7.74e-2
# MIN_ORDER separates convergence from a floor: a wrong term does not fall under
# refinement at all (p -> 0), so 0.5 sits well below every measured order and well above
# the failure it detects. MAX_CONSTANT carries ~3x margin over the measurement, making it
# a smoke alarm for a uniformly more diffusive scheme.
_RESID_MIN_ORDER = 0.5
_RESID_MAX_CONSTANT = {"den": 2.5, "m1": 5.0e-1, "nrg": 3.0e-1}
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
    runner.run(p, compute_mode="cpu", max_steps=4000)
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
    _BACKEND.reset_guard_census()
    p, first, final = _run(n, b_ref, _HOLD_TIME)
    f0, f1 = _prims(first, n), _prims(final, n)
    sol = p.michel_solution()
    rc = np.array(p.cell_centroids())
    ref = np.array([sol.primitive(r, michel_chart(p.spacetime))[0] for r in rc])
    interior = slice(2, n - 2)
    assert f1["pre"][interior].min() > 0.0, "pressure went non-positive"
    l1 = float(np.abs(f1["rho"][interior] / ref[interior] - 1.0).mean())
    db = float(np.abs(f1["bface"] - f0["bface"]).max())
    # the health of the discretization at the point the hold is read: the selected step
    # against the light-crossing step, and whether any limiter fired at all. a smooth
    # stationary solution needs neither, so both are structural rather than tuned.
    with h5py.File(final) as h:
        m = h["metadata"].attrs
        dt, cfl = float(m["dt"]), float(m["cfl"])
    rc = np.array(p.cell_centroids())
    dt_fraction = dt / (cfl * float(rc[1] - rc[0]))
    fb, fz, _, _ = _BACKEND.guard_census()
    return l1, db, f1, dt_fraction, (fb, fz)


@needs_backend
def test_magnetized_michel_holds_and_field_is_bitwise_static() -> None:
    l1_lo, db_lo, _, frac_lo, guards_lo = _hold(128, 0.5)
    l1_hi, db_hi, _, frac_hi, guards_hi = _hold(256, 0.5)
    # identity: the 1D induction flux of B^r is exactly zero (the normal field is the
    # SHARED staggered face value on both riemann sides), so B^r must not move AT ALL.
    assert db_lo == 0.0, f"staggered B^r moved: {db_lo:.3e}"
    assert db_hi == 0.0, f"staggered B^r moved at 256: {db_hi:.3e}"
    # the discretization is healthy where the hold is read: no limiter fires on a smooth
    # stationary solution, and the step stays within reach of the light-crossing step.
    # these fail FIRST if the timestep collapse ever migrates earlier in the evolution, so
    # the convergence assertion can never silently invert instead.
    for n, frac, (fb, fz) in ((128, frac_lo, guards_lo), (256, frac_hi, guards_hi)):
        assert (fb, fz) == (0, 0), (
            f"a limiter fired on the stationary michel solution at N={n}: "
            f"{fb} first-order fallbacks, {fz} freezes"
        )
        assert frac > _HOLD_MIN_DT_FRACTION, (
            f"the timestep collapsed at N={n}: dt is {frac:.3e} of the light-crossing "
            f"step (bound {_HOLD_MIN_DT_FRACTION:.0e}) -- something other than wave "
            "propagation is setting it"
        )
    # consistency: the hydro hold converges toward the michel solution, stated without
    # reference to the grid it was measured on.
    assert_converges(
        convergence(e_coarse=l1_lo, e_fine=l1_hi, n_coarse=128, n_fine=256),
        min_order=_HOLD_MIN_ORDER,
        max_constant=_HOLD_MAX_CONSTANT,
        label="michel hold L1",
    )


@needs_backend
def test_hydro_profile_is_field_independent() -> None:
    # radial B aligned with radial v: E = -v x B = 0, J = 0 — the lorentz force
    # vanishes, so the evolved hydro must not know the field strength. measured
    # b=0 vs b=2: rho 3.6e-8, v1 1.1e-8, pre 9.0e-7 relative over t=10.
    _, _, weak, _, _ = _hold(128, 0.0)
    _, _, strong, _, _ = _hold(128, 2.0)
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
    for k, max_c in _RESID_MAX_CONSTANT.items():
        fit = convergence(
            e_coarse=r_lo[k], e_fine=r_hi[k], n_coarse=128, n_fine=256
        )
        assert_converges(
            fit,
            min_order=_RESID_MIN_ORDER,
            max_constant=max_c,
            label=f"one-step {k} residual",
        )
    # the rows with NO generator: azimuthal/polar momentum (radial flow, radial B,
    # axisymmetric metric) and the radial induction row. these are exact cancellations,
    # so the claim is structural -- zero, not small -- and carries no resolution
    # dependence to normalize away.
    for k in ("m2", "m3", "b1"):
        assert_structurally_silent(
            r_lo[k], tol=_SILENT_ROWS_TOL, label=f"one-step {k}"
        )
