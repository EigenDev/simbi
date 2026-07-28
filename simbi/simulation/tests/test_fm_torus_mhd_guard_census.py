# =============================================================================
# test_fm_torus_mhd_guard_census.py
#
# the magnetized spinning fishbone-moncrief torus on the 3d cartesian kerr
# chart: the guard-activation gate for the MAGNETIZED admissibility projection.
#
# the unmagnetized torus gates the hydrodynamic projection, whose admissible set
# is a second-order cone -- necessary AND sufficient, so every cell is
# recoverable by blending toward the stage-input anchor. the magnetized set is
# strictly smaller: a state can satisfy the B-free cone
# q = E - sqrt(D^2 + |S|^2) > 0 and still admit no physical primitive, because
# the magnetic energy leaves no positive gas pressure to recover. the sufficient
# condition adds psi > 0 (Wu & Tang, arXiv:1709.05838, theorem 2.1). this run is
# the stress case for that distinction -- a weak poloidal loop threading the
# spinning torus, where the near-horizon infall is both magnetized and stiff.
#
# the gates:
# - the run completes to the end time within the step cap (a dt collapse burns
#   the cap and is reported as the failure it is);
# - ZERO freezes outside the horizon. a freeze is a cell no flux could update
#   admissibly, and outside the horizon that is a physical breakdown. the
#   interior of r_+ is causally disconnected fiction and is exempt, but its
#   count is REPORTED, because the magnetic field is constrained-transport
#   evolved and cannot be blended -- so the projection searches only the slice
#   B = B_candidate, and an anchor admissible with its own field need not be
#   admissible in that slice. those cells are genuinely unrecoverable by
#   blending and fall through to the freeze tier by construction, not by defect.
#
# usage:
#   pytest simbi/simulation/tests/test_fm_torus_mhd_guard_census.py -s
#   (-s to see the reported census counts)
# =============================================================================
import glob
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation import runner

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

from simbi_configs.examples.grmhd.gr_fishbone_moncrief_mhd_cartesian import (
    GrFishboneMoncriefMhdCartesian,
)

RES = 48
SPIN = 0.9
END_TIME = 5.0

# the STEP BUDGET is a WALL-CLOCK guard, not a physical claim: it exists so a genuine dt collapse
# ends the suite instead of running forever. exceeding it is reported as a budget overrun, NOT as a
# physics failure. those were previously the same assertion, and the consequence was that a constant
# calibrated against one dt regime (396 steps, before the source-admissibility rate was tightened)
# reported a healthy torus -- reaching t_final with zero freezes -- as a stalled one.
STEP_BUDGET = 8000

# THE dt COLLAPSE BOUND, measured against the light-crossing step rather than a step count.
#
# nothing propagates faster than c, so `dt_light = cfl * dx / c` is the largest step the hyperbolic
# system admits and the only resolution-independent scale to judge dt against. a collapse is orders
# of magnitude, not a factor of a few: on this problem the smallest HEALTHY step is ~4.7e-5 of
# dt_light (the source-admissibility rate exceeds the flux rate by ~560x here -- a real cost, but a
# cost, not a breakdown), whereas an actual collapse reached ~2e-15 of it. this bound sits four
# orders below the healthy floor and eleven above the collapse, so it detects the failure mode
# without encoding today's efficiency as a requirement. it needs no recalibration when the
# resolution, domain, or cfl change, because it scales with all three.
MIN_DT_FRACTION = 1.0e-8


@needs_backend
def test_magnetized_fm_torus_never_freezes_outside_the_horizon() -> None:
    d = tempfile.mkdtemp() + "/"
    # the thick-torus pair: at a = 0.9 the compact (r_in = 8, kappa = 1.01)
    # equilibrium degenerates to a sub-cell sliver.
    p = GrFishboneMoncriefMhdCartesian(
        kerr_spin=SPIN,
        r_in=6.0,
        kappa=1.15,
        resolution=(RES, RES, RES),
        end_time=END_TIME,
        checkpoint_interval=1.0e30,
        data_directory=Path(d),
    )
    runner.run(p, compute_mode="cpu", max_steps=STEP_BUDGET)

    fb, fz, fb_h, fz_h = _BACKEND.guard_census()
    ext_fz, ext_fb = fz - fz_h, fb - fb_h
    print(
        f"\nmagnetized FM torus (a={SPIN}, {RES}^3) guard census:"
        f"\n  first-order fallbacks : {fb} ({ext_fb} exterior, {fb_h} interior)"
        f"\n  freezes              : {fz} ({ext_fz} exterior, {fz_h} interior)"
    )

    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, f"magnetized FM torus run (a={SPIN}) crashed"
    with h5py.File(finals[0], "r") as h:
        t_final = float(h["metadata"].attrs["time"])
        # the run's OWN recorded cfl and final step, so the bound below cannot drift from the
        # problem it is judging.
        cfl = float(h["metadata"].attrs["cfl"])
        dt_final = float(h["metadata"].attrs["dt"])
        steps = int(h["metadata"].attrs["iteration"])
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - RES) // 2
        sl = slice(halo, halo + RES)
        rho = prims["rho"][sl, sl, sl]
        bsq = sum(prims[f"b{k}"][sl, sl, sl] ** 2 for k in (1, 2, 3))

    assert np.isfinite(rho).all(), "non-finite density in the magnetized torus"
    assert rho.min() > 0.0, "density went non-positive"

    # PREMISE: the field must survive the evolution. an unmagnetized run would
    # collapse psi onto the B-free cone, and every assertion below would hold
    # while testing nothing about the magnetized condition.
    assert bsq.max() > 0.0, (
        "the evolved state carries NO magnetic field; psi degenerates to the "
        "hydrodynamic cone and this gate is vacuous"
    )

    # the light-crossing step: c = 1 in these units, so this is cfl * dx.
    dx = 2.0 * p.half_width / RES
    dt_light = cfl * dx
    print(
        f"  steps                : {steps} (budget {STEP_BUDGET})"
        f"\n  dt_final / dt_light  : {dt_final / dt_light:.3e}"
        f"  (collapse bound {MIN_DT_FRACTION:.0e})"
    )

    # THE EVOLUTION COMPLETED. a budget overrun and a crash are different failures with different
    # fixes, so they are reported as different things rather than both as "stalled".
    assert t_final > END_TIME - 1e-3, (
        f"the magnetized a={SPIN} torus reached only t = {t_final:.4f} < {END_TIME}: "
        + (
            f"it exhausted the {STEP_BUDGET}-step WALL-CLOCK budget, which is a cost problem, "
            f"not a physics one -- check dt_final/dt_light = {dt_final / dt_light:.3e} against "
            "the collapse bound before treating this as a breakdown"
            if steps >= STEP_BUDGET
            else f"it stopped after {steps} steps without exhausting the budget, so the evolution "
            "terminated on its own -- a crash, not a cost"
        )
    )

    # THE TIMESTEP DID NOT COLLAPSE. this is the invariant the old step count was standing in for,
    # expressed against the only scale that means anything: a step is healthy while it remains a
    # finite fraction of what the fastest signal permits.
    assert dt_final > MIN_DT_FRACTION * dt_light, (
        f"the magnetized a={SPIN} torus timestep COLLAPSED: dt = {dt_final:.4e} is "
        f"{dt_final / dt_light:.3e} of the light-crossing step {dt_light:.4e} "
        f"(bound {MIN_DT_FRACTION:.0e}). a step this far below the hyperbolic limit is set by "
        "something other than wave propagation -- the source-admissibility rate against a cell "
        "whose admissible margin has gone to zero"
    )

    assert ext_fz == 0, (
        f"the magnetized a={SPIN} torus FROZE {ext_fz} cell-steps OUTSIDE the "
        f"horizon (interior, causally disconnected: {fz_h}; exterior first-order "
        f"fallbacks: {ext_fb}) -- a physical cell that no flux can update "
        "admissibly is a breakdown the projection is supposed to preclude"
    )
