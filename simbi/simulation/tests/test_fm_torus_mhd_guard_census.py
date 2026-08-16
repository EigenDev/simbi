# =============================================================================
# test_fm_torus_mhd_guard_census.py
#
# the magnetized spinning fishbone-moncrief torus on the 3d cartesian kerr
# chart: the guard-activation gate for the magnetized admissibility projection.
#
# the unmagnetized torus gates the hydrodynamic projection, whose admissible set
# is a second-order cone -- necessary and sufficient, so every cell is
# recoverable by blending toward the stage-input anchor. the magnetized set is
# strictly smaller: a state can satisfy the B-free cone
# q = E - sqrt(D^2 + |S|^2) > 0 and still admit no physical primitive, because
# the magnetic energy leaves no positive gas pressure to recover. the sufficient
# condition adds psi > 0 (Wu & Tang, arXiv:1709.05838, theorem 2.1). this run is
# the stress case for that distinction -- a weak poloidal loop threading the
# spinning torus, where the near-horizon infall is both magnetized and stiff.
#
# the gates are state invariants, measured over a few tens of steps rather than a
# march to an end time:
# - the timestep stays within reach of the light-crossing step. a mis-masked
#   source rate or a metric evaluated where it is singular drives dt orders below
#   the hyperbolic limit, and it does so from the first steps -- the collapse this
#   detects is a property of the discretization, not something that develops.
# - zero freezes outside the horizon. a freeze is a cell no flux could update
#   admissibly, and outside the horizon that is a physical breakdown. the
#   interior of r_+ is causally disconnected fiction and is exempt, but its
#   count is reported, because the magnetic field is constrained-transport
#   evolved and cannot be blended -- so the projection searches only the slice
#   B = B_candidate, and an anchor admissible with its own field need not be
#   admissible in that slice. those cells are genuinely unrecoverable by
#   blending and fall through to the freeze tier by construction, not by defect.
# - every cell stays in the admissible set, and the field survives (a magnetized
#   gate on a field-free state is vacuous).
#
# what this no longer asserts, deliberately: that the torus survives to t = 5.
# that is a soak -- it answers "does this stay healthy for a long time", which is
# a different question from "is the discretization sound", costs three orders of
# magnitude more, and cannot be run in a development loop. it belongs to a
# campaign launched from simbi_configs/examples/grmhd, not to a test suite.
#
# usage:
#   pytest simbi/simulation/tests/test_fm_torus_mhd_guard_census.py -s
#   (-s to see the reported census counts)
# =============================================================================
import tempfile
from pathlib import Path

import pytest

from simbi.simulation import runner
from simbi.simulation.tests.invariants import (
    assert_field_survived,
    assert_no_exterior_freezes,
    assert_state_is_admissible,
    assert_timestep_is_not_collapsed,
    run_and_measure,
)

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

from simbi_configs.examples.grmhd.gr_fishbone_moncrief_mhd_cartesian import (
    GrFishboneMoncriefMhdCartesian,
)

RES = 48
SPIN = 0.9

# enough steps for the near-horizon infall to engage the stiff magnetized recovery --
# the regime where the projection is actually exercised -- and few enough to run in
# seconds. the invariants are properties of the state at whatever step they are
# read, so this number sets cost, not sensitivity.
STEPS = 60


@needs_backend
def test_magnetized_fm_torus_holds_its_invariants_near_the_horizon() -> None:
    d = tempfile.mkdtemp() + "/"
    # the thick-torus pair: at a = 0.9 the compact (r_in = 8, kappa = 1.01)
    # equilibrium degenerates to a sub-cell sliver, so the resolution is part of the
    # setup rather than a cost knob.
    p = GrFishboneMoncriefMhdCartesian(
        kerr_spin=SPIN,
        r_in=6.0,
        kappa=1.15,
        resolution=(RES, RES, RES),
        end_time=1.0e30,
        checkpoint_interval=1.0e30,
        data_directory=Path(d),
    )
    # the physical cell width on this cartesian chart; the light-crossing step is
    # cfl * dx with c = 1.
    dx = 2.0 * p.half_width / RES
    health = run_and_measure(
        p, d, steps=STEPS, widths=[dx], backend=_BACKEND, components=3
    )

    fb, fz, fb_h, fz_h = health.guards
    print(
        f"\nmagnetized FM torus (a={SPIN}, {RES}^3) after {health.steps} steps:"
        f"\n  first-order fallbacks : {fb} ({fb - fb_h} exterior, {fb_h} interior)"
        f"\n  freezes               : {fz} ({fz - fz_h} exterior, {fz_h} interior)"
        f"\n  dt / dt_light         : {health.dt_fraction:.3e}"
    )

    label = f"magnetized a={SPIN} torus"
    assert_field_survived(health, label=label)
    assert_state_is_admissible(health, label=label)
    assert_timestep_is_not_collapsed(health, label=label)
    assert_no_exterior_freezes(health, label=label)
