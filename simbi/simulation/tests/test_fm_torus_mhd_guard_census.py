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
MAX_STEPS = 2000


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
    runner.run(p, compute_mode="cpu", max_steps=MAX_STEPS)

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

    assert t_final > END_TIME - 1e-3, (
        f"the magnetized a={SPIN} torus stalled at t = {t_final:.4f} < {END_TIME} "
        f"within {MAX_STEPS} steps -- a dt collapse, which is what the "
        "source-admissibility rate exists to prevent and what an over-aggressive "
        "step exposes"
    )

    assert ext_fz == 0, (
        f"the magnetized a={SPIN} torus FROZE {ext_fz} cell-steps OUTSIDE the "
        f"horizon (interior, causally disconnected: {fz_h}; exterior first-order "
        f"fallbacks: {ext_fb}) -- a physical cell that no flux can update "
        "admissibly is a breakdown the projection is supposed to preclude"
    )
