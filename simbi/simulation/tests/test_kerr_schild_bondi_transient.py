# =============================================================================
# test_kerr_schild_bondi_transient.py
#
# the transient positivity + accretion regression for the Valencia GRHD scheme on a
# horizon-penetrating (ingoing kerr-schild) background. the conserved momentum is the covariant
# S_r = rho h W^2 gamma_rr v^r, recovered/fluxed with the spatial metric and densitized on both the
# flux divergence and the geodesic source (font 2008). an error in the inward gravity/densitization
# balance over-accelerates the gas, pumps excess kinetic energy, and drives the recovered pressure
# p = (tau - KE)/3 negative near the hole, collapsing the wave speed.
#
# why transient: a steady state cannot see a densitization-power error at all — at d_t = 0 the
# equation reduces to div(F) = S regardless of what power of the lapse multiplies both sides. only
# a developing flow, a uniform gas at rest accreting into a transonic profile, separates them. this
# is the complement of the michel gates, which measure how exactly a known stationary solution is
# held and are blind to precisely this defect.
#
# why this chart: the flow crosses r = 2M. on the singular chart the inflow becomes
# ultra-relativistic there (V -> 1, W -> infinity) purely as an artifact of the static observer, so
# the inner boundary has to be parked outside the horizon and the gas piles against a wall it
# should cross freely. here the horizon is an ordinary surface, the domain spans it, and the
# excised interior is a one-way absorber.
#
# the banyuls-font wave-speed discriminant is not probed here. that defect — the physical velocity
# sqrt(gamma_nn) v^n conflated into the contravariant v^n slot — is a chart-independent algebra
# error, and it is gated directly in rust (`rhd/wave_speeds.rs`), where the cauchy-schwarz bound
# disc >= gamma^{nn} (1 - v^2)^2 is asserted across the whole parameter space rather than at the
# states one accretion run happens to visit.
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

# long enough for the inflow to develop and reach the horizon from rest: the free-fall time from
# the bondi radius dominates, and the densitization imbalance (growth rate ~ the local dynamical
# rate) has many e-folds to express itself before the gate reads the state.
_END_TIME = 15.0
_RESOLUTION = 128
_RHO_AMBIENT = 1.0
_R_HORIZON = 2.0
_R_EXCISION = 1.4


def _bondi_problem(data_dir: str):
    from simbi_configs.examples.grhd.gr_bondi import GrBondi

    p = GrBondi.from_cli(["--resolution", str(_RESOLUTION)])
    p.end_time = _END_TIME
    p.data_directory = data_dir
    p.checkpoint_interval = _END_TIME  # initial + final only
    return p


def _radii(p) -> np.ndarray:
    """volume-weighted centroids of the log-spaced radial grid."""
    (rmin, rmax) = p.bounds[0]
    q = (rmax / rmin) ** (1.0 / _RESOLUTION)
    rl = rmin * q ** np.arange(_RESOLUTION)
    rh = rl * q
    return 0.75 * (rh**4 - rl**4) / (rh**3 - rl**3)


def _read_interior(chkpt_path: str):
    with h5py.File(chkpt_path, "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - _RESOLUTION) // 2
        sl = slice(halo, halo + _RESOLUTION)
        return prims["rho"][sl], prims["pre"][sl], prims["v1"][sl]


@needs_backend
def test_bondi_transient_crosses_the_horizon_with_positive_pressure() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        p = _bondi_problem(d)
        _BACKEND.reset_guard_census()
        runner.run(p, compute_mode="cpu", max_steps=4000)

        finals = glob.glob(os.path.join(d, "*.chkpt.final*.h5"))
        assert finals, (
            "kerr-schild bondi transient crashed before completion "
            "(radial-momentum densitization regression)"
        )
        rho, pre, vel = _read_interior(finals[0])
        r = _radii(p)

        # the premise: the domain must actually span the horizon and the excision surface, with
        # live cells on both sides. a grid that stopped outside r_+ would test an inner wall, not a
        # horizon, and every through-horizon assertion would be vacuous.
        assert r[0] < _R_EXCISION < _R_HORIZON < r[-1], (
            f"the grid does not span the excision surface and the horizon "
            f"(r = [{r[0]:.3f}, {r[-1]:.3f}], r_exc = {_R_EXCISION}, r_+ = {_R_HORIZON})"
        )
        exterior = r > _R_EXCISION
        assert exterior.sum() > 0.5 * _RESOLUTION, "too few live cells outside the excision surface"

        # no floor: the pressure must stay strictly positive on its own across the live region.
        # an error in the gravity/densitization balance drives it negative near the inner
        # boundary.
        assert pre[exterior].min() > 0.0, (
            f"pressure went non-positive outside the excision surface: "
            f"min = {pre[exterior].min():.3e}"
        )
        assert np.isfinite(vel).all(), "velocity went non-finite"

        # the physical signature of correct inward densitization: gas accretes, so the density
        # rises above ambient as it approaches the hole. a densitization imbalance does the
        # opposite, depleting the inner density before the negative-pressure crash, so the
        # direction of the change discriminates, not its size.
        inner = np.argmax(exterior)  # innermost live cell
        assert rho[inner] > 1.1 * _RHO_AMBIENT, (
            f"density did not rise at the innermost live cell: rho = {rho[inner]:.3f} "
            f"(ambient {_RHO_AMBIENT}) — the gas is not accreting"
        )

        # the flow must actually be inward there: an accreting solution has v^r < 0 through the
        # horizon. a density rise with no inflow would be a static pile-up against a wall.
        assert vel[inner] < 0.0, (
            f"the innermost live cell is not inflowing (v^r = {vel[inner]:.3e}); "
            "the gas is piling up rather than accreting"
        )

        # the horizon is one-way: no limiter may fire outside the excision surface. inside it the
        # state is numerical padding the exterior never sees, so guards there are expected.
        fallback, freeze, fb_h, fz_h = _BACKEND.guard_census()
        assert (freeze - fz_h) == 0, (
            f"{freeze - fz_h} cell-steps FROZE outside the excision surface "
            f"(interior, exempt: {fz_h}); a physical cell no flux can update admissibly is a "
            "breakdown, not a cost"
        )


@needs_backend
def test_excised_interior_stays_frozen_at_the_vacuum_floor() -> None:
    # the excision contract on this chart: cells inside the surface hold the cold vacuum, and the
    # accreting exterior never re-populates them. if the fill were transmissive instead of
    # absorbing, interior gas would leak back out through the excision faces and these cells would
    # carry the exterior's density.
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        p = _bondi_problem(d)
        runner.run(p, compute_mode="cpu", max_steps=4000)
        finals = glob.glob(os.path.join(d, "*.chkpt.final*.h5"))
        assert finals, "kerr-schild bondi run crashed"
        rho, pre, _ = _read_interior(finals[0])
        r = _radii(p)

        excised = r < _R_EXCISION
        assert excised.sum() > 0, (
            f"no cell lies inside the excision surface (innermost r = {r[0]:.3f}, "
            f"r_exc = {_R_EXCISION}); the excision contract is not exercised"
        )
        # the floor is orders below the ambient state, so "far below ambient" separates a frozen
        # vacuum from any leaked exterior gas without pinning the floor's value.
        assert rho[excised].max() < 1.0e-3 * _RHO_AMBIENT, (
            f"excised cells carry gas (max rho = {rho[excised].max():.3e}): the excision fill is "
            "transmitting rather than absorbing"
        )
        assert pre[excised].max() < 1.0e-3 * _RHO_AMBIENT, (
            f"excised cells carry pressure (max p = {pre[excised].max():.3e})"
        )
