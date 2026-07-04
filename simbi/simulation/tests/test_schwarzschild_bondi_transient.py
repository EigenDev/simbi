# =============================================================================
# test_schwarzschild_bondi_transient.py
#
# transient positivity + accretion regression for the Valencia GRHD scheme on a static
# schwarzschild background. the conserved momentum is the COVARIANT S_r = rho h W^2 gamma_rr v^r,
# recovered/fluxed with the spatial metric and densitized by a single uniform lapse alpha on the
# flux divergence AND the geodesic source (d_t S_r = -alpha div(F) + alpha S, font 2008). an error
# in the inward gravity/densitization balance over-accelerates the gas, pumps excess kinetic energy,
# and drives the recovered pressure p = (tau - KE)/3 negative near the inner boundary, collapsing the
# wave speed (an earlier orthonormal-storage bug crashed at t ~ 2.34).
#
# the steady michel state cannot detect a densitization-power error (d_t = 0 -> div = S regardless),
# so the check MUST be transient: a uniform-at-rest gas developing transonic bondi accretion. correct
# inward densitization makes the density RISE at the inner boundary (gas piles up / accretes) and the
# pressure stay strictly positive with NO floor. requires the built cpu_ext backend; skipped otherwise.
# =============================================================================
import glob
import os
import tempfile

import h5py
import pytest

from simbi.simulation import runner

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

# the densitization crash was at t ~ 2.34; the riemann-fan crash at t ~ 11.8. run to 15 so
# the inner flow crosses |V| = alpha (t ~ 10-12), where the banyuls-font discriminant
# gamma^{nn}(1 - v^2 cs^2) - v^n v^n (1 - cs^2) goes negative if the physical velocity
# sqrt(gamma_nn) v^n is conflated into the contravariant v^n slot (NaN fan, wave-speed
# collapse). resolution is irrelevant to either defect; keep it small.
_END_TIME = 15.0
_RESOLUTION = 128
_RHO_AMBIENT = 1.0
_P_AMBIENT = 1.0e-2


def _bondi_problem(data_dir: str):
    from simbi_configs.examples.grhd.gr_bondi import GrBondi

    p = GrBondi.from_cli(["--resolution", str(_RESOLUTION)])
    p.end_time = _END_TIME
    p.data_directory = data_dir
    p.checkpoint_interval = _END_TIME  # initial + final only
    return p


def _read_interior(chkpt_path: str):
    # the stored arrays carry the ghost cells; the interior is the central
    # `_RESOLUTION` entries (owned_start/owned_fin are interior-relative and do
    # NOT index the ghost-padded arrays).
    with h5py.File(chkpt_path, "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - _RESOLUTION) // 2
        sl = slice(halo, halo + _RESOLUTION)
        rho = prims["rho"][sl]
        pre = prims["pre"][sl]
        vel = prims["v1"][sl]
    return rho, pre, vel


@needs_backend
def test_bondi_transient_survives_and_pressure_stays_positive() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        runner.run(_bondi_problem(d), compute_mode="cpu")

        # a clean completion writes <res>.chkpt.final*.h5; a crash writes only
        # <res>.chkpt.crashed.h5. before the fix the run never reached t = 6.
        finals = glob.glob(os.path.join(d, "*.chkpt.final*.h5"))
        assert finals, (
            "schwarzschild bondi transient crashed before completion "
            "(radial-momentum densitization regression)"
        )

        rho, pre, vel = _read_interior(finals[0])

        # NO FLOOR: the pressure must stay strictly positive on its own. the old
        # bug drove it negative near the inner boundary.
        assert pre.min() > 0.0, f"pressure went non-positive: min = {pre.min():.3e}"

        # the physical signature of correct inward densitization: gas accretes and
        # the density RISES at the inner boundary above the ambient value. the old
        # bug did the opposite — it DEPLETED the inner density to ~0.945 before the
        # negative-pressure crash — so "above ambient" cleanly separates fix from bug.
        assert rho[0] > 1.1 * _RHO_AMBIENT, (
            f"density did not rise at the inner boundary: rho_inner = {rho[0]:.3f} "
            f"(ambient {_RHO_AMBIENT})"
        )

        # the completion assertion only pins the riemann-fan discriminant if the flow
        # actually entered the |V| > alpha regime at the inner boundary (below that,
        # the conflated and correct banyuls-font forms are both real). verify it:
        # V = v^r / alpha on schwarzschild, at the innermost log-spaced cell centroid.
        r_inner = 3.0 * (100.0 / 3.0) ** (0.5 / _RESOLUTION)
        alpha_inner = (1.0 - 2.0 / r_inner) ** 0.5
        v_phys = abs(vel[0]) / alpha_inner
        assert v_phys > alpha_inner, (
            f"inner flow never crossed |V| = alpha (|V| = {v_phys:.3f}, "
            f"alpha = {alpha_inner:.3f}); the fan regression is not exercised"
        )
