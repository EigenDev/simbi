# =============================================================================
# test_schwarzschild_bondi_transient.py
#
# regression for the schwarzschild radial-momentum densitization. the stored
# radial momentum is the ORTHONORMAL S_rhat = rho h W^2 V_rhat, but the valencia
# conserved momentum is the COVARIANT S_r = S_rhat / alpha; substituting into
# d_t S_r = -alpha div(F) + alpha S (font 2008, static schwarzschild) gives
# d_t S_rhat = -alpha^2 div_flat(F) + alpha^2 S — TWO lapse factors on the radial
# momentum flux divergence AND its geometric+gravity source. the earlier code
# applied only ONE (the leading densitization lapse), leaving the inward gravity
# source too strong by 1/alpha (1.73x at r = 3M); the gas over-accelerated,
# pumped excess kinetic energy, and the recovered pressure p = (tau - KE)/3 was
# driven negative near the inner boundary, collapsing the wave speed at t ~ 2.34.
#
# the steady michel state cannot detect this (d_t = 0 -> div = S regardless of the
# lapse power), so the check MUST be transient: a uniform-at-rest gas developing
# transonic bondi accretion. with the correct alpha^2 the density RISES at the
# inner boundary (gas piles up / accretes) and the pressure stays strictly
# positive with NO floor. requires the built cpu_ext backend; skipped otherwise.
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

# the old crash was at t ~ 2.34; run well past it so the inner-boundary density
# rise is unambiguous (the bug DEPLETED it to ~0.945 before crashing; the fix
# builds it up above ambient). resolution is irrelevant to the defect (it is a
# source-term densitization error, not a resolution artifact), so keep it small.
_END_TIME = 10.0
_RESOLUTION = 128
_RHO_AMBIENT = 1.0
_P_AMBIENT = 1.0e-2


def _bondi_problem(data_dir: str):
    from simbi_configs.examples.gr_bondi import GrBondi

    p = GrBondi.from_cli(["--resolution", str(_RESOLUTION)])
    p.end_time = _END_TIME
    p.data_directory = data_dir
    p.checkpoint_interval = _END_TIME  # initial + final only
    return p


def _read_interior(chkpt_path: str):
    with h5py.File(chkpt_path, "r") as h:
        part = h["level_0/partition_0"]
        lo = int(part["owned_start"][0])
        fin = int(part["owned_fin"][0])
        prims = part["hydro/primitives"]
        rho = prims["rho"][lo:fin]
        pre = prims["pre"][lo:fin]
    return rho, pre


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

        rho, pre = _read_interior(finals[0])

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
