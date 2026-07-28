# =============================================================================
# test_dispatch_guard.py
#
# C4 / M9 regression: the dims/coords dispatch macros must FAIL LOUD on a
# non-minkowski spacetime that has no baked GR kernel, never silently fall
# through to a flat `(dims, coords)` arm and run on a Minkowski metric (wrong
# physics, zero warning — the empirically-confirmed C4 bug).
#
# strategy: take a baked GR config (GrMichel: 1D spherical Schwarzschild) and
# flip its spacetime to one that is UNBAKED for that (dims, coords). the IC stays
# valid (still spherical, r-based), so the run reaches the dispatch and must raise
# the dispatch guard.
#
# ALSO covers the sibling loud-rejection guard the review flagged as test-asserted
# nowhere: a curved spacetime on a NON-relativistic regime (the non-relativistic
# kernel rows are never baked with a spacetime slug) must be rejected before
# dispatch — a refactor dropping that guard would silently run flat gravity-free
# physics on a config that asked for GR.
# =============================================================================
import pytest

from simbi.simulation import runner
from simbi.types import Spacetime

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

from simbi_configs.examples.grhd.gr_michel import GrMichel
from simbi_configs.examples.newtonian.sod import SodProblem


def test_horizon_penetrating_spacetime_names_identify_the_solution() -> None:
    assert Spacetime.SCHWARZSCHILD_KS.value == "schwarzschild_ks"
    assert Spacetime.KERR_KS.value == "kerr_ks"
    assert "KERR_SCHILD" not in Spacetime.__members__
    assert "KERR" not in Spacetime.__members__


class _UnbakedKerrMichel(GrMichel):
    # (1, spherical, kerr) is NOT a baked GR-hydro arm (only 2D spherical kerr is);
    # the C4 guard must reject it before it can reach the flat (1, spherical) arm.
    spacetime: Spacetime = Spacetime.KERR_KS


@needs_backend
def test_dispatch_rejects_unbaked_gr_spacetime():
    p = _UnbakedKerrMichel.from_cli(["--resolution", "16"])
    with pytest.raises(Exception, match="no baked GR|refusing to run silently|Minkowski"):
        runner.run(p, compute_mode="cpu", max_steps=400)


@needs_backend
def test_baked_gr_spacetime_still_dispatches():
    # regression: the guard must NOT reject a genuinely baked GR combo. the default
    # GrMichel (1D spherical Schwarzschild) IS baked; run a couple of steps and assert
    # it does not raise the dispatch guard (it gets PAST it into real evolution).
    p = GrMichel.from_cli(["--resolution", "16", "--end-time", "0.001"])
    # any exception here must NOT be the C4 guard (baked combos dispatch fine).
    try:
        runner.run(p, compute_mode="cpu", max_steps=400)
    except Exception as e:  # pragma: no cover - only trips on a real regression
        assert "no baked GR" not in str(e) and "refusing to run silently" not in str(e), (
            f"the C4 guard wrongly rejected a BAKED GR combo: {e}"
        )


class _NewtonianOnSchwarzschild(SodProblem):
    # a NON-relativistic regime (newtonian) with a curved spacetime: the non-relativistic kernel rows
    # are never baked with a spacetime slug, so the regime-vs-spacetime guard must reject this before
    # dispatch. the mass is positive so the earlier
    # GR-parameter gate (a curved spacetime with M = 0 is rejected first) does not mask the
    # regime-vs-spacetime rejection this test exercises.
    spacetime: Spacetime = Spacetime.SCHWARZSCHILD
    schwarzschild_mass: float = 1.0


@needs_backend
def test_dispatch_rejects_gr_spacetime_on_nonrelativistic_regime():
    p = _NewtonianOnSchwarzschild.from_cli(["--resolution", "16"])
    with pytest.raises(Exception, match="requires a relativistic regime"):
        runner.run(p, compute_mode="cpu", max_steps=400)
