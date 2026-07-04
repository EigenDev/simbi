# =============================================================================
# test_schwarzschild_michel_steady.py
#
# the schwarzschild GRHD accuracy gate: the exact michel (1972) transonic accretion
# solution is stationary, so a correct valencia scheme (covariant momentum, uniform
# lapse densitization, geodesic sources, banyuls-font fan) must HOLD it — the
# evolved profile stays on the analytic one at truncation level, and the residual
# SHRINKS under grid refinement. an equation-level error (a misplaced lapse power, a
# wrong source term, a mis-sampled metric) shows up as a resolution-independent
# drift and fails the convergence assertion; a stable-but-inaccurate scheme cannot
# pass. complements the gr_bondi transient tests, which check development and
# positivity but have no exact reference.
#
# the profile is transonic on the [3, 100] domain (sonic radius ~ 22.7 M at the
# default ambient state): the inner-boundary exit is supersonic with |V| > alpha
# (the riemann-fan regime near the horizon), the outer-boundary inflow subsonic.
# both boundaries are zero-gradient ghosts, a first-order approximation of the
# analytic continuation, so the gate measures the interior away from them; the
# boundary cells are separately bounded and must also improve with resolution.
# requires the built cpu_ext backend; skipped otherwise.
# =============================================================================
import glob
import math
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

# several sound crossings of the inner region; long enough that a densitization or
# source imbalance (growth rate ~ the local dynamical rate) would dominate the
# truncation residual, short enough that the outer zero-gradient boundary error
# (inward propagation speed ~ c_inf - |v|) stays confined to the excluded cells.
_END_TIME = 10.0

# interior window for the truncation-level comparison: the innermost cells feel the
# first-order inner ghost, the outermost the first-order outer ghost.
_SKIP_INNER = 4
_SKIP_OUTER = 8

# measured interior L1 relative rho residuals at t = 10: 5.0e-5 (128 zones),
# 1.6e-5 (256 zones), ratio 0.33. tolerances carry ~3x margin.
_L1_TOL_128 = 1.5e-4
_L1_TOL_256 = 5.0e-5
_CONVERGENCE_RATIO = 0.55
# the boundary cells are first-order (zero-gradient ghosts): measured max relative
# rho error 2.0e-2 (128) at the innermost cell, halving with resolution.
_MAX_TOL_128 = 6.0e-2


def _michel_problem(res: int, data_dir: str):
    from simbi_configs.examples.grhd.gr_michel import GrMichel

    p = GrMichel.from_cli(["--resolution", str(res)])
    p.end_time = _END_TIME
    p.data_directory = data_dir
    p.checkpoint_interval = _END_TIME  # initial + final only
    return p


def _read_interior(chkpt_path: str, res: int):
    """interior primitives, halo excluded (the stored arrays carry the ghost cells;
    the interior is the central `res` entries)."""
    with h5py.File(chkpt_path, "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - res) // 2
        sl = slice(halo, halo + res)
        return prims["rho"][sl], prims["pre"][sl], prims["v1"][sl]


def test_michel_oracle_satisfies_the_flow_invariants() -> None:
    from simbi_configs.examples.grhd.gr_michel import MichelSolution

    sol = MichelSolution(mass=1.0, gamma=4.0 / 3.0, rho_inf=1.0, p_inf=1.0e-2)

    # the sonic point sits inside the gate domain and obeys the critical relations.
    assert 3.0 < sol.r_sonic < 100.0
    assert math.isclose(
        sol.u_sonic**2, sol.mass / (2.0 * sol.r_sonic), rel_tol=1e-12
    )

    for r in [3.0, 5.0, 10.0, sol.r_sonic, 40.0, 100.0]:
        rho, v1, pre = sol.primitive(r)
        u = sol.proper_velocity(r)
        f = 1.0 - 2.0 * sol.mass / r
        h = 1.0 + sol.gamma / (sol.gamma - 1.0) * pre / rho
        # bernoulli invariant h^2 (f + u^2) = h_inf^2, machine-exact along the flow.
        assert abs(h * h * (f + u * u) / sol.h_inf_sq - 1.0) < 1e-12, f"r={r}"
        # baryon-flux invariant r^2 rho u = jm.
        assert abs(r * r * rho * u / sol.jm - 1.0) < 1e-12, f"r={r}"
        # physical inflow: subluminal, negative, supersonic only inside r_sonic.
        big_v = abs(v1) / math.sqrt(f)
        a = math.sqrt(sol.gamma * pre / (rho * h))
        assert v1 < 0.0 and big_v < 1.0
        assert (big_v > a) == (r < sol.r_sonic) or math.isclose(
            big_v, a, rel_tol=1e-9
        )

    # sound speed equals the flow speed exactly at the sonic radius.
    rho, v1, pre = sol.primitive(sol.r_sonic)
    h = 1.0 + sol.gamma / (sol.gamma - 1.0) * pre / rho
    a_s = math.sqrt(sol.gamma * pre / (rho * h))
    f_s = 1.0 - 2.0 * sol.mass / sol.r_sonic
    assert math.isclose(abs(v1) / math.sqrt(f_s), a_s, rel_tol=1e-10)

    # asymptotic state: the profile relaxes to the ambient density far out.
    assert abs(sol.primitive(1.0e6)[0] - 1.0) < 1e-3


def _held_profile_errors(res: int):
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        p = _michel_problem(res, d)
        runner.run(p, compute_mode="cpu")

        finals = glob.glob(os.path.join(d, "*.chkpt.final*.h5"))
        assert finals, f"michel steady run crashed at {res} zones"
        rho, pre, v1 = _read_interior(finals[0], res)

        sol = p.michel_solution()
        ref = np.array([sol.primitive(r) for r in p.cell_centroids()])

    assert pre.min() > 0.0, f"pressure went non-positive: {pre.min():.3e}"
    e_rho = np.abs(rho / ref[:, 0] - 1.0)
    e_v1 = np.abs(v1 / ref[:, 1] - 1.0)
    interior = slice(_SKIP_INNER, res - _SKIP_OUTER)
    return e_rho[interior].mean(), e_v1[interior].mean(), e_rho.max()


@needs_backend
def test_michel_profile_is_held_at_truncation_level_and_converges() -> None:
    l1_128, l1v_128, emax_128 = _held_profile_errors(128)
    l1_256, l1v_256, emax_256 = _held_profile_errors(256)

    # truncation-level hold of the exact stationary solution.
    assert l1_128 < _L1_TOL_128, f"128-zone interior L1 rho residual {l1_128:.3e}"
    assert l1_256 < _L1_TOL_256, f"256-zone interior L1 rho residual {l1_256:.3e}"
    assert emax_128 < _MAX_TOL_128, f"128-zone max rho residual {emax_128:.3e}"

    # the residual must SHRINK under refinement — a resolution-independent floor
    # means an equation-level error (lapse power, source term, metric sampling),
    # not truncation.
    assert l1_256 < _CONVERGENCE_RATIO * l1_128, (
        f"rho residual does not converge: {l1_128:.3e} -> {l1_256:.3e}"
    )
    assert l1v_256 < _CONVERGENCE_RATIO * l1v_128, (
        f"velocity residual does not converge: {l1v_128:.3e} -> {l1v_256:.3e}"
    )
    assert emax_256 < 0.8 * emax_128, (
        f"boundary-cell residual does not improve: {emax_128:.3e} -> {emax_256:.3e}"
    )
