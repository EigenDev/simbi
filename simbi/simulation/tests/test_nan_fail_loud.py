# =============================================================================
# test_nan_fail_loud.py
#
# the fail-loud gate for STATE-INDEPENDENT wave-speed maps. the driver's only
# per-step blow-up detector is the dt chain ("NaN state -> NaN wave speeds ->
# NaN/zero dt -> halt"): a pure-geometry CFL map (the GR light-cone bound) breaks
# that chain — a NaN'd run would march to t_final and write garbage checkpoints
# as a "successful" result. the guard: the map probes the conserved state per
# cell and forces lambda -> +inf when it is non-finite (dt collapses to zero and
# the crash heuristics halt; a NaN lambda would be silently DROPPED by the
# max-reduce, so the poison must be +inf).
#
# the poison here: a kerr rotating-equilibrium (light-cone CFL map) with one
# driven-boundary expression producing NaN (sqrt of a negative constant) — the
# ghost state is NaN from the first fill, the first step's boundary flux pulls it
# into the interior, and the SECOND step's cfl must halt the run instead of
# completing. requires the built cpu_ext backend; skipped otherwise.
# =============================================================================
import glob
import os
import tempfile

import pytest

from simbi.simulation import runner

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)


def _poisoned_prescription() -> dict:
    import simbi.expression as expr

    g = expr.ExprGraph()
    r = expr.variable("r", g)
    _ = expr.variable("theta", g)
    nan = expr.sqrt(expr.constant(-1.0, g)) + r * 0.0
    one = expr.constant(1.0, g)
    zero = expr.constant(0.0, g)
    compiled = g.compile([nan, zero, zero, zero, one])
    return compiled.serialize_boundary(dim=3)


@needs_backend
def test_poisoned_state_halts_under_a_state_independent_cfl_map() -> None:
    from simbi_configs.examples.gr_rotating_equilibrium import GrRotatingEquilibrium

    class Poisoned(GrRotatingEquilibrium):
        """the kerr rotating equilibrium with a NaN-producing inner ghost band."""

        @property
        def bx1_inner_expressions(self) -> dict:
            return _poisoned_prescription()

    d = tempfile.mkdtemp() + "/"
    p = Poisoned.from_cli(["--nr", "64", "--npolar", "16", "--kerr-spin", "0.9"])
    p.end_time = 5.0
    p.data_directory = d
    p.checkpoint_interval = 5.0
    try:
        runner.run(p, compute_mode="cpu")
    except Exception:
        return  # a raised error is an acceptable loud halt
    crashed = glob.glob(os.path.join(d, "*crashed*.h5"))
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert crashed or not finals, (
        "a NaN-poisoned run completed as a success under the light-cone CFL map — "
        "the state-finiteness guard is not reaching dt"
    )
