# =============================================================================
# test_equilibrium_backend_wire.py
#
# the declared stationary target reaches the backend and acts there.
#
# every other test of this feature checks one layer: the python serializer emits the right
# schema, the rust hierarchy holds a target it is handed directly. between them sits the
# exec-dict crossing and the build-site call, and a break there is silent — the run works
# perfectly, simply without well-balancing, and the only symptom is an atmosphere that drifts
# as it always did.
#
# so the gate runs the real backend and demands the target change something. it declares a
# target that balances half the gravity the run applies, which the backend's refinement check
# must reject; a wire that never delivered the target would run happily to completion. the
# companion case, the correct target on the same config, must run — otherwise the rejection
# proves only that the config is broken.
# =============================================================================
import tempfile

import pytest
from pydantic import computed_field

import simbi.expression as expr
from simbi.simulation import runner
from simbi.types.typing import ExpressionDict

from simbi_configs.examples.newtonian.refined_atmosphere import RefinedAtmosphere

_MAX_STEPS = 5


class HalfGravityTarget(RefinedAtmosphere):
    """the same run, with a target built for half the gravity the source applies.

    it is a perfectly good hydrostatic atmosphere — for a different problem. nothing about
    the profile looks wrong: smooth, positive, monotone, and wrong by a constant in one term.
    """

    @computed_field
    @property
    def equilibrium_expressions(self) -> ExpressionDict:
        graph = expr.ExprGraph()
        radius = expr.variable("x1", graph) + expr.constant(self.offset, graph)
        halved = -expr.constant(0.5 * self.gm, graph) / radius
        atmosphere = expr.isentropic_atmosphere(
            halved,
            gamma=self.adiabatic_index,
            k_entropy=self.entropy,
            dim=1,
            reference_density=1.0,
            reference_point=[self.bounds[0][1]],
        )
        return graph.compile(atmosphere.primitives).serialize_equilibrium(dim=1)


def _run(problem) -> None:
    problem.data_directory = tempfile.mkdtemp() + "/"
    problem.checkpoint_interval = 1.0e30
    runner.run(problem, compute_mode="cpu", max_steps=_MAX_STEPS)


_SUSPECT = "may not be a steady state"


@pytest.mark.simulation
def test_the_correct_target_runs(capfd) -> None:
    # the control, in both directions: without it the report below could equally mean the config
    # is broken, and a diagnostic that fired on every target -- correct or not -- would carry no
    # information at all.
    _run(RefinedAtmosphere())
    assert _SUSPECT not in capfd.readouterr().err, (
        "the backend flagged the atmosphere that IS in balance against the gravity the run "
        "applies; a diagnostic that fires on a correct target cannot distinguish anything"
    )


@pytest.mark.simulation
def test_a_target_that_is_not_a_steady_state_is_reported_by_the_backend(capfd) -> None:
    # the backend measures the target's imbalance on two levels and reports -- never refuses --
    # a component whose imbalance fails to shrink when the cell width halves. truncation error
    # falls by at least 2 under refinement; a continuum residual does not.
    #
    # the report is advisory because the measurement cannot be both sound and complete: a target
    # that is genuinely steady but sharply stratified carries its imbalance exactly where the grid
    # fails to resolve it, so refusing on that evidence rejects correct equilibria for being
    # steep. the run therefore completes, and the diagnostic firing is itself the proof that the
    # target crossed the exec dict and was applied at the hierarchy build site.
    _run(HalfGravityTarget())
    assert _SUSPECT in capfd.readouterr().err, (
        "a profile balancing half the gravity the run applies drew no warning; the stationarity "
        "diagnostic is not reaching the declared target"
    )
