# =============================================================================
# test_equilibrium_backend_wire.py
#
# the declared stationary target reaches the BACKEND and acts there.
#
# every other test of this feature checks one layer: the python serializer emits the right
# schema, the rust hierarchy holds a target it is handed directly. between them sits the
# exec-dict crossing and the build-site call, and a break there is silent — the run works
# perfectly, simply without well-balancing, and the only symptom is an atmosphere that drifts
# as it always did.
#
# so the gate runs the real backend and demands the target CHANGE something. it declares a
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


@pytest.mark.simulation
def test_the_correct_target_runs() -> None:
    # the control. without this the rejection below could equally mean the config is broken.
    _run(RefinedAtmosphere())


@pytest.mark.simulation
def test_a_target_that_is_not_a_steady_state_is_rejected_by_the_backend() -> None:
    # the backend measures the target's imbalance on two levels and refuses it when it fails
    # to shrink under refinement. reaching that check at all proves the target crossed the
    # exec dict and was applied at the hierarchy build site.
    with pytest.raises(BaseException, match="not a steady state"):
        _run(HalfGravityTarget())
