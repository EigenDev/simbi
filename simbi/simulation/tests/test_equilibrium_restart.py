# =============================================================================
# test_equilibrium_restart.py
#
# a restart must continue against the same stationary target it started with.
#
# the target is not a field: it leaves no trace in the checkpoint's data, so a run resumed
# against a different one subtracts a different imbalance at every stage and holds a
# different state exactly, while every plot looks the way it should. the ordinary
# immutable-field merge cannot catch it, because the target is a computed expression graph
# rather than a model field — hence a check of its own.
# =============================================================================
import json

import pytest

import simbi.expression as expr
from simbi.simulation.checkpoint import _assert_same_equilibrium_target
from simbi.simulation.problem import ConfigError
from simbi.types.input import Metadata


def atmosphere_payload(gm: float = 100.0) -> dict:
    graph = expr.ExprGraph()
    radius = expr.variable("x1", graph) + expr.constant(1.0, graph)
    potential = -expr.constant(gm, graph) / radius
    equilibrium = expr.isentropic_atmosphere(
        potential, gamma=5.0 / 3.0, k_entropy=1.0, dim=1
    )
    return graph.compile(equilibrium.primitives).serialize_equilibrium(dim=1)


class FakeProblem:
    def __init__(self, payload: dict | None):
        self.equilibrium_expressions = payload or {}


def metadata_with(
    target: dict | None,
    solver: str = "hllc",
    wb_reconstruction: bool | None = None,
) -> Metadata:
    # only the field under test matters; the rest carry the dataclass defaults.
    return Metadata(
        solver=solver,
        wb_reconstruction=wb_reconstruction,
        time=0.0,
        dt=0.0,
        dlogt=0.0,
        tend=1.0,
        iteration=0,
        checkpoint_index=0,
        gamma=5.0 / 3.0,
        cfl=0.4,
        plm_theta=1.5,
        viscosity=0.0,
        resistivity=0.0,
        dimensions=1,
        coord_system="cartesian",
        halo_radius=2,
        is_mhd=False,
        is_relativistic=False,
        regime="newtonian",
        reconstruction="plm",
        timestepping="rk2",
        equilibrium_target=json.dumps(target) if target else "",
    )


def test_the_same_target_resumes() -> None:
    payload = atmosphere_payload()
    _assert_same_equilibrium_target(FakeProblem(payload), metadata_with(payload))


def test_a_run_with_no_target_resumes() -> None:
    # the ordinary case: no target declared, nothing recorded, nothing to compare.
    _assert_same_equilibrium_target(FakeProblem(None), metadata_with(None))


def test_a_changed_target_is_refused() -> None:
    # a different GM is the mistake this exists for: same shape, same node count, same
    # everything a glance would check, and a different atmosphere held exactly.
    with pytest.raises(ConfigError, match="differs from the one this checkpoint"):
        _assert_same_equilibrium_target(
            FakeProblem(atmosphere_payload(gm=50.0)),
            metadata_with(atmosphere_payload(gm=100.0)),
        )


def test_dropping_the_target_on_restart_is_refused() -> None:
    with pytest.raises(ConfigError, match="now declares none"):
        _assert_same_equilibrium_target(
            FakeProblem(None), metadata_with(atmosphere_payload())
        )


def test_adding_a_target_on_restart_is_refused() -> None:
    with pytest.raises(ConfigError, match="written without one"):
        _assert_same_equilibrium_target(
            FakeProblem(atmosphere_payload()), metadata_with(None)
        )


def test_a_retired_acoustic_ramp_checkpoint_is_refused() -> None:
    # the acoustic-speed ramp family scaled the signal speeds, which damps the pressure jump
    # along with the velocity jump; hllc_plus rescales the velocity jumps instead. continuing
    # a series across that change would swap the numerics mid-run under an unchanged config,
    # and nothing downstream of the checkpoint could tell.
    for retired in ("hllc_lm", "hllc-lm", "hllc_acoustic"):
        with pytest.raises(ConfigError, match="retired"):
            _assert_same_equilibrium_target(
                FakeProblem(None),
                metadata_with(None, solver=retired, wb_reconstruction=None),
            )


def test_the_refusal_reads_the_solver_not_the_balance_attribute() -> None:
    # the solver name alone decides. a retired-scheme file that happens to record a
    # reconstruction balance is still a retired-scheme file, and a surviving solver resumes
    # whether or not the attribute is present -- refusing on its absence would strand every
    # archived series written before the attribute entered the format.
    for wb in (True, False, None):
        with pytest.raises(ConfigError, match="retired"):
            _assert_same_equilibrium_target(
                FakeProblem(None),
                metadata_with(None, solver="hllc_lm", wb_reconstruction=wb),
            )
        _assert_same_equilibrium_target(
            FakeProblem(None),
            metadata_with(None, solver="hllc_plus", wb_reconstruction=wb),
        )


def test_other_solvers_never_trip_the_scheme_change_guard() -> None:
    # only the retired family changed; an old checkpoint from any surviving solver must
    # resume untouched.
    for solver in ("hllc", "hlle", "hlld"):
        _assert_same_equilibrium_target(
            FakeProblem(None),
            metadata_with(None, solver=solver, wb_reconstruction=None),
        )


def test_the_comparison_is_structural_not_textual() -> None:
    # two json writers may order keys differently or space them differently; that is not a
    # physics change and must not read as one.
    payload = atmosphere_payload()
    reordered = dict(reversed(list(payload.items())))
    assert list(reordered) != list(payload)
    _assert_same_equilibrium_target(FakeProblem(payload), metadata_with(reordered))
