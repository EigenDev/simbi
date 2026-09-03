# =============================================================================
# test_anchor_ab.py
#
# the read-only projection-anchor A/B comparison harness, on synthetic fixtures:
# - identical arms produce zero cross-arm deltas
# - a signed cancellation with nonzero gross injection reports the two budgets
#   separately, never conflated
# - a rejected attempt keeps attempted and accepted receipts distinct
# - a zero initial mass/energy leaves the normalized metric out, never divides
# - non-finite input is rejected
# - a configuration mismatch between the arms is rejected
# - two resolutions report "insufficient convergence evidence"
# - a known three-resolution power law recovers its order
# =============================================================================

import copy
from typing import Any

import pytest

from simbi.analysis.anchor_ab import (
    AnchorComparisonError,
    RunRecord,
    compare_pair,
    compare_suite,
    to_dict,
    to_json,
    to_markdown,
)


def _receipts(
    passes: int,
    fired: int,
    cells: int,
    min_theta: float,
    mass: tuple[float, float],
    seg: tuple[float, float],
    raise_: tuple[float, float],
    weight: float = 1.0,
) -> dict[str, Any]:
    intervention = {
        "mass": list(mass),
        "energy_segment": list(seg),
        "energy_raise": list(raise_),
    }
    injected = {
        "mass": [mass[0] * weight, mass[1] * weight],
        "energy_segment": [seg[0] * weight, seg[1] * weight],
        "energy_raise": [raise_[0] * weight, raise_[1] * weight],
    }
    return {
        "passes": passes,
        "passes_fired": fired,
        "projected_cells": cells,
        "min_theta": min_theta,
        "intervention": intervention,
        "injected": injected,
    }


def _record(
    convention: str,
    resolution: int,
    *,
    accepted: dict[str, Any] | None = None,
    attempted: dict[str, Any] | None = None,
    first: dict[str, Any] | None = None,
    conserved_initial: dict[str, Any] | None = None,
    conserved_final: dict[str, Any] | None = None,
    observables: dict[str, Any] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    acc = accepted or _receipts(
        2, 2, 5, 0.4, (1.0, 1.0), (0.5, 0.5), (-0.2, 0.2)
    )
    att = attempted or acc
    fired_attempted = att["passes_fired"] > 0
    fired_accepted = acc["passes_fired"] > 0
    default_first = {
        "attempted_first_pass": 1 if fired_attempted else -1,
        "accepted_first_pass": 1 if fired_accepted else -1,
        "accepted_first_time": 0.25 if fired_accepted else -1,
        "accepted_first_iteration": 1 if fired_accepted else -1,
    }
    config = {
        "initial_conditions": "torus_A",
        "end_time": 4.0,
        "integrator": "rk3",
        "cfl": 0.3,
        "solver": "hlld",
        "eos": "ideal_gas:1.3333",
        "chart": "kerr_schild_cartesian",
        "grid": "cube",
        "run_config": "baseline",
    }
    if config_overrides:
        config.update(config_overrides)
    return {
        "convention": convention,
        "resolution": resolution,
        "config": config,
        "anchor_report": {"attempted": att, "accepted": acc},
        "anchor_first": first or default_first,
        "guards": {
            "fallback": 3,
            "freeze": 1,
            "fallback_inside_horizon": 0,
            "freeze_inside_horizon": 0,
            "replay_outcomes": {"conservative_replay": 2, "shared_redo": 1},
        },
        "conserved_initial": conserved_initial or {"mass": 10.0, "energy": 20.0},
        "conserved_final": conserved_final or {"mass": 10.0, "energy": 20.0},
        "observables": observables or {"accretion_rate": 1.5},
    }


def _pair(resolution: int, **kwargs: Any) -> list[dict[str, Any]]:
    return [
        _record("stage_input", resolution, **kwargs),
        _record("eulerian_rebuilt", resolution, **kwargs),
    ]


def test_identical_arms_have_zero_cross_arm_deltas() -> None:
    suite = compare_suite(_pair(64))
    assert suite.resolutions == (64,)
    p = suite.pairs[0]
    for bucket in ("attempted", "accepted"):
        for delta in p.receipts[bucket].values():
            assert delta.absolute == 0.0
            assert delta.relative in (0.0, None)
    for delta in p.guards.values():
        assert delta.absolute == 0.0
    assert p.conserved["mass_drift"].absolute == 0.0
    assert p.observables["accretion_rate"].absolute == 0.0
    # the emitted forms round-trip without raising.
    assert "disclaimer" in to_dict(suite)
    assert "does not select" in to_json(suite)
    assert "does not select" in to_markdown(suite)


def test_signed_cancellation_keeps_gross_budget_separate() -> None:
    # two cells inject +3 and -3: the signed net is 0, the gross budget is 6.
    acc = _receipts(1, 1, 2, 0.5, (0.0, 6.0), (0.0, 0.0), (0.0, 0.0))
    records = _pair(64, accepted=acc)
    suite = compare_suite(records)
    injected = suite.pairs[0].receipts["accepted"]
    # both arms carry the same values here, so the cross-arm deltas are zero,
    # but the signed and gross metrics remain distinct quantities.
    assert injected["injected.mass.signed"].stage_input == 0.0
    assert injected["injected.mass.gross"].stage_input == 6.0
    # they are reported under separate keys, never merged.
    assert "injected.mass.signed" in injected
    assert "injected.mass.gross" in injected


def test_rejected_attempt_keeps_attempted_and_accepted_distinct() -> None:
    attempted = _receipts(3, 3, 9, 0.2, (5.0, 5.0), (1.0, 1.0), (0.0, 0.0))
    accepted = _receipts(1, 1, 3, 0.6, (1.0, 1.0), (0.3, 0.3), (0.0, 0.0))
    records = _pair(64, attempted=attempted, accepted=accepted)
    suite = compare_suite(records)
    p = suite.pairs[0]
    assert p.receipts["attempted"]["passes"].stage_input == 3.0
    assert p.receipts["accepted"]["passes"].stage_input == 1.0
    assert (
        p.receipts["attempted"]["injected.mass.signed"].stage_input
        != p.receipts["accepted"]["injected.mass.signed"].stage_input
    )


def test_accepted_cannot_exceed_attempted() -> None:
    attempted = _receipts(1, 1, 1, 0.5, (1.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    accepted = _receipts(2, 2, 2, 0.5, (1.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    with pytest.raises(AnchorComparisonError, match="accepted passes"):
        compare_suite(_pair(64, attempted=attempted, accepted=accepted))


def test_zero_initial_mass_leaves_normalized_metric_out() -> None:
    records = _pair(
        64,
        conserved_initial={"mass": 0.0, "energy": 20.0},
        conserved_final={"mass": 0.5, "energy": 20.0},
    )
    suite = compare_suite(records)
    normalized = suite.pairs[0].normalized["accepted"]
    # no mass-normalized key exists (the denominator is zero); energy ones do.
    assert not any(k.endswith("mass.signed") for k in normalized)
    assert any("energy_segment.signed" in k for k in normalized)
    # and the un-normalized raw receipts are still present.
    assert "injected.mass.signed" in suite.pairs[0].receipts["accepted"]
    # the mass drift normalization is likewise absent, not infinite.
    assert "mass_drift" not in suite.pairs[0].conserved_normalized


def test_non_finite_input_is_rejected() -> None:
    bad = _receipts(1, 1, 1, 0.5, (float("nan"), 1.0), (0.0, 0.0), (0.0, 0.0))
    with pytest.raises(AnchorComparisonError, match="non-finite|below"):
        compare_suite(_pair(64, accepted=bad))


def test_configuration_mismatch_is_rejected() -> None:
    stage = _record("stage_input", 64)
    rebuilt = _record("eulerian_rebuilt", 64, config_overrides={"cfl": 0.4})
    with pytest.raises(AnchorComparisonError, match="config mismatch"):
        compare_suite([stage, rebuilt])


def test_duplicate_record_is_rejected() -> None:
    records = _pair(64) + [_record("stage_input", 64)]
    with pytest.raises(AnchorComparisonError, match="duplicate"):
        compare_suite(records)


def test_missing_arm_is_rejected() -> None:
    with pytest.raises(AnchorComparisonError, match="missing the eulerian_rebuilt"):
        compare_suite([_record("stage_input", 64)])


def test_first_event_sentinel_must_agree_with_fired_count() -> None:
    accepted = _receipts(1, 1, 1, 0.5, (1.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    bad_first = {
        "attempted_first_pass": -1,  # fired but sentinel: inconsistent
        "accepted_first_pass": 1,
        "accepted_first_time": 0.1,
        "accepted_first_iteration": 1,
    }
    with pytest.raises(AnchorComparisonError, match="attempted fired"):
        compare_suite(_pair(64, accepted=accepted, first=bad_first))


def test_two_resolutions_give_insufficient_convergence_evidence() -> None:
    records = _pair(32) + _pair(64)
    suite = compare_suite(records)
    assert suite.resolutions == (32, 64)
    for trend in suite.trends.values():
        assert trend.order is None
        assert "insufficient convergence evidence" in trend.order_note


def test_three_resolution_power_law_recovers_its_order() -> None:
    # a second-order sequence: the accepted injected signed mass scales as h^2,
    # i.e. proportional to 1/resolution^2. seed each arm identically so the
    # per-arm trend is the clean power law.
    records = []
    for res in (32, 64, 128):
        h = 1.0 / res
        signed = 4.0 * h * h  # value ~ h^2 -> log-log slope 2
        acc = _receipts(1, 1, 1, 0.5, (signed, abs(signed)), (0.0, 0.0), (0.0, 0.0))
        records.extend(_pair(res, accepted=acc))
    suite = compare_suite(records)
    order = suite.trends["stage_input::accepted.injected.mass.signed"].order
    assert order is not None
    assert order == pytest.approx(2.0, abs=1e-9)


def test_schema_incompatible_record_is_rejected() -> None:
    broken = _record("stage_input", 64)
    del broken["anchor_first"]
    with pytest.raises(AnchorComparisonError, match="anchor_first"):
        compare_suite([broken, _record("eulerian_rebuilt", 64)])


def test_injected_budget_exceeding_intervention_is_rejected() -> None:
    # the injected gross budget cannot exceed the intervention gross budget
    # (downstream weights lie in (0, 1]); an inflated injected ledger is bad data.
    acc = _receipts(1, 1, 1, 0.5, (1.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    acc["injected"]["mass"] = [2.0, 2.0]  # exceeds the intervention gross of 1.0
    with pytest.raises(AnchorComparisonError, match="exceeds intervention"):
        compare_suite(_pair(64, accepted=acc))


def test_observable_key_mismatch_is_rejected() -> None:
    stage = _record("stage_input", 64, observables={"accretion_rate": 1.5})
    rebuilt = _record(
        "eulerian_rebuilt", 64, observables={"accretion_rate": 1.5, "torque": 0.2}
    )
    with pytest.raises(AnchorComparisonError, match="observable key mismatch"):
        compare_suite([stage, rebuilt])


def test_wrong_arm_order_in_pair_is_rejected() -> None:
    stage = _record("stage_input", 64)
    rebuilt = _record("eulerian_rebuilt", 64)
    with pytest.raises(AnchorComparisonError, match="expected the stage_input arm"):
        compare_pair(RunRecord.from_dict(rebuilt), RunRecord.from_dict(stage))


def test_deep_copy_of_records_is_not_mutated() -> None:
    records = _pair(64)
    snapshot = copy.deepcopy(records)
    compare_suite(records)
    assert records == snapshot
