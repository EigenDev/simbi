# =============================================================================
# anchor_ab.py
#
# read-only comparison harness for a completed projection-anchor A/B study: the
# two anchor conventions (`stage_input`, `eulerian_rebuilt`) run under identical
# physics and swept over resolution, compared cell for cell on their recorded
# receipts, guard activations, conserved totals, and physical observables.
#
# this module ingests data a run already produced and never invokes the solver,
# mutates an output, or selects a physically correct anchor. it validates that
# a pair of arms shares its whole configuration before comparing, rejects data
# that is missing / duplicated / non-finite / truncated / schema-incompatible,
# and reports the cross-arm evidence as machine-readable JSON and a compact
# Markdown table.
#
# input schema (one record per run; the run's serializer fills it from the
# python bindings):
#   {
#     "convention": "stage_input" | "eulerian_rebuilt",
#     "resolution": <int>,                  # the refinement sweep key (cells/dim)
#     "config": {                           # must match across the arm pair
#       "initial_conditions": <hashable>,   # an IC identity (name / hash)
#       "end_time": <float>,
#       "integrator": <str>,                # "euler" | "rk2" | "rk3"
#       "cfl": <float>,
#       "solver": <str>,
#       "eos": <hashable>,
#       "chart": <str>,                     # geometry / coordinate chart
#       "grid": <hashable>,                 # grid shape / extent identity
#       "run_config": <hashable>            # any remaining run identity
#     },
#     "anchor_report": {                    # anchor_experiment_report()
#       "attempted": <receipts>, "accepted": <receipts>
#     },
#     "anchor_first": {                     # anchor_experiment_first()
#       "attempted_first_pass": <int, -1 = none>,
#       "accepted_first_pass":  <int, -1 = none>,
#       "accepted_first_time":  <float, -1 = none>,
#       "accepted_first_iteration": <int, -1 = none>
#     },
#     "guards": {                           # guard_census() + fallback counters
#       "c2p_failures": <int>, "troubled_cells": <int>, "freezes": <int>,
#       "exterior_freezes": <int>, "retries": <int>,
#       "replay_outcomes": { <str>: <int>, ... }
#     },
#     "conserved_initial": { "mass": <float>, "energy": <float> },
#     "conserved_final":   { "mass": <float>, "energy": <float>, ... },
#     "observables": { <str>: <float>, ... }   # the agreed physical observables
#   }
#
# a <receipts> block, mirroring one tuple half of anchor_experiment_report():
#   {
#     "passes": <int>, "passes_fired": <int>, "projected_cells": <int>,
#     "min_theta": <float in [0, 1]>,
#     "intervention": { "mass": [signed, abs],
#                       "energy_segment": [signed, abs],
#                       "energy_raise":   [signed, abs] },
#     "injected":     { "mass": [signed, abs],
#                       "energy_segment": [signed, abs],
#                       "energy_raise":   [signed, abs] }
#   }
# the injected `signed` is the exact SSP-weighted conservation contribution; the
# injected `abs` is the scheme-weighted gross L1 budget, reported separately.
#
# usage:
#   from simbi.analysis.anchor_ab import compare_suite, to_json, to_markdown
#   suite = compare_suite(records)            # records: iterable of dicts
#   json_text = to_json(suite)
#   md_text = to_markdown(suite)
# =============================================================================

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

ARMS: tuple[str, str] = ("stage_input", "eulerian_rebuilt")
COMPONENTS: tuple[str, str, str] = ("mass", "energy_segment", "energy_raise")
INTEGRATORS: tuple[str, ...] = ("euler", "rk2", "rk3")
GUARD_COUNTERS: tuple[str, ...] = (
    "c2p_failures",
    "troubled_cells",
    "freezes",
    "exterior_freezes",
    "retries",
)
CONFIG_KEYS: tuple[str, ...] = (
    "initial_conditions",
    "end_time",
    "integrator",
    "cfl",
    "solver",
    "eos",
    "chart",
    "grid",
    "run_config",
)

# the sentinel the python first-event bindings use for "no such event".
FIRST_SENTINEL = -1

# a relative delta whose scale falls at or below this magnitude is reported as
# None: dividing by it turns roundoff into a spurious ratio. the raw absolute
# delta is always retained alongside.
REL_FLOOR = 1e-300

# the injected gross budget cannot exceed the intervention gross budget by more
# than this fraction (downstream weights lie in (0, 1]); a larger excess means
# the two ledgers were built inconsistently.
INJECTION_BUDGET_TOL = 1e-9

# the least count of resolutions that can support an order-of-convergence claim.
MIN_CONVERGENCE_POINTS = 3

DISCLAIMER = (
    "This harness reports evidence for comparing the projection-anchor "
    "conventions. It does not select the physically correct anchor."
)


class AnchorComparisonError(ValueError):
    """a record or a pairing that fails validation before any comparison."""


# =============================================================================
# typed records
# =============================================================================


@dataclass(frozen=True)
class SignedAbs:
    """a signed sum paired with its gross (L1) magnitude."""

    signed: float
    abs: float

    @staticmethod
    def from_pair(value: Any, where: str) -> "SignedAbs":
        if not (isinstance(value, Sequence) and not isinstance(value, (str, bytes))):
            raise AnchorComparisonError(f"{where}: expected a [signed, abs] pair")
        if len(value) != 2:
            raise AnchorComparisonError(f"{where}: expected exactly [signed, abs]")
        signed = _finite_float(value[0], f"{where}.signed")
        gross = _finite_float(value[1], f"{where}.abs")
        if gross < 0.0:
            raise AnchorComparisonError(f"{where}.abs is negative ({gross})")
        if gross + INJECTION_BUDGET_TOL < abs(signed):
            raise AnchorComparisonError(
                f"{where}: gross abs {gross} is below |signed| {abs(signed)}"
            )
        return SignedAbs(signed=signed, abs=gross)


@dataclass(frozen=True)
class Injection:
    """the mass and the two energy channels of one intervention/injection ledger."""

    mass: SignedAbs
    energy_segment: SignedAbs
    energy_raise: SignedAbs

    @staticmethod
    def from_dict(d: Any, where: str) -> "Injection":
        m = _require_mapping(d, where)
        return Injection(
            mass=SignedAbs.from_pair(_require(m, "mass", where), f"{where}.mass"),
            energy_segment=SignedAbs.from_pair(
                _require(m, "energy_segment", where), f"{where}.energy_segment"
            ),
            energy_raise=SignedAbs.from_pair(
                _require(m, "energy_raise", where), f"{where}.energy_raise"
            ),
        )

    def component(self, name: str) -> SignedAbs:
        return getattr(self, name)

    def energy_total_signed(self) -> float:
        """the combined energy injection: the exact-defect claim adds linearly."""
        return self.energy_segment.signed + self.energy_raise.signed

    def energy_total_gross(self) -> float:
        """the combined gross budget: L1 magnitudes add, they never cancel."""
        return self.energy_segment.abs + self.energy_raise.abs


@dataclass(frozen=True)
class Receipts:
    """one bucket (attempted or accepted) of the anchor report."""

    passes: int
    passes_fired: int
    projected_cells: int
    min_theta: float
    intervention: Injection
    injected: Injection

    @staticmethod
    def from_dict(d: Any, where: str) -> "Receipts":
        m = _require_mapping(d, where)
        passes = _nonneg_int(_require(m, "passes", where), f"{where}.passes")
        fired = _nonneg_int(
            _require(m, "passes_fired", where), f"{where}.passes_fired"
        )
        cells = _nonneg_int(
            _require(m, "projected_cells", where), f"{where}.projected_cells"
        )
        theta = _finite_float(_require(m, "min_theta", where), f"{where}.min_theta")
        if fired > passes:
            raise AnchorComparisonError(
                f"{where}: passes_fired {fired} exceeds passes {passes}"
            )
        if not (0.0 <= theta <= 1.0):
            raise AnchorComparisonError(
                f"{where}.min_theta {theta} outside [0, 1]"
            )
        intervention = Injection.from_dict(
            _require(m, "intervention", where), f"{where}.intervention"
        )
        injected = Injection.from_dict(
            _require(m, "injected", where), f"{where}.injected"
        )
        for c in COMPONENTS:
            iv = intervention.component(c).abs
            ij = injected.component(c).abs
            if ij > iv * (1.0 + INJECTION_BUDGET_TOL) + INJECTION_BUDGET_TOL:
                raise AnchorComparisonError(
                    f"{where}.{c}: injected gross {ij} exceeds intervention gross {iv}"
                )
        return Receipts(
            passes=passes,
            passes_fired=fired,
            projected_cells=cells,
            min_theta=theta,
            intervention=intervention,
            injected=injected,
        )


@dataclass(frozen=True)
class FirstEvents:
    """the first attempted and first accepted projection, with -1 sentinels."""

    attempted_first_pass: int
    accepted_first_pass: int
    accepted_first_time: float
    accepted_first_iteration: int

    @staticmethod
    def from_dict(d: Any, where: str) -> "FirstEvents":
        m = _require_mapping(d, where)
        return FirstEvents(
            attempted_first_pass=_sentinel_int(
                _require(m, "attempted_first_pass", where),
                f"{where}.attempted_first_pass",
            ),
            accepted_first_pass=_sentinel_int(
                _require(m, "accepted_first_pass", where),
                f"{where}.accepted_first_pass",
            ),
            accepted_first_time=_sentinel_float(
                _require(m, "accepted_first_time", where),
                f"{where}.accepted_first_time",
            ),
            accepted_first_iteration=_sentinel_int(
                _require(m, "accepted_first_iteration", where),
                f"{where}.accepted_first_iteration",
            ),
        )

    def attempted_fired(self) -> bool:
        return self.attempted_first_pass != FIRST_SENTINEL

    def accepted_fired(self) -> bool:
        return self.accepted_first_pass != FIRST_SENTINEL


@dataclass(frozen=True)
class Guards:
    """the guard-activation counters and the replay-outcome tally."""

    counters: Mapping[str, int]
    replay_outcomes: Mapping[str, int]

    @staticmethod
    def from_dict(d: Any, where: str) -> "Guards":
        m = _require_mapping(d, where)
        counters = {
            k: _nonneg_int(_require(m, k, where), f"{where}.{k}")
            for k in GUARD_COUNTERS
        }
        raw_replay = _require(m, "replay_outcomes", where)
        replay_map = _require_mapping(raw_replay, f"{where}.replay_outcomes")
        replay = {
            str(k): _nonneg_int(v, f"{where}.replay_outcomes.{k}")
            for k, v in replay_map.items()
        }
        return Guards(counters=counters, replay_outcomes=replay)


@dataclass(frozen=True)
class RunRecord:
    """one completed run of one arm at one resolution."""

    convention: str
    resolution: int
    config: Mapping[str, Any]
    attempted: Receipts
    accepted: Receipts
    first: FirstEvents
    guards: Guards
    conserved_initial: Mapping[str, float]
    conserved_final: Mapping[str, float]
    observables: Mapping[str, float]

    @staticmethod
    def from_dict(d: Any, where: str = "record") -> "RunRecord":
        m = _require_mapping(d, where)
        convention = _require(m, "convention", where)
        if convention not in ARMS:
            raise AnchorComparisonError(
                f"{where}.convention {convention!r} not one of {ARMS}"
            )
        resolution = _nonneg_int(
            _require(m, "resolution", where), f"{where}.resolution"
        )
        if resolution <= 0:
            raise AnchorComparisonError(f"{where}.resolution must be positive")
        config = _validate_config(_require(m, "config", where), f"{where}.config")
        report = _require_mapping(
            _require(m, "anchor_report", where), f"{where}.anchor_report"
        )
        attempted = Receipts.from_dict(
            _require(report, "attempted", f"{where}.anchor_report"),
            f"{where}.anchor_report.attempted",
        )
        accepted = Receipts.from_dict(
            _require(report, "accepted", f"{where}.anchor_report"),
            f"{where}.anchor_report.accepted",
        )
        if accepted.passes > attempted.passes:
            raise AnchorComparisonError(
                f"{where}: accepted passes {accepted.passes} exceed "
                f"attempted passes {attempted.passes}"
            )
        first = FirstEvents.from_dict(
            _require(m, "anchor_first", where), f"{where}.anchor_first"
        )
        _cross_check_first(attempted, accepted, first, where)
        guards = Guards.from_dict(_require(m, "guards", where), f"{where}.guards")
        conserved_initial = _finite_float_map(
            _require(m, "conserved_initial", where),
            f"{where}.conserved_initial",
            required=("mass", "energy"),
        )
        conserved_final = _finite_float_map(
            _require(m, "conserved_final", where),
            f"{where}.conserved_final",
            required=("mass", "energy"),
        )
        observables = _finite_float_map(
            _require(m, "observables", where), f"{where}.observables", required=()
        )
        return RunRecord(
            convention=convention,
            resolution=resolution,
            config=config,
            attempted=attempted,
            accepted=accepted,
            first=first,
            guards=guards,
            conserved_initial=conserved_initial,
            conserved_final=conserved_final,
            observables=observables,
        )

    def mass_drift(self) -> float:
        return self.conserved_final["mass"] - self.conserved_initial["mass"]

    def energy_drift(self) -> float:
        return self.conserved_final["energy"] - self.conserved_initial["energy"]


# =============================================================================
# comparison
# =============================================================================


@dataclass(frozen=True)
class ScalarDelta:
    """a cross-arm scalar comparison: the two values, their difference, and a
    safe relative difference (None when the scale is unresolvable)."""

    stage_input: float
    eulerian_rebuilt: float
    absolute: float
    relative: float | None

    @staticmethod
    def of(stage_value: float, rebuilt_value: float) -> "ScalarDelta":
        absolute = rebuilt_value - stage_value
        scale = max(abs(stage_value), abs(rebuilt_value))
        relative = absolute / scale if scale > REL_FLOOR else None
        return ScalarDelta(
            stage_input=stage_value,
            eulerian_rebuilt=rebuilt_value,
            absolute=absolute,
            relative=relative,
        )


@dataclass(frozen=True)
class PairComparison:
    """the full cross-arm comparison at one resolution."""

    resolution: int
    receipts: Mapping[str, Mapping[str, ScalarDelta]]
    normalized: Mapping[str, Mapping[str, ScalarDelta]]
    first_events: Mapping[str, Mapping[str, Any]]
    guards: Mapping[str, ScalarDelta]
    replay_outcomes: Mapping[str, ScalarDelta]
    conserved: Mapping[str, ScalarDelta]
    conserved_normalized: Mapping[str, ScalarDelta]
    observables: Mapping[str, ScalarDelta]


def compare_pair(stage: RunRecord, rebuilt: RunRecord) -> PairComparison:
    """compare the two arms at a shared resolution after confirming they differ
    only in the anchor convention."""
    if stage.convention != "stage_input":
        raise AnchorComparisonError(
            f"expected the stage_input arm, got {stage.convention!r}"
        )
    if rebuilt.convention != "eulerian_rebuilt":
        raise AnchorComparisonError(
            f"expected the eulerian_rebuilt arm, got {rebuilt.convention!r}"
        )
    if stage.resolution != rebuilt.resolution:
        raise AnchorComparisonError(
            f"resolution mismatch: {stage.resolution} vs {rebuilt.resolution}"
        )
    _require_config_match(stage, rebuilt)
    _require_observable_keys_match(stage, rebuilt)

    receipts = _receipt_deltas(stage, rebuilt)
    normalized = _normalized_receipt_deltas(stage, rebuilt)
    first_events = _first_event_report(stage, rebuilt)
    guards = {
        k: ScalarDelta.of(
            float(stage.guards.counters[k]), float(rebuilt.guards.counters[k])
        )
        for k in GUARD_COUNTERS
    }
    replay_outcomes = _replay_deltas(stage, rebuilt)
    conserved = {
        "mass_drift": ScalarDelta.of(stage.mass_drift(), rebuilt.mass_drift()),
        "energy_drift": ScalarDelta.of(stage.energy_drift(), rebuilt.energy_drift()),
    }
    conserved_normalized = _conserved_normalized(stage, rebuilt)
    observables = {
        name: ScalarDelta.of(stage.observables[name], rebuilt.observables[name])
        for name in sorted(stage.observables)
    }
    return PairComparison(
        resolution=stage.resolution,
        receipts=receipts,
        normalized=normalized,
        first_events=first_events,
        guards=guards,
        replay_outcomes=replay_outcomes,
        conserved=conserved,
        conserved_normalized=conserved_normalized,
        observables=observables,
    )


@dataclass(frozen=True)
class Trend:
    """a per-arm or cross-arm refinement trend for one scalar metric.

    `order` is the log-log least-squares slope of |value| vs mesh spacing h =
    1/resolution, populated only when at least MIN_CONVERGENCE_POINTS positive
    finite points support it; otherwise it is None with the reason stated."""

    metric: str
    resolutions: tuple[int, ...]
    values: tuple[float, ...]
    order: float | None
    order_note: str


@dataclass(frozen=True)
class SuiteComparison:
    """the whole A/B study: the per-resolution pairs and the refinement trends."""

    resolutions: tuple[int, ...]
    pairs: tuple[PairComparison, ...]
    trends: Mapping[str, Trend]
    disclaimer: str = DISCLAIMER


def compare_suite(records: Iterable[Mapping[str, Any]]) -> SuiteComparison:
    """validate every record, pair the arms by resolution, and compute the
    per-resolution comparisons and the refinement trends."""
    parsed = [
        RunRecord.from_dict(r, f"record[{ii}]") for ii, r in enumerate(records)
    ]
    if not parsed:
        raise AnchorComparisonError("no records supplied")

    by_key: dict[tuple[int, str], RunRecord] = {}
    for rec in parsed:
        key = (rec.resolution, rec.convention)
        if key in by_key:
            raise AnchorComparisonError(
                f"duplicate record for resolution {rec.resolution} "
                f"arm {rec.convention}"
            )
        by_key[key] = rec

    resolutions = sorted({res for (res, _) in by_key})
    pairs: list[PairComparison] = []
    for res in resolutions:
        stage = by_key.get((res, "stage_input"))
        rebuilt = by_key.get((res, "eulerian_rebuilt"))
        if stage is None or rebuilt is None:
            missing = "stage_input" if stage is None else "eulerian_rebuilt"
            raise AnchorComparisonError(
                f"resolution {res} is missing the {missing} arm"
            )
        pairs.append(compare_pair(stage, rebuilt))

    trends = _refinement_trends(pairs)
    return SuiteComparison(
        resolutions=tuple(resolutions),
        pairs=tuple(pairs),
        trends=trends,
    )


# =============================================================================
# receipt / normalization deltas
# =============================================================================


def _injection_scalars(receipts: Receipts) -> dict[str, float]:
    """flatten one receipts bucket into the compared scalar metrics."""
    out: dict[str, float] = {
        "passes": float(receipts.passes),
        "passes_fired": float(receipts.passes_fired),
        "projected_cells": float(receipts.projected_cells),
        "min_theta": receipts.min_theta,
    }
    for ledger_name, ledger in (
        ("intervention", receipts.intervention),
        ("injected", receipts.injected),
    ):
        for c in COMPONENTS:
            sa = ledger.component(c)
            out[f"{ledger_name}.{c}.signed"] = sa.signed
            out[f"{ledger_name}.{c}.gross"] = sa.abs
        out[f"{ledger_name}.energy_total.signed"] = ledger.energy_total_signed()
        out[f"{ledger_name}.energy_total.gross"] = ledger.energy_total_gross()
    return out


def _receipt_deltas(
    stage: RunRecord, rebuilt: RunRecord
) -> dict[str, dict[str, ScalarDelta]]:
    out: dict[str, dict[str, ScalarDelta]] = {}
    for bucket in ("attempted", "accepted"):
        s = _injection_scalars(getattr(stage, bucket))
        r = _injection_scalars(getattr(rebuilt, bucket))
        out[bucket] = {k: ScalarDelta.of(s[k], r[k]) for k in s}
    return out


def _normalized_receipt_deltas(
    stage: RunRecord, rebuilt: RunRecord
) -> dict[str, dict[str, ScalarDelta]]:
    """the injection ledgers normalized by each arm's own initial mass/energy,
    where that denominator is meaningful. mass channels divide by initial mass,
    energy channels by initial energy; counts and min_theta stay raw and are
    absent here."""
    out: dict[str, dict[str, ScalarDelta]] = {}
    for bucket in ("attempted", "accepted"):
        s_rec = getattr(stage, bucket)
        r_rec = getattr(rebuilt, bucket)
        deltas: dict[str, ScalarDelta] = {}
        for ledger_name in ("intervention", "injected"):
            s_led = getattr(s_rec, ledger_name)
            r_led = getattr(r_rec, ledger_name)
            _add_normalized_component(
                deltas, f"{ledger_name}.mass", s_led.mass, r_led.mass,
                stage.conserved_initial["mass"], rebuilt.conserved_initial["mass"],
            )
            for c in ("energy_segment", "energy_raise"):
                _add_normalized_component(
                    deltas, f"{ledger_name}.{c}",
                    s_led.component(c), r_led.component(c),
                    stage.conserved_initial["energy"],
                    rebuilt.conserved_initial["energy"],
                )
            _add_normalized_scalar(
                deltas, f"{ledger_name}.energy_total.signed",
                s_led.energy_total_signed(), r_led.energy_total_signed(),
                stage.conserved_initial["energy"],
                rebuilt.conserved_initial["energy"],
            )
            _add_normalized_scalar(
                deltas, f"{ledger_name}.energy_total.gross",
                s_led.energy_total_gross(), r_led.energy_total_gross(),
                stage.conserved_initial["energy"],
                rebuilt.conserved_initial["energy"],
            )
        out[bucket] = deltas
    return out


def _add_normalized_component(
    out: dict[str, ScalarDelta],
    prefix: str,
    s: SignedAbs,
    r: SignedAbs,
    s_denom: float,
    r_denom: float,
) -> None:
    _add_normalized_scalar(out, f"{prefix}.signed", s.signed, r.signed, s_denom, r_denom)
    _add_normalized_scalar(out, f"{prefix}.gross", s.abs, r.abs, s_denom, r_denom)


def _add_normalized_scalar(
    out: dict[str, ScalarDelta],
    key: str,
    s_value: float,
    r_value: float,
    s_denom: float,
    r_denom: float,
) -> None:
    """add a normalized delta only when both denominators are resolvable; a
    zero (or near-zero) initial mass/energy leaves the metric out rather than
    manufacturing a ratio."""
    if abs(s_denom) <= REL_FLOOR or abs(r_denom) <= REL_FLOOR:
        return
    out[key] = ScalarDelta.of(s_value / s_denom, r_value / r_denom)


def _conserved_normalized(
    stage: RunRecord, rebuilt: RunRecord
) -> dict[str, ScalarDelta]:
    out: dict[str, ScalarDelta] = {}
    _add_normalized_scalar(
        out, "mass_drift", stage.mass_drift(), rebuilt.mass_drift(),
        stage.conserved_initial["mass"], rebuilt.conserved_initial["mass"],
    )
    _add_normalized_scalar(
        out, "energy_drift", stage.energy_drift(), rebuilt.energy_drift(),
        stage.conserved_initial["energy"], rebuilt.conserved_initial["energy"],
    )
    return out


def _first_event_report(
    stage: RunRecord, rebuilt: RunRecord
) -> dict[str, dict[str, Any]]:
    def arm(rec: RunRecord) -> dict[str, Any]:
        f = rec.first
        return {
            "attempted_fired": f.attempted_fired(),
            "attempted_first_pass": (
                f.attempted_first_pass if f.attempted_fired() else None
            ),
            "accepted_fired": f.accepted_fired(),
            "accepted_first_pass": (
                f.accepted_first_pass if f.accepted_fired() else None
            ),
            "accepted_first_time": (
                f.accepted_first_time if f.accepted_fired() else None
            ),
            "accepted_first_iteration": (
                f.accepted_first_iteration if f.accepted_fired() else None
            ),
        }

    return {"stage_input": arm(stage), "eulerian_rebuilt": arm(rebuilt)}


def _replay_deltas(stage: RunRecord, rebuilt: RunRecord) -> dict[str, ScalarDelta]:
    keys = sorted(
        set(stage.guards.replay_outcomes) | set(rebuilt.guards.replay_outcomes)
    )
    return {
        k: ScalarDelta.of(
            float(stage.guards.replay_outcomes.get(k, 0)),
            float(rebuilt.guards.replay_outcomes.get(k, 0)),
        )
        for k in keys
    }


# =============================================================================
# refinement trends
# =============================================================================


def _refinement_trends(pairs: Sequence[PairComparison]) -> dict[str, Trend]:
    """per-arm and cross-arm refinement trends for the accepted injected signed
    and gross budgets, the conserved drifts, and the observables."""
    resolutions = [p.resolution for p in pairs]
    trends: dict[str, Trend] = {}

    def add(
        metric: str, series_stage: Sequence[float], series_rebuilt: Sequence[float]
    ) -> None:
        trends[f"stage_input::{metric}"] = _trend(
            f"stage_input::{metric}", resolutions, series_stage
        )
        trends[f"eulerian_rebuilt::{metric}"] = _trend(
            f"eulerian_rebuilt::{metric}", resolutions, series_rebuilt
        )
        cross = [r - s for s, r in zip(series_stage, series_rebuilt)]
        trends[f"cross::{metric}"] = _trend(f"cross::{metric}", resolutions, cross)

    def receipt_series(metric_key: str) -> tuple[list[float], list[float]]:
        s = [p.receipts["accepted"][metric_key].stage_input for p in pairs]
        r = [p.receipts["accepted"][metric_key].eulerian_rebuilt for p in pairs]
        return s, r

    for c in COMPONENTS:
        for kind in ("signed", "gross"):
            key = f"injected.{c}.{kind}"
            s, r = receipt_series(key)
            add(f"accepted.{key}", s, r)
    for tot in ("signed", "gross"):
        key = f"injected.energy_total.{tot}"
        s, r = receipt_series(key)
        add(f"accepted.{key}", s, r)

    for drift in ("mass_drift", "energy_drift"):
        s = [p.conserved[drift].stage_input for p in pairs]
        r = [p.conserved[drift].eulerian_rebuilt for p in pairs]
        add(f"conserved.{drift}", s, r)

    observable_names = sorted(pairs[0].observables) if pairs else []
    for name in observable_names:
        s = [p.observables[name].stage_input for p in pairs]
        r = [p.observables[name].eulerian_rebuilt for p in pairs]
        add(f"observable.{name}", s, r)

    return trends


def _trend(metric: str, resolutions: Sequence[int], values: Sequence[float]) -> Trend:
    order, note = _convergence_order(resolutions, values)
    return Trend(
        metric=metric,
        resolutions=tuple(resolutions),
        values=tuple(values),
        order=order,
        order_note=note,
    )


def _convergence_order(
    resolutions: Sequence[int], values: Sequence[float]
) -> tuple[float | None, str]:
    """the observed order of convergence: the least-squares slope of log|value|
    against log(mesh spacing h = 1/resolution). only claimed when at least
    MIN_CONVERGENCE_POINTS points have positive, finite magnitude."""
    points = [
        (math.log(1.0 / res), math.log(abs(v)))
        for res, v in zip(resolutions, values)
        if math.isfinite(v) and abs(v) > REL_FLOOR and res > 0
    ]
    if len(points) < MIN_CONVERGENCE_POINTS:
        return None, (
            f"insufficient convergence evidence: "
            f"{len(points)} usable point(s) < {MIN_CONVERGENCE_POINTS}"
        )
    slope = _least_squares_slope(points)
    if slope is None:
        return None, "degenerate spacing: no spread in resolution"
    return slope, f"log-log slope over {len(points)} resolutions"


def _least_squares_slope(points: Sequence[tuple[float, float]]) -> float | None:
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    if abs(denom) <= REL_FLOOR:
        return None
    return (n * sxy - sx * sy) / denom


# =============================================================================
# serialization
# =============================================================================


def to_dict(suite: SuiteComparison) -> dict[str, Any]:
    """the machine-readable comparison as a plain nested dict."""
    return {
        "disclaimer": suite.disclaimer,
        "resolutions": list(suite.resolutions),
        "pairs": [_pair_to_dict(p) for p in suite.pairs],
        "trends": {name: _trend_to_dict(t) for name, t in suite.trends.items()},
    }


def to_json(suite: SuiteComparison, indent: int = 2) -> str:
    return json.dumps(to_dict(suite), indent=indent, sort_keys=True)


def _delta_to_dict(d: ScalarDelta) -> dict[str, Any]:
    return {
        "stage_input": d.stage_input,
        "eulerian_rebuilt": d.eulerian_rebuilt,
        "absolute": d.absolute,
        "relative": d.relative,
    }


def _delta_map(m: Mapping[str, ScalarDelta]) -> dict[str, Any]:
    return {k: _delta_to_dict(v) for k, v in m.items()}


def _pair_to_dict(p: PairComparison) -> dict[str, Any]:
    return {
        "resolution": p.resolution,
        "receipts": {b: _delta_map(m) for b, m in p.receipts.items()},
        "normalized": {b: _delta_map(m) for b, m in p.normalized.items()},
        "first_events": {k: dict(v) for k, v in p.first_events.items()},
        "guards": _delta_map(p.guards),
        "replay_outcomes": _delta_map(p.replay_outcomes),
        "conserved": _delta_map(p.conserved),
        "conserved_normalized": _delta_map(p.conserved_normalized),
        "observables": _delta_map(p.observables),
    }


def _trend_to_dict(t: Trend) -> dict[str, Any]:
    return {
        "metric": t.metric,
        "resolutions": list(t.resolutions),
        "values": list(t.values),
        "order": t.order,
        "order_note": t.order_note,
    }


def to_markdown(suite: SuiteComparison) -> str:
    """a compact human-readable table of the headline cross-arm evidence."""
    lines: list[str] = []
    lines.append("# projection-anchor A/B comparison")
    lines.append("")
    lines.append(f"_{suite.disclaimer}_")
    lines.append("")

    lines.append("## accepted injection (cross-arm: rebuilt - stage_input)")
    lines.append("")
    header = (
        "| resolution | signed mass | gross mass | signed energy | gross energy |"
    )
    lines.append(header)
    lines.append("| --- | --- | --- | --- | --- |")
    for p in suite.pairs:
        acc = p.receipts["accepted"]
        row = (
            f"| {p.resolution} "
            f"| {_fmt(acc['injected.mass.signed'].absolute)} "
            f"| {_fmt(acc['injected.mass.gross'].absolute)} "
            f"| {_fmt(acc['injected.energy_total.signed'].absolute)} "
            f"| {_fmt(acc['injected.energy_total.gross'].absolute)} |"
        )
        lines.append(row)
    lines.append("")

    lines.append("## conserved drift (cross-arm absolute; relative in parentheses)")
    lines.append("")
    lines.append("| resolution | mass drift | energy drift |")
    lines.append("| --- | --- | --- |")
    for p in suite.pairs:
        md = p.conserved["mass_drift"]
        ed = p.conserved["energy_drift"]
        lines.append(
            f"| {p.resolution} | {_fmt_rel(md)} | {_fmt_rel(ed)} |"
        )
    lines.append("")

    lines.append("## guard activations (cross-arm: rebuilt - stage_input)")
    lines.append("")
    lines.append("| resolution | " + " | ".join(GUARD_COUNTERS) + " |")
    lines.append("| --- |" + " --- |" * len(GUARD_COUNTERS))
    for p in suite.pairs:
        cells = " | ".join(_fmt(p.guards[k].absolute) for k in GUARD_COUNTERS)
        lines.append(f"| {p.resolution} | {cells} |")
    lines.append("")

    lines.append("## refinement (order claimed only with >= 3 resolutions)")
    lines.append("")
    lines.append("| metric | order | note |")
    lines.append("| --- | --- | --- |")
    for name in sorted(suite.trends):
        t = suite.trends[name]
        order = "n/a" if t.order is None else f"{t.order:.3f}"
        lines.append(f"| {name} | {order} | {t.order_note} |")
    lines.append("")

    return "\n".join(lines)


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.6e}"


def _fmt_rel(d: ScalarDelta) -> str:
    rel = "n/a" if d.relative is None else f"{d.relative:.3e}"
    return f"{_fmt(d.absolute)} ({rel})"


# =============================================================================
# validation helpers
# =============================================================================


def _require(m: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in m:
        raise AnchorComparisonError(f"{where}: missing required field {key!r}")
    return m[key]


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnchorComparisonError(f"{where}: expected an object")
    return value


def _finite_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnchorComparisonError(f"{where}: expected a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise AnchorComparisonError(f"{where}: non-finite value {out}")
    return out


def _nonneg_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnchorComparisonError(f"{where}: expected an integer, got {value!r}")
    if value < 0:
        raise AnchorComparisonError(f"{where}: negative count {value}")
    return value


def _sentinel_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnchorComparisonError(f"{where}: expected an integer, got {value!r}")
    if value < 0 and value != FIRST_SENTINEL:
        raise AnchorComparisonError(
            f"{where}: negative value {value} is not the {FIRST_SENTINEL} sentinel"
        )
    return value


def _sentinel_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnchorComparisonError(f"{where}: expected a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise AnchorComparisonError(f"{where}: non-finite value {out}")
    return out


def _finite_float_map(
    value: Any, where: str, required: Sequence[str]
) -> dict[str, float]:
    m = _require_mapping(value, where)
    out = {str(k): _finite_float(v, f"{where}.{k}") for k, v in m.items()}
    for key in required:
        if key not in out:
            raise AnchorComparisonError(f"{where}: missing required field {key!r}")
    return out


def _validate_config(value: Any, where: str) -> dict[str, Any]:
    m = _require_mapping(value, where)
    for key in CONFIG_KEYS:
        if key not in m:
            raise AnchorComparisonError(f"{where}: missing required field {key!r}")
    integrator = m["integrator"]
    if integrator not in INTEGRATORS:
        raise AnchorComparisonError(
            f"{where}.integrator {integrator!r} not one of {INTEGRATORS}"
        )
    _finite_float(m["end_time"], f"{where}.end_time")
    _finite_float(m["cfl"], f"{where}.cfl")
    return dict(m)


def _require_config_match(stage: RunRecord, rebuilt: RunRecord) -> None:
    for key in CONFIG_KEYS:
        s_val = stage.config[key]
        r_val = rebuilt.config[key]
        if s_val != r_val:
            raise AnchorComparisonError(
                f"config mismatch at resolution {stage.resolution}: "
                f"{key} = {s_val!r} (stage_input) vs {r_val!r} (eulerian_rebuilt)"
            )
    if set(stage.conserved_initial) != set(rebuilt.conserved_initial):
        raise AnchorComparisonError(
            f"conserved_initial key mismatch at resolution {stage.resolution}"
        )


def _require_observable_keys_match(stage: RunRecord, rebuilt: RunRecord) -> None:
    if set(stage.observables) != set(rebuilt.observables):
        raise AnchorComparisonError(
            f"observable key mismatch at resolution {stage.resolution}: "
            f"{sorted(stage.observables)} vs {sorted(rebuilt.observables)}"
        )


def _cross_check_first(
    attempted: Receipts,
    accepted: Receipts,
    first: FirstEvents,
    where: str,
) -> None:
    """the first-event sentinels must agree with the fired counts: a bucket that
    fired names its first pass, one that did not carries the sentinel."""
    if (attempted.passes_fired > 0) != first.attempted_fired():
        raise AnchorComparisonError(
            f"{where}: attempted fired {attempted.passes_fired} disagrees with "
            f"attempted_first_pass sentinel"
        )
    if (accepted.passes_fired > 0) != first.accepted_fired():
        raise AnchorComparisonError(
            f"{where}: accepted fired {accepted.passes_fired} disagrees with "
            f"accepted-first sentinels"
        )
    if first.accepted_fired():
        if first.accepted_first_time == FIRST_SENTINEL or first.accepted_first_time < 0.0:
            raise AnchorComparisonError(
                f"{where}: accepted fired but accepted_first_time is a sentinel"
            )
        if first.accepted_first_iteration == FIRST_SENTINEL:
            raise AnchorComparisonError(
                f"{where}: accepted fired but accepted_first_iteration is a sentinel"
            )


# =============================================================================
# command-line entry
# =============================================================================


def _load_records(path: str) -> list[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping):
        records = data.get("records")
        if records is None:
            raise AnchorComparisonError(
                f"{path}: expected a 'records' list or a top-level list"
            )
    else:
        records = data
    if not isinstance(records, list):
        raise AnchorComparisonError(f"{path}: 'records' must be a list")
    return records


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="read-only projection-anchor A/B comparison; reports evidence, "
        "does not select the physically correct anchor."
    )
    parser.add_argument("manifest", help="JSON file: a list of run records, or {records: [...]}")
    parser.add_argument("--json", dest="json_out", help="write the JSON comparison here")
    parser.add_argument("--md", dest="md_out", help="write the Markdown table here")
    args = parser.parse_args(argv)

    suite = compare_suite(_load_records(args.manifest))
    json_text = to_json(suite)
    md_text = to_markdown(suite)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(json_text)
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as handle:
            handle.write(md_text)
    if not args.json_out and not args.md_out:
        print(md_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
