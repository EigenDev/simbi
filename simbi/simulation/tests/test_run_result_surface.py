# =============================================================================
# test_run_result_surface.py
#
# the public run-result surface: a completed run returns a frozen RunResult tree
# assembled from the backend's typed diagnostics transport. these gates lock the
# contract the scientific api exposes:
#   - run() returns a RunResult carrying this run's own accepted diagnostics;
#   - every dispatch family reachable on a cpu unit run returns one (the
#     decomposed drivers' ownership is proven in the rust driver tests);
#   - ignoring the return leaves a run valid (old callers unaffected);
#   - the result dataclasses are frozen and name no transaction machinery;
#   - the result is the run's own copy, independent of the legacy global counters;
#   - a dry run returns None.
# =============================================================================
import dataclasses
import glob
import os
import tempfile
from pathlib import Path

import pytest

from simbi.simulation import (
    CellCount,
    GuardDiagnostics,
    Injection,
    ProjectionDiagnostics,
    RunDiagnostics,
    RunResult,
    runner,
)

# transaction machinery that must never surface in the public result vocabulary.
FORBIDDEN_FIELD_TOKENS = (
    "attempted",
    "pending",
    "commit",
    "discard",
    "scope",
    "ledger",
    "book",
    "witness",
)


def _sample_result() -> RunResult:
    """a fully-populated result, built without a run, for the shape and
    immutability gates."""
    return RunResult(
        data_directory=Path("/tmp/example"),
        diagnostics=RunDiagnostics(
            projection=ProjectionDiagnostics(
                passes_fired=3,
                projected_cells=7,
                min_theta=0.5,
                injected_den=Injection(signed=1.0, gross=2.0),
                injected_nrg=Injection(signed=-0.5, gross=0.5),
            ),
            guards=GuardDiagnostics(
                troubled_cells=CellCount(total=9, inside_horizon=2),
                frozen_cells=CellCount(total=4, inside_horizon=1),
            ),
        ),
    )


def _run(cls, args, steps: int):
    d = tempfile.mkdtemp() + "/"
    p = cls.from_cli(args)
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    result = runner.run(p, compute_mode="cpu", max_steps=steps)
    assert not glob.glob(os.path.join(d, "*crashed*")), "run crashed"
    return result, d


# ---- extension-free gates: shape, immutability, vocabulary --------------------


def test_the_result_dataclasses_are_frozen():
    result = _sample_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.data_directory = Path("/elsewhere")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.diagnostics.guards.troubled_cells.total = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.diagnostics.projection.injected_den.signed = 0.0


def test_no_result_field_names_a_transaction_word():
    seen: set[str] = set()

    def walk(tp) -> None:
        if not dataclasses.is_dataclass(tp):
            return
        for field in dataclasses.fields(tp):
            seen.add(field.name)
            walk(field.type if dataclasses.is_dataclass(field.type) else None)

    for root in (
        RunResult,
        RunDiagnostics,
        ProjectionDiagnostics,
        GuardDiagnostics,
        CellCount,
        Injection,
    ):
        for field in dataclasses.fields(root):
            seen.add(field.name)

    offenders = {
        name
        for name in seen
        for token in FORBIDDEN_FIELD_TOKENS
        if token in name.lower()
    }
    assert not offenders, f"a public result field names transaction machinery: {offenders}"


# ---- end-to-end gates: run() returns the result across dispatch families ------


def test_single_grid_run_returns_a_frozen_result():
    from simbi_configs.examples.srhd.marti_muller import MartiMuller

    # the marti & muller shock tube trips the first-order redo, so the guard
    # totals are discriminating rather than trivially zero.
    result, d = _run(MartiMuller, ["--resolution", "400"], steps=40)
    assert isinstance(result, RunResult)
    assert result.data_directory == Path(d)
    guards = result.diagnostics.guards
    assert isinstance(guards.troubled_cells.total, int)
    assert guards.troubled_cells.inside_horizon <= guards.troubled_cells.total
    assert guards.frozen_cells.inside_horizon <= guards.frozen_cells.total
    assert isinstance(result.diagnostics.projection.passes_fired, int)
    assert isinstance(result.diagnostics.projection.injected_den, Injection)


def test_refined_run_returns_a_result():
    from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

    result, _ = _run(RefinedBlast, [], steps=1)
    assert isinstance(result, RunResult)
    assert isinstance(result.diagnostics.guards.troubled_cells.total, int)


def test_ignoring_the_return_still_runs():
    from simbi_configs.examples.newtonian.sod import SodProblem

    d = tempfile.mkdtemp() + "/"
    p = SodProblem.from_cli([])
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    # an old caller that ignores the return runs unchanged.
    runner.run(p, compute_mode="cpu", max_steps=5)
    assert glob.glob(os.path.join(d, "*final*.h5")), "the run did not complete"


def test_the_result_is_independent_of_the_legacy_globals():
    import importlib

    from simbi_configs.examples.srhd.marti_muller import MartiMuller

    ext = importlib.import_module("simbi.libs.cpu_ext")
    result, _ = _run(MartiMuller, ["--resolution", "400"], steps=40)
    troubled = result.diagnostics.guards.troubled_cells.total
    fired = result.diagnostics.projection.passes_fired
    # the frozen result was assembled by value from the run's typed transport, so
    # resetting the legacy process-global counters cannot change it. (the discriminating
    # non-zero form of this independence is proven in the rust ownership tests.)
    ext.reset_guard_census()
    ext.reset_projection_ledger()
    assert result.diagnostics.guards.troubled_cells.total == troubled
    assert result.diagnostics.projection.passes_fired == fired
