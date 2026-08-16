# =============================================================================
# conftest.py
#
# the test-layer contract: python proves wiring, rust proves physics.
#
# a python test may drive the solver only far enough to show the plumbing connects
# -- the config validates, the intended dispatch arm is selected, a step completes,
# the checkpoint carries the expected schema. the moment an assertion is about a
# number the physics produced (a convergence order, a hold against an analytic
# solution, a conservation drift), it belongs in a rust integration test, which runs
# in-process, in parallel, and can see intermediate state a checkpoint never exposes.
#
# this is not a style preference. the python layer is a single process driving an
# out-of-process solver: it cannot assert on anything but the output file, its tests
# share one process's global state (one panic under a lock once destroyed 42
# unrelated tests here), and its expensive cases are small-grid many-step runs that
# cannot use more than about one core. rust runs 2400+ tests in three minutes for
# exactly the opposite reasons.
#
# the split is enforced, not documented: `_step_budget_guard` fails any driver call
# that is unbounded or too large. a science run cannot be written as a python test.
#
# tiers:
#   pytest                  -> pure-python logic (no solver)
#   pytest -m simulation    -> wiring, bounded by the budget below
#   pytest -m ""            -> both
# =============================================================================
import functools
import math
import pathlib
import re

import pytest

# the substring shared by every backend-gated test's skipif reason:
# `pytest.mark.skipif(_BACKEND is None, reason="rust cpu_ext backend not built")`.
_backend_gate_reason = "backend not built"

# a call into the solver driver: the definition of "this test evolves a grid".
_driver_call = re.compile(r"runner\.run\(|\.simulate\(|\.run\(compute")

# the cost ceiling for one driver call in a python test, in zone-cycles
# (interior cells x steps). the solver reports its own throughput in these units, so
# the budget is stated in the same currency the run is measured in.
#
# the step bound above is the real control; this catches the outliers a small step
# count cannot. it is deliberately generous: a few steps at a production resolution is
# cheap and worth keeping, because some config-level defects (an unfilled driven
# corner ghost) only appear on the grid the config actually declares, and shrinking
# the grid to save time would hide them. what it refuses is the thousands-of-steps
# run, which is a science campaign regardless of how small the grid is.
MAX_ZONE_CYCLES_PER_RUN = 50_000_000

# a driver call must also declare a step bound. running to an `end_time` makes the
# cost a function of the CFL condition, so it cannot be checked before the run and
# drifts silently when the physics changes the timestep.
_UNBOUNDED = (
    "a python test must bound its run: pass `max_steps=...` to the driver.\n"
    "running to an end_time makes the cost depend on the cfl timestep, which is\n"
    "unknowable before the run and drifts whenever the physics changes."
)


@functools.lru_cache(maxsize=None)
def _module_evolves(path: str) -> bool:
    try:
        return _driver_call.search(pathlib.Path(path).read_text()) is not None
    except OSError:
        return False


def _is_backend_gated(item) -> bool:
    for mark in item.iter_markers(name="skipif"):
        if _backend_gate_reason in str(mark.kwargs.get("reason", "")):
            return True
    return False


def pytest_collection_modifyitems(config, items):
    for item in items:
        module_file = getattr(item.module, "__file__", None)
        if _is_backend_gated(item) or (module_file and _module_evolves(module_file)):
            item.add_marker(pytest.mark.simulation)


def _interior_cells(problem) -> int:
    res = getattr(problem, "resolution", None)
    if res is None:
        return 0
    if isinstance(res, int):
        return res
    return int(math.prod(int(n) for n in res if n))


@pytest.fixture(autouse=True, scope="session")
def _step_budget_guard():
    """refuse a driver call that is unbounded or past the zone-cycle budget.

    the guard is what keeps the contract from drifting back. a rule that lives only
    in a comment gets violated by the next person who needs "just one longer run",
    and the suite grows an hour at a time -- which is exactly how it got there before.
    """
    from simbi.simulation import runner

    original = runner.run

    import inspect

    signature = inspect.signature(original)

    @functools.wraps(original)
    def guarded(*args, **kwargs):
        # bind through the real signature so a positionally-passed bound is seen too.
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        problem = bound.arguments["problem"]
        max_steps = bound.arguments.get("max_steps", 0)
        if not max_steps:
            raise AssertionError(_UNBOUNDED)
        cells = _interior_cells(problem)
        cost = cells * int(max_steps)
        if cost > MAX_ZONE_CYCLES_PER_RUN:
            raise AssertionError(
                f"this run costs {cost:,} zone-cycles ({cells:,} cells x {max_steps:,} "
                f"steps), past the {MAX_ZONE_CYCLES_PER_RUN:,} a python test may spend.\n"
                "python tests prove the wiring connects; a run this size is measuring\n"
                "physics, which belongs in a rust integration test (in-process, parallel,\n"
                "and able to see state a checkpoint never exposes) or in a\n"
                "simbi_configs/ campaign you launch deliberately."
            )
        return original(*args, **kwargs)

    runner.run = guarded
    try:
        yield
    finally:
        runner.run = original
