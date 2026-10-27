# =============================================================================
# conftest.py (simulation tests)
#
# every command a simulation test runs through `subprocess.run` leads its own
# process group (`process_guard.run_reaped`): a timeout kills and reaps the
# command's descendants, a launcher's workers included, and a descendant
# outliving a finished command fails the call. a launch whose worker never
# exits therefore ends its test with no process left behind. SIGTERM and
# SIGINT raise KeyboardInterrupt in the running test, so a batch ended at its
# bound unwinds through the same reaping.
#
# with SIMBI_PARITY_RESULTS set (a parity batch), the session records one json
# line per test: its outcome, the category of a failure (parity_harness), the
# session's backend, placement and source identity, the runner version
# (SIMBI_PARITY_RUNNER), the hosts every launch's workers started on and the
# most distinct hosts one launch spanned, the device count of every in-process
# run it made (its single-device and same-partition host-tile references), and
# its duration. the
# batch refuses a stale or dirty backend build before any test, and stops at
# an infrastructure failure: wrong device placement, storage below
# SIMBI_PARITY_MIN_FREE_GB (default 20) on the test directory's filesystem, or
# processes a test left behind.
# =============================================================================
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.process_guard import run_reaped

RESULTS = os.environ.get("SIMBI_PARITY_RESULTS")


def _interrupted(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


# a bounded batch ends its session with SIGTERM (`timeout`, or srun forwarding it): raised as
# KeyboardInterrupt inside the running test, it unwinds through `run_reaped`, which kills and reaps
# the command's process group, a launch's workers included. a session started as a background job
# inherits SIGINT ignored; the default handler is restored so an interrupt unwinds the same way
signal.signal(signal.SIGTERM, _interrupted)
signal.signal(signal.SIGINT, signal.default_int_handler)
MIN_FREE_GB = float(os.environ.get("SIMBI_PARITY_MIN_FREE_GB", "20"))
_identity: dict[str, str] = {}


@pytest.fixture(autouse=True)
def reaped_subprocesses():
    # patched and restored here rather than through `monkeypatch`: a test's own monkeypatch then
    # unwinds before the module's process guards tear down, in the order the test set them up
    original = subprocess.run
    subprocess.run = run_reaped
    try:
        yield
    finally:
        subprocess.run = original


def _write(record: dict) -> None:
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")


def pytest_collection_modifyitems(config, items):
    # an experimental discovery session runs every launch under the opt-in, so the gates that
    # assert an admission refusal test nothing there
    if not ph.EXPERIMENTAL:
        return
    skip = pytest.mark.skip(reason="an admission gate; the experimental discovery session runs every launch under the opt-in")
    for item in items:
        name = item.name
        if "refused" in name or "without_the_opt_in" in name:
            item.add_marker(skip)


def pytest_sessionstart(session):
    if not RESULTS:
        return
    _identity.update(ph.source_identity())
    stale = ph.stale_binary(_identity)
    _write({"session": time.time(), **_identity, "stale": stale})
    if stale:
        pytest.exit(f"infrastructure failure: {stale}", returncode=3)


def pytest_runtest_setup(item):
    if not RESULTS:
        return
    free = ph.free_gigabytes(Path(item.config._tmp_path_factory.getbasetemp()))
    if free < MIN_FREE_GB:
        item.session.shouldstop = f"infrastructure failure: {free:.1f} GB free under the test directory"
        raise ph.InfrastructureFailure(f"{free:.1f} GB free, below {MIN_FREE_GB} GB")


def _category(report, call) -> str:
    if report.passed:
        return "passed"
    if report.skipped:
        return "skipped"
    exc = call.excinfo.value if call.excinfo is not None else None
    if isinstance(exc, (ph.ParityFailure, ph.NumericalDisagreement)):
        return exc.category
    text = str(report.longrepr)
    if report.when == "teardown" and ("left processes behind" in text or "processes (peak" in text):
        return "infrastructure failure"
    if isinstance(exc, AssertionError):
        # a premise, completion or physical-check assertion outside the exact comparisons
        return "assertion"
    return "harness failure"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not RESULTS:
        return
    record_now = report.when == "call" or (report.when == "setup" and not report.passed) or (report.when == "teardown" and report.failed)
    if not record_now:
        return
    category = _category(report, call)
    if category == "infrastructure failure":
        item.session.shouldstop = f"infrastructure failure in {item.nodeid}"
    message = ""
    if report.failed or report.skipped:
        message = str(report.longrepr)[-1500:]
    launches = ph.observed_launch_hosts.pop(item.nodeid, [])
    references = ph.observed_reference_devices.pop(item.nodeid, [])
    _write({
        "test": item.nodeid,
        "phase": report.when,
        "outcome": report.outcome,
        "category": category,
        "duration": round(report.duration, 2),
        "message": message,
        "runner": os.environ.get("SIMBI_PARITY_RUNNER", ""),
        "launch_hosts": launches,
        # the in-process runs the gate made: 1 a single device, more the same partition's host tiles
        "reference_devices": references,
        "max_hosts_per_launch": max((len(h) for h in launches), default=0),
        **{k: _identity.get(k, "") for k in ("backend", "placement", "nodes", "built", "head", "experimental")},
    })
