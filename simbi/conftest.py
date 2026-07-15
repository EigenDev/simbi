# =============================================================================
# conftest.py
#
# splits the test suite into a fast default set (pure-python logic) and an
# opt-in `simulation` set (evolves a real grid through the rust cpu_ext driver).
# a test is a simulation iff its module drives the solver -- it calls
# `runner.run(...)` / `.simulate(...)` -- or is gated on the built backend (a
# skipif whose reason names the unbuilt cpu_ext backend). such tests are
# auto-tagged `simulation` and excluded from the default run (pyproject sets
# addopts = -m "not simulation"). run the physics suite with `pytest -m
# simulation`, or everything with `pytest -m ""`.
#
# the driver-call signal is self-maintaining: a new evolution test calls the
# driver and is tagged automatically; a new logic test does not and stays fast.
# no hand-maintained file list to drift.
# =============================================================================
import functools
import pathlib
import re

import pytest

# the substring shared by every backend-gated test's skipif reason:
# `pytest.mark.skipif(_BACKEND is None, reason="rust cpu_ext backend not built")`.
_backend_gate_reason = "backend not built"

# a call into the solver driver: the definition of "this test evolves a grid".
_driver_call = re.compile(r"runner\.run\(|\.simulate\(|\.run\(compute")


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
