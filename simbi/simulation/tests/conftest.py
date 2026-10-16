# =============================================================================
# conftest.py (simulation tests)
#
# every command a simulation test runs through `subprocess.run` leads its own
# process group (`process_guard.run_reaped`): a timeout kills and reaps the
# command's descendants, a launcher's workers included, and a descendant
# outliving a finished command fails the call. a launch whose worker never
# exits therefore ends its test with no process left behind.
# =============================================================================
import subprocess

import pytest

from simbi.simulation.tests.process_guard import run_reaped


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
