# =============================================================================
# test_gpu_page_migration.py
#
# the gpu backend allocates fields as unified/managed memory. on amd such an
# allocation only migrates onto the device if the gpu can fault on an absent
# page, which requires xnack -- disabled by default on gfx90a. a run without it
# leaves every field host-resident and reads it across the host bus at a ~24x
# throughput cost, silently.
#
# these check the enabling variable is exported before the gpu extension loads,
# that an explicit setting survives, and that the cpu path stays untouched. no
# gpu is required: the backend import is expected to fail off-device, and the
# environment is inspected regardless.
# =============================================================================

import os

from simbi.simulation.runner import _load_backend


def test_gpu_backend_enables_page_migration(monkeypatch):
    monkeypatch.delenv("HSA_XNACK", raising=False)
    _load_backend("gpu")
    assert os.environ.get("HSA_XNACK") == "1"


def test_an_explicit_setting_is_not_overridden(monkeypatch):
    # reaching the host-resident behavior on purpose has to stay possible, so a
    # value already in the environment wins.
    monkeypatch.setenv("HSA_XNACK", "0")
    _load_backend("gpu")
    assert os.environ["HSA_XNACK"] == "0"


def test_the_cpu_backend_does_not_set_it(monkeypatch):
    # nothing is allocated on a device, so the variable has no meaning here.
    monkeypatch.delenv("HSA_XNACK", raising=False)
    _load_backend("cpu")
    assert "HSA_XNACK" not in os.environ
