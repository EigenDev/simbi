# =============================================================================
# test_relativistic_viscosity_refusal.py
#
# the relativistic regimes carry no viscous transport, so a run that
# configures a viscosity on one is refused by name before any evolution: the
# runner raises, naming the regime, and no checkpoint is written. an inviscid
# run of the same problem starts.
#
# run: pytest simbi/simulation/tests/test_relativistic_viscosity_refusal.py
# =============================================================================
import glob
import os
import tempfile

import pytest
from pydantic import computed_field

from simbi.simulation import runner
from simbi_configs.examples.srhd.ram44 import Ram44

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(_BACKEND is None, reason="rust cpu_ext backend not built")


class _ViscousRam44(Ram44):
    """the ram44 shock tube asking for a constant viscosity."""

    @computed_field
    @property
    def viscosity(self) -> float:
        return 0.01


@needs_backend
def test_a_viscous_rhd_run_is_refused_before_evolution() -> None:
    d = tempfile.mkdtemp() + "/"
    with pytest.raises(Exception, match="viscous transport is not built for the rhd regime"):
        runner.run(_ViscousRam44(data_directory=d, checkpoint_interval=1e9), compute_mode="cpu", max_steps=1)
    assert not glob.glob(os.path.join(d, "*.h5")), "a refused run wrote a checkpoint"


@needs_backend
def test_an_inviscid_rhd_run_starts() -> None:
    d = tempfile.mkdtemp() + "/"
    runner.run(Ram44(data_directory=d, checkpoint_interval=1e9), compute_mode="cpu", max_steps=1)
    assert glob.glob(os.path.join(d, "*final*.h5")), "the inviscid run did not complete its step"
