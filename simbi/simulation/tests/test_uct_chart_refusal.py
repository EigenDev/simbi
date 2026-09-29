# =============================================================================
# test_uct_chart_refusal.py
#
# the flat cylindrical r-z plane builds the Contact corner EMF alone, so a run
# that selects UCT constrained transport there is refused by name before any
# evolution: the runner raises, naming the chart and the method, and no
# checkpoint is written. the same problem runs under Contact, and UCT runs on
# the cylindrical r-phi plane, whose single out-of-plane edge UCT covers.
#
# run: pytest -m simulation simbi/simulation/tests/test_uct_chart_refusal.py
# =============================================================================
import glob
import os
import tempfile

import pytest

from simbi.simulation import runner
from simbi.types import CoordSystem
from simbi.types.input import CtMethod
from simbi_configs.examples.newtonian.nmhd_rotor_cyl import NewtonianRotorCyl

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(_BACKEND is None, reason="rust cpu_ext backend not built")


def _rz(ct: CtMethod, d: str) -> NewtonianRotorCyl:
    """the rotor's swirling core on the axisymmetric (r, z) plane, unmagnetized."""
    return NewtonianRotorCyl(
        data_directory=d,
        bz=0.0,
        checkpoint_interval=1e9,
        coord_system=CoordSystem.CYLINDRICAL,
        bounds=[(0.1, 1.0), (0.0, 1.0)],
        resolution=(16, 16, 1),
        ct_method=ct,
    )


@needs_backend
def test_uct_on_the_cylindrical_rz_plane_is_refused_before_evolution() -> None:
    d = tempfile.mkdtemp() + "/"
    with pytest.raises(Exception, match="UCT constrained transport is not built for the flat cylindrical r-z plane"):
        runner.run(_rz(CtMethod.UCT, d), compute_mode="cpu", max_steps=1)
    assert not glob.glob(os.path.join(d, "*.h5")), "a refused run wrote a checkpoint"


@needs_backend
def test_contact_on_the_cylindrical_rz_plane_runs() -> None:
    d = tempfile.mkdtemp() + "/"
    runner.run(_rz(CtMethod.CONTACT, d), compute_mode="cpu", max_steps=1)
    assert glob.glob(os.path.join(d, "*final*.h5")), "the Contact run did not complete its step"


@needs_backend
def test_uct_on_the_cylindrical_rphi_plane_runs() -> None:
    d = tempfile.mkdtemp() + "/"
    problem = NewtonianRotorCyl(data_directory=d, checkpoint_interval=1e9, resolution=(16, 16, 1), ct_method=CtMethod.UCT)
    runner.run(problem, compute_mode="cpu", max_steps=1)
    assert glob.glob(os.path.join(d, "*final*.h5")), "the r-phi UCT run did not complete its step"
