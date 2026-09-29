# =============================================================================
# test_ct_chart_admission.py
#
# both constrained-transport methods run on the flat cylindrical planes through
# the python runner: the rotor's swirling core on the axisymmetric (r, z) plane
# completes its step under UCT and under Contact, each writing its final
# checkpoint, and UCT runs on the (r, phi) disk. the chart admission itself is
# the kernel set's, decided from the emitted kernels at its first step.
#
# run: pytest -m simulation simbi/simulation/tests/test_ct_chart_admission.py
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
@pytest.mark.parametrize("ct", [CtMethod.UCT, CtMethod.CONTACT])
def test_both_methods_run_on_the_cylindrical_rz_plane(ct: CtMethod) -> None:
    d = tempfile.mkdtemp() + "/"
    runner.run(_rz(ct, d), compute_mode="cpu", max_steps=1)
    assert glob.glob(os.path.join(d, "*final*.h5")), f"the {ct} run did not complete its step"


@needs_backend
def test_uct_on_the_cylindrical_rphi_plane_runs() -> None:
    d = tempfile.mkdtemp() + "/"
    problem = NewtonianRotorCyl(data_directory=d, checkpoint_interval=1e9, resolution=(16, 16, 1), ct_method=CtMethod.UCT)
    runner.run(problem, compute_mode="cpu", max_steps=1)
    assert glob.glob(os.path.join(d, "*final*.h5")), "the r-phi UCT run did not complete its step"
