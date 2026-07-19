# =============================================================================
# test_example_smoke.py
#
# end-to-end wiring gate: every example config is discovered, forced to a tiny
# grid, run for a few steps through the real backend, and its final state
# checked finite and positive. this exercises the full config -> exec dict ->
# rust build -> kernel dispatch -> checkpoint path per regime/chart/dimension
# combination, so a dropped config field, an unbaked kernel, a DOF-mismatched
# write, or a dead dial surfaces as a NaN, a panic, or a crash checkpoint in
# this gate before a user's science run. unit tests cannot catch this class:
# every historical wiring bug lived between components that were each
# individually correct.
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np
import pytest

from simbi.simulation import runner
from simbi.simulation.tests.test_arity_example_sweep import (
    _example_modules,
    _problem_classes,
)

# native resolution, bounded steps: overriding `resolution` post-construction
# desyncs configs whose true dials are separate fields (nr/npolar mirrors set
# by a validator), and native is the exact surface a user runs. the step bound
# keeps the cost per config to a few grid sweeps.
_MAX_STEPS = 5


def _assert_final_state_finite(data_dir: str, name: str) -> None:
    assert not glob.glob(os.path.join(data_dir, "*crashed*")), (
        f"{name}: run crashed within {_MAX_STEPS} steps"
    )
    final = glob.glob(os.path.join(data_dir, "*final*.h5"))
    assert final, f"{name}: no final checkpoint written"
    with h5py.File(final[0]) as h:
        prim = h["level_0/partition_0/hydro/primitives"]
        for field in prim:
            vals = prim[field][:]
            assert np.isfinite(vals).all(), (
                f"{name}: non-finite {field} after {_MAX_STEPS} steps"
            )
        rho = prim["rho"][:]
        assert float(rho.min()) > 0.0, f"{name}: density went non-positive"
        if "pre" in prim:
            pre = prim["pre"][:]
            assert float(pre.min()) > 0.0, f"{name}: pressure went non-positive"


@pytest.mark.parametrize("module_name", _example_modules())
def test_example_config_steps_without_nan(module_name: str) -> None:
    classes = _problem_classes(module_name)
    if not classes:
        pytest.skip(f"{module_name} defines no SimbiProblem subclass")
    for cls in classes:
        problem = cls()
        d = tempfile.mkdtemp() + "/"
        problem.data_directory = d
        problem.checkpoint_interval = 1.0e30
        runner.run(problem, compute_mode="cpu", max_steps=_MAX_STEPS)
        _assert_final_state_finite(d, cls.__name__)
