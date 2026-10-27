# =============================================================================
# test_launch_viscous.py
#
# the viscous transport through `simbi launch`. the post-step recipe folds it
# once per step after the RK combination on every tile; its stencil reads the
# primitive velocity one cell into the exchanged halo and its diffusion limit
# is part of every tile's time step, which the collective minimum combines.
# the viscous shear layer on two workers equals the single-grid run, fresh and
# resumed onto three uneven workers from a checkpoint, and the launched run
# with the viscosity switched off differs from it, so the launch carries nu
# rather than dropping it. the recipe recovers the primitives after the pass
# and refreshes the halos, so a resume equals the uninterrupted run.
#
# the single-grid references and the launches run on the session's backend and
# placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_viscous.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

from pathlib import Path

import h5py
import pytest

from simbi_configs.examples.newtonian.viscous_shear import ViscousShear

from simbi.simulation.tests.test_launch import (
    STEPS,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _launch, _same_state, _script_reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation

SCRIPT = str(Path(ViscousShear.__module__.replace(".", "/") + ".py"))
RESOLUTION = (32, 32, 1)
NU = ["--nu", "0.05"]


def _reference(out: Path, steps: int) -> Path:
    return _script_reference(ViscousShear.__module__, "ViscousShear", RESOLUTION, out, steps, fields="nu=0.05")


def test_a_viscous_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    reference = _reference(tmp_path / "single", STEPS)
    two = dict(workers=2, shape=(2, 1))
    launched = _launch(SCRIPT, RESOLUTION, tmp_path / "two", STEPS, **two, flags=NU)
    _same_state(launched, reference, "two viscous workers")
    # the launch carries the viscosity: with nu switched off the run differs
    inviscid = _launch(SCRIPT, RESOLUTION, tmp_path / "inviscid", STEPS, **two, flags=["--nu", "0.0"])
    _assert_flag_changed(inviscid, launched)


def test_a_viscous_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path) -> None:
    # the recipe recovers the primitives after the viscous pass and refreshes the halos, so a
    # checkpoint at the step boundary holds one state and a resume equals the uninterrupted run:
    # first on one worker, then onto three unevenly owned tiles from a two-worker checkpoint
    reference = _reference(tmp_path / "single", STEPS)
    one = dict(workers=1, shape=(1, 1))
    partial_one = _launch(SCRIPT, RESOLUTION, tmp_path / "half", STEPS // 2, **one, flags=NU)
    with h5py.File(partial_one, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    # the cap is the absolute root iteration: the checkpoint's steps and the rest
    resumed_one = _launch(SCRIPT, RESOLUTION, tmp_path / "resumed_one", STEPS, **one, checkpoint=partial_one, flags=NU)
    _same_state(resumed_one, reference, "viscous, resumed on one worker against the uninterrupted run")
    partial = _launch(SCRIPT, RESOLUTION, tmp_path / "first", STEPS // 2, workers=2, shape=(2, 1), flags=NU)
    # the cap is the absolute root iteration: the checkpoint's steps and the rest
    resumed = _launch(SCRIPT, RESOLUTION, tmp_path / "three", STEPS, workers=3, cuts=[[10, 21], []], checkpoint=partial, flags=NU)
    _same_state(resumed, reference, "viscous, resumed on three workers against the uninterrupted run")
