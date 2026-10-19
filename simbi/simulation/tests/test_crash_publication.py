# =============================================================================
# test_crash_publication.py
#
# a run that halts on a state it cannot advance publishes that state as crashed
# and never as a completed run. a uniform force whose acceleration is sqrt(-1)
# turns a refined box's momentum non-finite on its first step; the first-order
# correction then freezes the same zone on every retried substage and the step
# halts. the single grid, refined host tiles and refined launched workers all
# end with a failing exit and no final checkpoint, and the in-process drivers
# write the `.crashed` snapshot of the last state.
#
# a retry guard that counts every step as severely reduced and ends the run at
# the first one refuses the first step of a clean box, refined or flat, on
# every driver: a failing exit naming the guard, no final checkpoint, and a
# `.crashed` snapshot of the step entry, the last accepted state at t = 0, the
# launched workers writing theirs together after agreeing to halt.
#
# run: pytest -m simulation simbi/simulation/tests/test_crash_publication.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

PROBE = '''
import os

from pydantic import computed_field
from typing import Annotated, Any

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver

REGIONS = [[0.25, 0.75, 0.25, 0.75]]


class Poisoned(SimbiProblem):
    """a refined quiescent box under a force whose acceleration is sqrt(-1)"""

    resolution: Annotated[tuple[int, ...], ProblemParam((16, 16), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0)] * 2, description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime("newtonian"), description="regime")]
    solver: Annotated[Solver, ProblemParam(Solver("hlle"), description="solver")]
    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    boundary_conditions: Annotated[list[Any], ProblemParam([BoundaryCondition("outflow")], description="walls")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    gpus: Annotated[int, ProblemParam(int(os.environ.get("POISON_GPUS", "1")), cli=True, description="devices")]
    refinement_enabled: Annotated[bool, ProblemParam(os.environ.get("POISON_REFINED", "1") == "1", description="refined")]
    retry_severe_ratio: Annotated[
        float,
        ProblemParam(float(os.environ.get("POISON_SEVERE_RATIO", "0.0625")), gt=0.0, le=1.0, cli=True, checkpoint_safe=True, description="severe"),
    ]
    retry_sustained_steps: Annotated[
        int,
        ProblemParam(int(os.environ.get("POISON_SUSTAINED", "8")), ge=1, cli=True, checkpoint_safe=True, description="sustained"),
    ]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam(REGIONS, description="boxes")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]

    def initial_primitive_state(self):
        def gas():
            for _ in range(16 * 16):
                yield (1.0, 0.0, 0.0, 1.0)
        return gas

    @computed_field
    @property
    def n_tracers(self) -> int:
        return int(os.environ.get("POISON_TRACERS", "0"))

    @property
    def source_expressions(self):
        if os.environ.get("POISON_CLEAN") == "1":
            return []
        return [{"kind": "force", "dim": 2, "outputs": [1, 2], "params": [],
                 "vocabulary": {"reads": [], "params": []},
                 "nodes": [{"op": "CONSTANT", "value": -1.0}, {"op": "SQRT", "left": 0},
                           {"op": "CONSTANT", "value": 0.0}]}]
'''


@pytest.fixture
def probe(tmp_path: Path) -> Path:
    path = tmp_path / "poisoned.py"
    path.write_text(PROBE)
    return path


def _env(gpus: int, refined: bool = True) -> dict:
    env = dict(os.environ, POISON_GPUS=str(gpus), POISON_REFINED="1" if refined else "0", SYMBI_GPU_OVERSUBSCRIBE="1")
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    return env


def _in_process(probe: Path, out: Path, gpus: int, env: dict | None = None) -> subprocess.CompletedProcess:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(probe.parent)!r})\n"
        "from simbi.simulation import runner\n"
        "from poisoned import Poisoned\n"
        "runner.run(Poisoned(data_directory=sys.argv[1], checkpoint_interval=1e9), compute_mode='cpu', validate=False, max_steps=5)\n"
    )
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600, env=env or _env(gpus))


def _launch(probe: Path, out: Path, tmp_path: Path, env: dict) -> subprocess.CompletedProcess:
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(probe), "--layout", str(layout), "--max-steps", "5",
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)


def _assert_mark(path: Path, restartable: bool, label: str) -> None:
    """the crash file's restartable mark: true where the halted step's entry was restored"""
    with h5py.File(path, "r") as f:
        assert "restartable" in f["metadata"].attrs, f"{label}: the crash file carries no restartable mark"
        assert bool(f["metadata"].attrs["restartable"]) is restartable, f"{label}: the crash file is marked restartable={not restartable}"


def _assert_crashed(done: subprocess.CompletedProcess, out: Path, label: str, snapshot: bool) -> None:
    flat = " ".join((done.stdout + done.stderr).split())
    assert "freeze fired on 16 consecutive substages" in flat, f"{label}: premise: the step never halted:\n{flat[-1500:]}"
    assert done.returncode != 0, f"{label}: a crashed run exited cleanly"
    assert not list(out.glob("*final*.h5")), f"{label}: a crashed run published a final checkpoint"
    if snapshot:
        crashed = list(out.glob("*crashed*.h5"))
        assert crashed, f"{label}: no crashed snapshot of the last state"
        _assert_mark(crashed[0], True, label)


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "host-tiles"])
def test_a_crashed_refined_run_publishes_its_state_as_crashed(tmp_path: Path, probe: Path, gpus: int) -> None:
    out = tmp_path / "out"
    _assert_crashed(_in_process(probe, out, gpus), out, f"refined run on {gpus} device(s)", snapshot=True)


def test_a_crashed_refined_launch_publishes_no_final_checkpoint(tmp_path: Path, probe: Path) -> None:
    out = tmp_path / "launched"
    _assert_crashed(_launch(probe, out, tmp_path, _env(1)), out, "refined launch", snapshot=False)


GUARD = "1 consecutive root steps accepted at or below 1 of their candidate timestep"


def _refusing(gpus: int, refined: bool, tracers: int = 0) -> dict:
    """a clean box under a guard that refuses its first step"""
    return dict(_env(gpus, refined), POISON_CLEAN="1", POISON_SEVERE_RATIO="1.0", POISON_SUSTAINED="1", POISON_TRACERS=str(tracers))


def _assert_guard_halt(done: subprocess.CompletedProcess, out: Path, label: str, snapshot: bool) -> None:
    flat = " ".join((done.stdout + done.stderr).split())
    assert GUARD in flat, f"{label}: the retry guard did not end the run:\n{flat[-1500:]}"
    assert done.returncode != 0, f"{label}: a run the retry guard ended exited cleanly"
    assert not list(out.glob("*final*.h5")), f"{label}: a run the retry guard ended published a final checkpoint"
    if snapshot:
        crashed = list(out.glob("*crashed*.h5"))
        assert crashed, f"{label}: no crashed snapshot of the step entry"
        _assert_mark(crashed[0], True, label)
        with h5py.File(crashed[0], "r") as f:
            clock = (int(f["level_0"].attrs["iteration"]), float(f["level_0"].attrs["time"]))
        assert clock == (0, 0.0), f"{label}: the crashed snapshot holds {clock}, past the refused step's entry"


@pytest.mark.parametrize("refined", [True, False], ids=["refined", "flat"])
@pytest.mark.parametrize("driver", ["single", "host-tiles", "launch"])
def test_a_run_the_retry_guard_ends_publishes_the_step_entry_as_crashed(tmp_path: Path, probe: Path, driver: str, refined: bool) -> None:
    out = tmp_path / "out"
    label = f"{driver}, {'refined' if refined else 'flat'}"
    if driver == "launch":
        _assert_guard_halt(_launch(probe, out, tmp_path, _refusing(1, refined)), out, label, snapshot=True)
    else:
        gpus = 1 if driver == "single" else 2
        _assert_guard_halt(_in_process(probe, out, gpus, _refusing(gpus, refined)), out, label, snapshot=True)


# a traced run restores its tracers with the step entry: the tiles and workers whose tracers crossed
# a cut during the refused step take their own records back, so the guard's halt publishes the step
# entry on every driver
@pytest.mark.parametrize("refined", [True, False], ids=["refined", "flat"])
@pytest.mark.parametrize("driver", ["single", "host-tiles", "launch"])
def test_a_traced_run_the_retry_guard_ends_publishes_the_step_entry(tmp_path: Path, probe: Path, driver: str, refined: bool) -> None:
    out = tmp_path / "out"
    label = f"traced {driver}, {'refined' if refined else 'flat'}"
    if driver == "launch":
        _assert_guard_halt(_launch(probe, out, tmp_path, _refusing(1, refined, tracers=200)), out, label, snapshot=True)
    else:
        gpus = 1 if driver == "single" else 2
        _assert_guard_halt(_in_process(probe, out, gpus, _refusing(gpus, refined, tracers=200)), out, label, snapshot=True)


@pytest.mark.parametrize("refined", [True, False], ids=["refined", "flat"])
def test_a_launch_no_timestep_can_advance_publishes_a_diagnostic_crash_file(tmp_path: Path, probe: Path, refined: bool) -> None:
    # worker 0's timestep candidate at step 2 is invalid: the workers agree to halt on an accepted
    # state no timestep advances, and its crash file serves inspection alone. the in-process
    # drivers' watchdog mark is gated in timestep_agreement.rs, since no admissible initial state
    # reaches it from here
    out = tmp_path / "out"
    label = f"launch {'refined' if refined else 'flat'}"
    env = dict(_env(1, refined), POISON_CLEAN="1", SIMBI_FABRIC_INJECT="bad_cfl:0:2")
    done = _launch(probe, out, tmp_path, env)
    flat = " ".join((done.stdout + done.stderr).split())
    assert done.returncode != 0, f"{label}: a run with no valid timestep exited cleanly"
    assert "invalid CFL candidate" in flat, f"{label}: premise: the watchdog never fired:\n{flat[-1500:]}"
    crashed = list(out.glob("*crashed*.h5"))
    assert crashed, f"{label}: no crash file:\n{flat[-1500:]}"
    _assert_mark(crashed[0], False, label)
