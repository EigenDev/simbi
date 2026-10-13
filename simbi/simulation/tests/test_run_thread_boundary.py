# =============================================================================
# test_run_thread_boundary.py
#
# the boundary between the calling python thread and the thread a run executes
# on. the backend drains every python input (the initial-state iterators, the
# scale-factor callables) under the gil, then releases it and blocks the caller
# until the solve returns; the solve itself calls no python. these gates hold
# that boundary from the outside:
#   - a running simulation stops on SIGINT within a bounded time and leaves a
#     restart checkpoint;
#   - an exception raised by the initial-state iterator surfaces to the caller
#     as itself;
#   - an exception raised by a scale-factor callable surfaces to the caller as
#     itself.
# each run is a script file (the runner records the problem class's source) in a
# fresh interpreter under a time limit, so a deadlock fails the test instead of
# holding the suite.
# =============================================================================
import signal
import subprocess
import sys
import time
from pathlib import Path

LONG_RUN = """
import sys
from simbi.simulation import runner
from simbi_configs.examples.newtonian.ordered_sources import OrderedSources
problem = OrderedSources(data_directory=sys.argv[1], resolution=(256, 256), end_time=1.0e3, checkpoint_interval=1.0e-4)
runner.run(problem, compute_mode="cpu")
"""

FAILING_ITERATOR = """
import sys
from simbi.simulation import runner
from simbi_configs.examples.newtonian.ordered_sources import OrderedSources
class FailingState(OrderedSources):
    def initial_primitive_state(self):
        cells = super().initial_primitive_state()
        def gas_state():
            for nn, cell in enumerate(cells()):
                if nn == 5:
                    raise ValueError("the initial state failed at cell 5")
                yield cell
        return gas_state
runner.run(FailingState(data_directory=sys.argv[1]), compute_mode="cpu", max_steps=1)
"""

FAILING_CALLBACK = """
import sys
from simbi.simulation import runner
from simbi_configs.examples.newtonian.uniform_sphere import UniformSphere
class FailingScale(UniformSphere):
    @property
    def scale_factor(self):
        def a(t):
            if t == self.start_time:
                raise ValueError("the scale factor failed at the start time")
            return 1.0 + self.expansion_rate * t
        return a
runner.run(FailingScale(data_directory=sys.argv[1]), compute_mode="cpu", max_steps=1)
"""

STARTUP_LIMIT_S = 120.0
STOP_LIMIT_S = 30.0


def _script(snippet: str, tmp_path: Path) -> list[str]:
    script = tmp_path / "run.py"
    script.write_text(snippet)
    return [sys.executable, str(script), str(tmp_path / "out")]


def _run_to_failure(snippet: str, tmp_path: Path) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(_script(snippet, tmp_path), capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise AssertionError("the failing run held its interpreter for 120 s; the exception never reached the caller") from None


def _raised_inside_the_backend(stderr: str) -> bool:
    # the backend consumes the python input; an exception raised before the call tests nothing here
    return "backend.run_simulation(" in stderr


def test_a_running_simulation_stops_on_sigint_within_a_bounded_time(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run = subprocess.Popen(_script(LONG_RUN, tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # two published checkpoints place the run inside its evolution loop, past the signal trap
        deadline = time.monotonic() + STARTUP_LIMIT_S
        while len(list(out.glob("*.h5"))) < 2:
            assert run.poll() is None, f"the long run exited before the signal:\n{run.stderr.read()[-2000:]}"
            assert time.monotonic() < deadline, f"the long run published no second checkpoint in {STARTUP_LIMIT_S} s"
            time.sleep(0.2)
        sent = time.monotonic()
        run.send_signal(signal.SIGINT)
        try:
            _, stderr = run.communicate(timeout=STOP_LIMIT_S)
        except subprocess.TimeoutExpired:
            raise AssertionError(f"the run ignored SIGINT for {STOP_LIMIT_S} s") from None
        stopped_after = time.monotonic() - sent
    finally:
        if run.poll() is None:
            run.kill()
            run.communicate()
    assert run.returncode == 0, f"the interrupted run exited {run.returncode} after {stopped_after:.1f} s:\n{stderr[-2000:]}"
    assert sorted(out.glob("*interrupted*.h5")), "the interrupted run wrote no restart checkpoint"
    assert not sorted(out.glob("*final*.h5")), "the run reached its end time; the signal interrupted nothing"


def test_an_initial_state_iterator_exception_reaches_the_caller(tmp_path: Path) -> None:
    done = _run_to_failure(FAILING_ITERATOR, tmp_path)
    assert done.returncode != 0, "the run succeeded on an initial state that raised"
    assert _raised_inside_the_backend(done.stderr), f"the input was consumed before the backend call:\n{done.stderr[-2000:]}"
    assert "ValueError: the initial state failed at cell 5" in done.stderr, done.stderr[-2000:]


def test_a_scale_factor_callback_exception_reaches_the_caller(tmp_path: Path) -> None:
    done = _run_to_failure(FAILING_CALLBACK, tmp_path)
    assert done.returncode != 0, "the run succeeded on a scale factor that raised"
    assert _raised_inside_the_backend(done.stderr), f"the input was consumed before the backend call:\n{done.stderr[-2000:]}"
    assert "ValueError: the scale factor failed at the start time" in done.stderr, done.stderr[-2000:]
