# =============================================================================
# test_run_stack.py
#
# a run's stack is the extension's own: the dispatch of a run onto its
# monomorphized driver holds a frame of several megabytes, and the run
# completes whatever stack the calling thread was given. the canonical
# example with user sources, one production step, from a thread holding the
# 8 MiB a python main thread commonly has (the macos default and most linux
# builds), in a fresh interpreter under a time limit so a stalled run fails
# the test instead of holding the suite.
# =============================================================================
import subprocess
import sys
from pathlib import Path

SNIPPET = """
import sys, threading
from simbi.simulation import runner
from simbi_configs.examples.newtonian.ordered_sources import OrderedSources
threading.stack_size(8 * 1024 * 1024)
failure = []
def run():
    try:
        runner.run(OrderedSources(data_directory=sys.argv[1]), compute_mode="cpu", max_steps=1)
    except BaseException as error:
        failure.append(repr(error))
thread = threading.Thread(target=run)
thread.start()
thread.join()
if failure:
    raise SystemExit(failure[0])
"""


def test_a_run_completes_from_a_thread_with_an_eight_mebibyte_stack(tmp_path: Path) -> None:
    out = tmp_path / "out"
    try:
        done = subprocess.run([sys.executable, "-c", SNIPPET, str(out)], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise AssertionError("one step on an 8 MiB thread stack did not finish in 120 s; the run exhausted the caller's stack") from None
    assert done.returncode == 0, f"the run failed:\n{done.stderr[-2000:]}"
    assert sorted(out.glob("*final*.h5")), "the run published no final checkpoint"
