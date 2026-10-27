# =============================================================================
# test_parity_harness.py
#
# the parity harness's session selection and failure classification, read in
# fresh interpreters so each case sees the environment it names:
# - the experimental session flag is a bool, False unless
#   SIMBI_PARITY_EXPERIMENTAL=1, and the session identity records it as such
# - the experimental announcement the launch prints is matched by its own name
# - failures classify by their text: an infrastructure fault, a device-scope
#   refusal and a host-only device stop as missing implementation, an admission
#   refusal, resource exhaustion, and an execution failure otherwise
# =============================================================================
import os
import subprocess
import sys

from simbi.simulation.tests import parity_harness as ph

PROBE = (
    "from simbi.simulation.tests import parity_harness as ph\n"
    "print(type(ph.EXPERIMENTAL).__name__, ph.EXPERIMENTAL, ph.EXPERIMENTAL_ANNOUNCEMENT)\n"
)


def _probe(value: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "SIMBI_PARITY_EXPERIMENTAL"}
    if value is not None:
        env["SIMBI_PARITY_EXPERIMENTAL"] = value
    done = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=120, env=env)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.strip()


def test_the_experimental_session_flag_is_a_bool_set_by_its_variable_alone() -> None:
    assert _probe(None) == "bool False experimental combination"
    assert _probe("0") == "bool False experimental combination"
    assert _probe("1") == "bool True experimental combination"


def test_a_launch_reports_the_experimental_announcement_by_its_text() -> None:
    run = ph.Launch(out=None, returncode=0, stdout="", stderr="SIMBI launch: experimental combination without a distributed gate: x", exits={})
    assert run.experimental
    assert not ph.Launch(out=None, returncode=0, stdout="", stderr="SIMBI launch: worker=0/2 steps=1", exits={}).experimental


def test_failures_classify_by_their_text() -> None:
    cases = {
        "ValueError: 3 gpu workers but ROCR_VISIBLE_DEVICES=0,1 makes 2 devices visible": ph.InfrastructureFailure,
        "workers 0 and 1 bound one physical gpu": ph.InfrastructureFailure,
        "a tracer population on device memory with mesh refinement is outside the distributed first release": ph.MissingImplementation,
        "a horizon body on device memory is outside the distributed first release": ph.MissingImplementation,
        "the magnetic-slip solve sums on host memory; its device path is outside the gated scope": ph.MissingImplementation,
        "mesh refinement on gpu device memory with 3 refined regions is installed on the launched path without a distributed gate": ph.AdmissionRefusal,
        "hipErrorOutOfMemory": ph.ResourceExhaustion,
        "No space left on device": ph.ResourceExhaustion,
        "thread 'main' panicked at c2p: no root": ph.ExecutionFailure,
    }
    for text, kind in cases.items():
        assert ph.classify(text) is kind, f"{text!r}: {ph.classify(text).__name__}, expected {kind.__name__}"


INTERRUPTED = """
import os, signal, subprocess, sys, threading
from simbi.simulation.tests import conftest  # noqa: F401  installs the session's signal handlers
from simbi.simulation.tests.process_guard import run_reaped
threading.Timer(2.0, lambda: os.kill(os.getpid(), getattr(signal, sys.argv[1]))).start()
try:
    run_reaped([sys.executable, "-c", "import subprocess, time; subprocess.Popen(['sleep', sys.argv[2]]); time.sleep(300)".replace("sys.argv[2]", repr(sys.argv[2]))],
               capture_output=True, timeout=600)
except KeyboardInterrupt:
    print("interrupted")
"""


def _alive(marker: str) -> int:
    done = subprocess.run(["pgrep", "-f", f"sleep {marker}"], capture_output=True, text=True)
    return len(done.stdout.split())


def test_a_signalled_session_reaps_the_running_commands_process_group() -> None:
    """SIGTERM (a batch's bound) and SIGINT (also from a session started with SIGINT ignored, as a
    background job is) raise KeyboardInterrupt in the waiting test, and the command's process
    group, a grandchild included, is killed and reaped before the interrupt propagates"""
    import signal

    for number, name in ((331, "SIGTERM"), (337, "SIGINT")):
        marker = str(number)
        done = subprocess.run([sys.executable, "-c", INTERRUPTED, name, marker], capture_output=True, text=True, timeout=120,
                              preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
        assert "interrupted" in done.stdout, f"{name}: {done.stdout!r} {done.stderr[-1500:]!r}"
        assert _alive(marker) == 0, f"{name}: the command's grandchild outlived the interrupt"
